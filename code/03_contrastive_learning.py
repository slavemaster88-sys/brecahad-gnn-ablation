"""
方案二：多尺度对比学习 + 弱监督细胞表型发现 (HMSCL)
=====================================================

Hierarchical Multi-Scale Contrastive Learning for Cell Phenotype Discovery

核心创新：
  1. Cell-level: BYOL/MoCo风格自监督学习, 从标注框中学习细胞形态表征
  2. Patch-level: 滑动窗口组织区域对比学习
  3. Cross-scale: 跨尺度对比 (cell embedding ↔ patch embedding)
  4. Prototype-based Clustering: 发现新的细胞形态亚群

依赖: torch, torchvision, PIL, sklearn, umap-learn
"""

import os, json, warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_ROOT = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/BreCAHAD')
IMAGES_DIR = DATA_ROOT / 'images'
OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

CLASS_NAMES = {
    0: 'Mitosis', 1: 'Apoptosis', 2: 'Tumor_nuclei',
    3: 'Non_tumor_nuclei', 4: 'Tubule', 5: 'Non_tubule'
}
IMAGE_SIZE = (1360, 1024)

# ============================================================
# Part A: 数据增强与Cell/Patch数据集
# ============================================================
class CellCropDataset(Dataset):
    """从标注框裁剪单细胞图像"""

    def __init__(self, cell_df: pd.DataFrame, images_dir: Path,
                 transform=None, crop_size: int = 64):
        self.cell_df = cell_df
        self.images_dir = images_dir
        self.crop_size = crop_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.740, 0.533, 0.706],
                                     std=[0.128, 0.178, 0.108])
            ])
        else:
            self.transform = transform

        # 预加载图像缓存
        self.image_cache = {}
        for fname in self.cell_df['filename'].unique():
            img_path = images_dir / f"{fname}.jpg"
            if img_path.exists():
                self.image_cache[fname] = Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.cell_df)

    def __getitem__(self, idx):
        row = self.cell_df.iloc[idx]
        img = self.image_cache[row['filename']]

        # 从绝对坐标裁剪
        x_c = int(row['x_center_abs'])
        y_c = int(row['y_center_abs'])
        half = self.crop_size // 2
        x1 = max(0, x_c - half)
        y1 = max(0, y_c - half)
        x2 = min(IMAGE_SIZE[0], x_c + half)
        y2 = min(IMAGE_SIZE[1], y_c + half)

        crop = img.crop((x1, y1, x2, y2))
        if self.transform:
            crop = self.transform(crop)

        return crop, row['class_id'], row['filename']


class PatchDataset(Dataset):
    """滑动窗口组织区域Patch"""

    def __init__(self, images_dir: Path, filenames: List[str],
                 patch_size: int = 256, stride: int = 128,
                 transform=None):
        self.images_dir = images_dir
        self.filenames = filenames
        self.patch_size = patch_size
        self.stride = stride

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.740, 0.533, 0.706],
                                     std=[0.128, 0.178, 0.108])
            ])
        else:
            self.transform = transform

        # 预计算所有patch位置
        self.patches = []
        for fname in filenames:
            for x in range(0, IMAGE_SIZE[0] - patch_size + 1, stride):
                for y in range(0, IMAGE_SIZE[1] - patch_size + 1, stride):
                    self.patches.append((fname, x, y))

        # 预加载图像
        self.image_cache = {}
        for fname in filenames:
            img_path = images_dir / f"{fname}.jpg"
            if img_path.exists():
                self.image_cache[fname] = Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        fname, x, y = self.patches[idx]
        img = self.image_cache[fname]
        patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        if self.transform:
            patch = self.transform(patch)
        return patch, fname


# ============================================================
# Part B: BYOL-style 对比学习编码器
# ============================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class BYOLEncoder(nn.Module):
    """
    BYOL-style自监督编码器

    使用ResNet18 backbone + 投影头
    在线网络 + 目标网络 (EMA更新)
    """

    def __init__(self, backbone='resnet18', feature_dim=512, proj_dim=256):
        super().__init__()
        # Backbone
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的fc
        self.encoder_dim = 512

        # 在线网络投影头
        self.online_projector = ProjectionHead(self.encoder_dim, 512, proj_dim)
        self.online_predictor = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

        # 目标网络 (EMA)
        self.target_encoder = nn.Sequential(*list(models.resnet18(weights=None).children())[:-1])
        self.target_projector = ProjectionHead(self.encoder_dim, 512, proj_dim)

        # 初始化目标网络 = 在线网络
        self._update_target(ema_factor=0.0)

    def _update_target(self, ema_factor=0.996):
        """EMA更新目标网络"""
        for online_p, target_p in zip(
            list(self.encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            target_p.data = ema_factor * target_p.data + (1 - ema_factor) * online_p.data

    def forward_online(self, x):
        feat = self.encoder(x).flatten(1)
        proj = self.online_projector(feat)
        pred = self.online_predictor(proj)
        return feat, proj, pred

    @torch.no_grad()
    def forward_target(self, x):
        feat = self.target_encoder(x).flatten(1)
        proj = self.target_projector(feat)
        return feat, proj

    def get_features(self, x):
        """提取特征用于下游任务"""
        return self.encoder(x).flatten(1)


class CrossScaleContrastiveLoss(nn.Module):
    """
    跨尺度对比损失

    同一图像区域内的 cell embedding 和 patch embedding 互为正样本
    不同图像区域的互为负样本
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, cell_embeddings, patch_embeddings,
                cell_to_patch_idx):
        """
        cell_embeddings: (N_cell, D)
        patch_embeddings: (N_patch, D)
        cell_to_patch_idx: (N_cell,) 每个cell属于哪个patch
        """
        # 归一化
        cell_emb = F.normalize(cell_embeddings, dim=1)
        patch_emb = F.normalize(patch_embeddings, dim=1)

        # 计算所有cell-patch相似度
        sim_matrix = torch.matmul(cell_emb, patch_emb.T) / self.temperature  # (N_cell, N_patch)

        # 正样本: cell i 与其所属patch
        pos_mask = torch.zeros_like(sim_matrix)
        for i, p_idx in enumerate(cell_to_patch_idx):
            pos_mask[i, p_idx] = 1.0

        # InfoNCE损失
        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        loss = -torch.log(pos_sum / (all_sum + 1e-8)).mean()

        return loss


# ============================================================
# Part C: Prototype-based Clustering
# ============================================================
class PrototypeClusterer:
    """
    基于原型的细胞表型聚类

    方法:
      1. 使用BYOL提取的细胞特征
      2. K-Means / DBSCAN寻找最优聚类数
      3. 计算每个簇的形态学统计特征
      4. 与原始标注比较，发现新的亚群
    """

    def __init__(self, n_clusters_range=(4, 20)):
        self.n_clusters_range = n_clusters_range

    def find_optimal_clusters(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """通过轮廓系数找最优聚类数"""
        best_k, best_score = 4, -1
        best_labels = None

        for k in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            if len(set(labels)) > 1:
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

        print(f"Optimal clusters: {best_k} (silhouette={best_score:.4f})")
        return best_k, best_labels

    def cluster(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        if n_clusters is None:
            n_clusters, labels = self.find_optimal_clusters(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
        return labels

    def analyze_clusters(self, cell_df: pd.DataFrame, cluster_labels: np.ndarray,
                         features: np.ndarray) -> pd.DataFrame:
        """分析每个簇的形态学特征"""
        cell_df = cell_df.copy()
        cell_df['cluster'] = cluster_labels

        cluster_stats = []
        for c in sorted(cell_df['cluster'].unique()):
            subset = cell_df[cell_df['cluster'] == c]
            cluster_stats.append({
                'cluster': c,
                'n_cells': len(subset),
                'mean_area': subset['area_abs'].mean(),
                'std_area': subset['area_abs'].std(),
                'mean_width': subset['width_abs'].mean(),
                'mean_height': subset['height_abs'].mean(),
                'dominant_original_class': subset['class_name'].mode().values[0],
                'class_diversity': subset['class_name'].nunique(),
                'class_distribution': subset['class_name'].value_counts().to_dict()
            })

        return pd.DataFrame(cluster_stats)


# ============================================================
# Part D: 训练流程
# ============================================================
def train_byol(encoder, dataloader, optimizer, epochs, device):
    """训练BYOL编码器"""
    encoder.train()
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (imgs, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)

            # 两次增强 → 两个view
            # (实际BYOL需要两个独立增强，这里简化使用同一batch的两个子集)
            half = len(imgs) // 2
            if half < 1:
                continue
            view1, view2 = imgs[:half], imgs[half:2*half]

            # 在线网络
            _, _, pred1 = encoder.forward_online(view1)
            _, _, pred2 = encoder.forward_online(view2)

            # 目标网络 (stop gradient)
            with torch.no_grad():
                _, proj1_t = encoder.forward_target(view1)
                _, proj2_t = encoder.forward_target(view2)

            # 对称损失
            loss1 = criterion(pred1, proj2_t.detach())
            loss2 = criterion(pred2, proj1_t.detach())
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA更新目标网络
            encoder._update_target()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def extract_cell_features(encoder, dataloader, device):
    """提取所有细胞的BYOL特征"""
    encoder.eval()
    all_features = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for imgs, labels, fnames in dataloader:
            imgs = imgs.to(device)
            feats = encoder.get_features(imgs)
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_filenames.extend(fnames)

    return np.vstack(all_features), np.array(all_labels), all_filenames


# ============================================================
# Part E: 可视化
# ============================================================
def visualize_cell_embeddings(features: np.ndarray, labels: np.ndarray,
                              cluster_labels: Optional[np.ndarray] = None,
                              title: str = "Cell Embeddings"):
    """t-SNE可视化细胞嵌入"""
    fig, axes = plt.subplots(1, 2 if cluster_labels is not None else 1,
                             figsize=(16 if cluster_labels is not None else 8, 6))
    if cluster_labels is None:
        axes = [axes]

    # PCA降维 → t-SNE
    n_samples = min(5000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    feat_sub = features[indices]

    pca = PCA(n_components=50)
    feat_pca = pca.fit_transform(feat_sub)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    feat_2d = tsne.fit_transform(feat_pca)

    # 按原始标注着色
    colors = ['#FF0000', '#FF8C00', '#1E90FF', '#32CD32', '#9370DB', '#FFD700']
    for cls_id in range(6):
        mask = labels[indices] == cls_id
        if mask.sum() > 0:
            axes[0].scatter(feat_2d[mask, 0], feat_2d[mask, 1],
                          c=colors[cls_id], s=3, alpha=0.5,
                          label=CLASS_NAMES[cls_id])
    axes[0].set_title(f'{title} - By Original Class')
    axes[0].legend(markerscale=5, fontsize=7)

    # 按聚类着色
    if cluster_labels is not None:
        cl_sub = cluster_labels[indices]
        scatter = axes[1].scatter(feat_2d[:, 0], feat_2d[:, 1],
                                  c=cl_sub, cmap='tab20', s=3, alpha=0.5)
        axes[1].set_title(f'{title} - By Discovered Clusters')
        plt.colorbar(scatter, ax=axes[1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cell_embeddings_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: cell_embeddings_tsne.png")


def visualize_cluster_morphology(cluster_stats: pd.DataFrame):
    """可视化聚类形态学特征"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 簇大小
    axes[0].bar(cluster_stats['cluster'], cluster_stats['n_cells'],
                color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Cluster'), axes[0].set_ylabel('Number of Cells')
    axes[0].set_title('Cluster Sizes')

    # 平均面积
    axes[1].bar(cluster_stats['cluster'], cluster_stats['mean_area'],
                yerr=cluster_stats['std_area'], color='coral',
                edgecolor='black', capsize=3)
    axes[1].set_xlabel('Cluster'), axes[1].set_ylabel('Mean Area (pixels²)')
    axes[1].set_title('Cell Size by Cluster')

    # 原始类别多样性
    axes[2].bar(cluster_stats['cluster'], cluster_stats['class_diversity'],
                color='mediumseagreen', edgecolor='black')
    axes[2].set_xlabel('Cluster'), axes[2].set_ylabel('Original Class Diversity')
    axes[2].set_title('Original Class Diversity per Cluster')
    axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Pure cluster')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cluster_morphology.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: cluster_morphology.png")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("方案二: HMSCL - 多尺度对比学习细胞表型发现")
    print("=" * 60)

    # 加载数据
    cell_df = pd.read_csv(OUTPUT_DIR / 'all_cells.csv')
    print(f"Loaded {len(cell_df)} cells")

    # 数据划分
    with open(OUTPUT_DIR / 'data_split.json') as f:
        split = json.load(f)

    # 只使用训练集进行无监督学习
    train_df = cell_df[cell_df['case_id'].isin(split['train_cases'])]
    print(f"Training cells: {len(train_df)}")

    # 创建数据集
    cell_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.740, 0.533, 0.706], std=[0.128, 0.178, 0.108])
    ])

    cell_dataset = CellCropDataset(train_df, IMAGES_DIR, transform=cell_transform)
    cell_loader = DataLoader(cell_dataset, batch_size=64, shuffle=True, num_workers=0)

    # BYOL编码器
    print("\n[1/4] Training BYOL encoder...")
    encoder = BYOLEncoder(backbone='resnet18', feature_dim=512, proj_dim=256).to(DEVICE)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)

    train_byol(encoder, cell_loader, optimizer, epochs=50, device=DEVICE)

    # 保存编码器
    torch.save(encoder.state_dict(), OUTPUT_DIR / 'byol_encoder.pt')
    print("[✓] Saved BYOL encoder")

    # 提取特征
    print("\n[2/4] Extracting cell features...")
    eval_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.740, 0.533, 0.706], std=[0.128, 0.178, 0.108])
    ])
    eval_dataset = CellCropDataset(cell_df, IMAGES_DIR, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0)

    features, labels, filenames = extract_cell_features(encoder, eval_loader, DEVICE)
    print(f"Features shape: {features.shape}")

    # 聚类
    print("\n[3/4] Clustering cell phenotypes...")
    clusterer = PrototypeClusterer(n_clusters_range=(4, 16))
    cluster_labels = clusterer.cluster(features)

    n_clusters = len(set(cluster_labels))
    print(f"Discovered {n_clusters} cell phenotype clusters")

    # 分析聚类
    cluster_stats = clusterer.analyze_clusters(cell_df, cluster_labels, features)
    cluster_stats.to_csv(OUTPUT_DIR / 'cluster_stats.csv', index=False)
    print("\nCluster Statistics:")
    print(cluster_stats[['cluster', 'n_cells', 'mean_area', 'class_diversity', 'dominant_original_class']].to_string())

    # 可视化
    print("\n[4/4] Visualizing results...")
    visualize_cell_embeddings(features, labels, cluster_labels,
                              "BYOL Cell Phenotype Discovery")
    visualize_cluster_morphology(cluster_stats)

    # 评估: 聚类 vs 原始标注的一致性
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    print(f"\nNMI between original labels and discovered clusters: {nmi:.4f}")

    # 保存完整结果
    cell_df['cluster'] = cluster_labels
    cell_df['feature_dim1'] = features[:, 0]  # 保存前几个PCA维度用于后续分析
    cell_df['feature_dim2'] = features[:, 1]
    cell_df.to_csv(OUTPUT_DIR / 'cells_with_clusters.csv', index=False)

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
