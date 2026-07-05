"""
方案三：基于空间转录组学启发的细胞邻域组成分析 + 图注意力多任务学习
====================================================================

Cellular Neighborhood Composition Analysis with Multi-Task GAT

核心创新：
  1. 借鉴空间转录组学"细胞邻域"概念构建 Cell Neighborhood Composition Matrix
  2. Multi-Task GAT: 同时进行细胞分类 + 有丝分裂检测 + 腺管完整性评分
  3. Uncertainty Weighting 自动平衡多任务损失
  4. Spatial Attention 机制学习不同空间尺度的注意力
  5. SHAP可解释性分析

依赖: torch, torch_geometric, shap, networkx
"""

import os, json, warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.utils import to_networkx

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, mean_squared_error
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_ROOT = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/BreCAHAD')
OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

CLASS_NAMES = {
    0: 'Mitosis', 1: 'Apoptosis', 2: 'Tumor_nuclei',
    3: 'Non_tumor_nuclei', 4: 'Tubule', 5: 'Non_tubule'
}
N_CLASSES = 6
IMAGE_SIZE = (1360, 1024)


# ============================================================
# Part A: 细胞邻域组成分析 (CNCM)
# ============================================================
class CellularNeighborhoodAnalyzer:
    """
    构建细胞邻域组成矩阵 (CNCM)

    对每个细胞核，提取其邻域内（多个半径尺度）的细胞类型组成向量，
    以及邻域结构的拓扑特征。

    受空间转录组学中"细胞邻域"概念启发:
      - Schurch et al., Cell 2020
      - Bhate et al., Nature Methods 2022
    """

    def __init__(self, radii: List[float] = None):
        """
        radii: 邻域半径列表 (微米)
        病理图像中: 1 pixel ≈ 0.25 μm (40x magnification)
        """
        if radii is None:
            # 对应 12.5, 25, 50, 75, 100 μm
            self.radii = [50, 100, 200, 300, 400]
        else:
            self.radii = radii

    def compute_neighborhood(self, cells: List[Dict]) -> np.ndarray:
        """
        计算每个细胞的邻域组成矩阵

        返回: (n_cells, n_radii * n_classes + n_radii + 3)
        - 每个半径下的细胞类型组成 (n_radii * 6)
        - 每个半径下的细胞计数 (n_radii)
        - 邻域熵 (1)
        - 邻域异质性指数 (1)
        - 距最近异类型细胞距离 (1)
        """
        n = len(cells)
        coords = np.array([[c['x_center_abs'], c['y_center_abs']] for c in cells])
        classes = np.array([c['class_id'] for c in cells])

        n_features = len(self.radii) * (N_CLASSES + 1) + 3
        cncm = np.zeros((n, n_features), dtype=np.float32)

        if n < 2:
            return cncm

        tree = KDTree(coords)

        for i in range(n):
            feat_idx = 0

            for r in self.radii:
                # 查询邻域内所有细胞
                indices = tree.query_ball_point(coords[i], r)

                # 细胞类型组成
                type_counts = np.zeros(N_CLASSES)
                for j in indices:
                    if j != i:
                        type_counts[classes[j]] += 1

                total = len(indices) - 1  # 排除自身
                if total > 0:
                    cncm[i, feat_idx:feat_idx + N_CLASSES] = type_counts / total
                feat_idx += N_CLASSES

                # 邻域细胞数
                cncm[i, feat_idx] = total
                feat_idx += 1

            # 邻域熵 (基于最小半径的组成)
            comp = cncm[i, :N_CLASSES]
            comp_pos = comp[comp > 0]
            if len(comp_pos) > 0:
                cncm[i, feat_idx] = entropy(comp_pos)
            feat_idx += 1

            # 邻域异质性: 邻域内是否包含多种细胞类型
            min_r_indices = tree.query_ball_point(coords[i], self.radii[0])
            unique_types = len(set(classes[j] for j in min_r_indices if j != i))
            cncm[i, feat_idx] = unique_types / N_CLASSES
            feat_idx += 1

            # 距最近异类型细胞距离
            diff_type_dists = []
            for j in range(n):
                if j != i and classes[j] != classes[i]:
                    diff_type_dists.append(np.linalg.norm(coords[i] - coords[j]))
            if diff_type_dists:
                cncm[i, feat_idx] = min(diff_type_dists) / 500.0  # 归一化
            feat_idx += 1

        return cncm

    def compute_niche_entropy(self, cells: List[Dict]) -> np.ndarray:
        """计算每个细胞的微环境熵 (Niche Entropy)"""
        n = len(cells)
        coords = np.array([[c['x_center_abs'], c['y_center_abs']] for c in cells])
        classes = np.array([c['class_id'] for c in cells])

        niche_entropies = np.zeros(n)
        if n < 2:
            return niche_entropies

        tree = KDTree(coords)
        for i in range(n):
            for r in self.radii:
                indices = tree.query_ball_point(coords[i], r)
                if len(indices) > 2:
                    type_counts = np.bincount(classes[indices], minlength=N_CLASSES)
                    type_probs = type_counts / type_counts.sum()
                    type_probs = type_probs[type_probs > 0]
                    niche_entropies[i] = max(niche_entropies[i], entropy(type_probs))

        return niche_entropies


# ============================================================
# Part B: 多任务图注意力网络 (MT-GAT)
# ============================================================
class SpatialAttention(nn.Module):
    """空间注意力: 学习不同空间尺度的注意力权重"""

    def __init__(self, n_scales: int, hidden_dim: int):
        super().__init__()
        self.n_scales = n_scales
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_scales)
        )

    def forward(self, x_scales):
        """
        x_scales: (batch, n_scales, hidden_dim)
        """
        # 计算注意力权重
        attn_weights = self.attention(x_scales.mean(dim=2))  # (batch, n_scales)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权融合
        weighted = (x_scales * attn_weights.unsqueeze(-1)).sum(dim=1)
        return weighted, attn_weights


class MultiTaskGAT(nn.Module):
    """
    多任务图注意力网络

    三个任务:
      Task 1: 6类细胞分类 (主任务, CrossEntropy)
      Task 2: 有丝分裂检测 (二分类, BCE)
      Task 3: 腺管完整性评分 (回归, MSE)

    使用 Uncertainty Weighting (Kendall et al., CVPR 2018) 自动平衡损失
    """

    def __init__(self, node_dim, cncm_dim, hidden_dim=128,
                 n_classes=6, dropout=0.3):
        super().__init__()

        # 节点特征编码器 (原始特征 + CNCM特征)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim + cncm_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 多尺度GAT
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout),
            GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout),
            GATv2Conv(hidden_dim, hidden_dim // 2, heads=4, dropout=dropout),
        ])

        # 空间注意力 (融合不同GAT层的输出)
        self.spatial_attn = SpatialAttention(n_scales=3, hidden_dim=hidden_dim)

        # 任务特定头
        # Task 1: 细胞分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

        # Task 2: 有丝分裂检测 (二分类)
        self.mitosis_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Task 3: 腺管完整性回归
        self.tubule_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 可学习的任务不确定性权重 (log variance)
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_mit = nn.Parameter(torch.zeros(1))
        self.log_var_tub = nn.Parameter(torch.zeros(1))

    def forward(self, data, cncm_features):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 拼接节点特征和CNCM特征
        x_combined = torch.cat([x, cncm_features], dim=1)
        x = self.node_encoder(x_combined)

        # 多层GAT + 保存中间输出用于空间注意力
        gat_outputs = []
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            gat_outputs.append(x)

        # 空间注意力融合
        stacked = torch.stack(gat_outputs, dim=1)  # (N, n_layers, hidden_dim)
        x_fused, attn_weights = self.spatial_attn(stacked)

        # 全局特征增强 (拼接自身和邻居聚合)
        x_final = torch.cat([x_fused, x_combined[:, :x_fused.shape[1]]], dim=1)

        # 三个任务
        cls_logits = self.classifier(x_final)
        mitosis_logit = self.mitosis_detector(x_final)
        tubule_score = self.tubule_scorer(x_final)

        return {
            'classification': cls_logits,
            'mitosis': mitosis_logit,
            'tubule': tubule_score,
            'attention_weights': attn_weights
        }


class MultiTaskLoss(nn.Module):
    """
    Uncertainty Weighting 多任务损失

    L_total = L_cls / (2*σ₁²) + L_mit / (2*σ₂²) + L_tub / (2*σ₃²) + log(σ₁σ₂σ₃)

    Kendall, Gal, Cipolla. "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics." CVPR 2018.
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets, model):
        # Task 1: 分类
        cls_loss = self.ce_loss(outputs['classification'], targets['class_labels'])

        # Task 2: 有丝分裂检测
        mit_loss = self.bce_loss(
            outputs['mitosis'].squeeze(),
            targets['mitosis_labels'].float()
        )

        # Task 3: 腺管完整性
        tub_loss = self.mse_loss(
            outputs['tubule'].squeeze(),
            targets['tubule_scores'].float()
        )

        # Uncertainty weighting
        precision_cls = torch.exp(-model.log_var_cls)
        precision_mit = torch.exp(-model.log_var_mit)
        precision_tub = torch.exp(-model.log_var_tub)

        total_loss = (
            precision_cls * cls_loss + model.log_var_cls +
            precision_mit * mit_loss + model.log_var_mit +
            precision_tub * tub_loss + model.log_var_tub
        )

        return total_loss, {
            'cls_loss': cls_loss.item(),
            'mit_loss': mit_loss.item(),
            'tub_loss': tub_loss.item(),
            'sigma_cls': torch.exp(model.log_var_cls).item(),
            'sigma_mit': torch.exp(model.log_var_mit).item(),
            'sigma_tub': torch.exp(model.log_var_tub).item()
        }


# ============================================================
# Part C: 数据集
# ============================================================
class MultiTaskCellDataset(Dataset):
    """多任务细胞数据集"""

    def __init__(self, cell_df, case_list, neighborhood_analyzer):
        super().__init__()
        self.neighborhood_analyzer = neighborhood_analyzer

        # 按图像分组
        self.image_groups = cell_df[cell_df['case_id'].isin(case_list)].groupby('filename')
        self.filenames = list(self.image_groups.groups.keys())

        self.graphs = []
        self.cncm_features = []
        self.targets = []

        print(f"Building multi-task data for {len(self.filenames)} images...")
        for i, fname in enumerate(self.filenames):
            group = self.image_groups.get_group(fname)
            cells = group.to_dict('records')

            # 构建图 (简化版, 复用方案一的图构建)
            graph = self._build_simple_graph(cells)

            # CNCM特征
            cncm = neighborhood_analyzer.compute_neighborhood(cells)

            # 多任务目标
            class_labels = torch.tensor([c['class_id'] for c in cells], dtype=torch.long)

            # Mitosis: class_id == 0
            mitosis_labels = torch.tensor(
                [1.0 if c['class_id'] == 0 else 0.0 for c in cells],
                dtype=torch.float
            )

            # Tubule score: 邻域内Tubule比例
            tubule_scores = []
            for j, c in enumerate(cells):
                tubule_idx = 4  # Tubule class
                tubule_comp = cncm[j, tubule_idx]
                tubule_scores.append(tubule_comp)
            tubule_scores = torch.tensor(tubule_scores, dtype=torch.float)

            self.graphs.append(graph)
            self.cncm_features.append(torch.tensor(cncm, dtype=torch.float))
            self.targets.append({
                'class_labels': class_labels,
                'mitosis_labels': mitosis_labels,
                'tubule_scores': tubule_scores
            })

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(self.filenames)}")

    def _build_simple_graph(self, cells):
        """构建简化图 (仅节点特征和k-NN边)"""
        n = len(cells)
        features = np.zeros((n, 15), dtype=np.float32)
        coords = np.array([[c['x_center_abs'], c['y_center_abs']] for c in cells])

        for i, cell in enumerate(cells):
            cls_onehot = np.zeros(N_CLASSES)
            cls_onehot[cell['class_id']] = 1.0
            features[i, :6] = cls_onehot
            features[i, 6] = cell['x_center']
            features[i, 7] = cell['y_center']
            features[i, 8] = cell['width']
            features[i, 9] = cell['height']
            features[i, 10] = np.log1p(cell['area_abs']) / 15.0
            w, h = cell['width_abs'], cell['height_abs']
            features[i, 13] = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0
            features[i, 14] = cell['area_abs'] / (w * h) if (w * h) > 0 else 1.0

        # k-NN边
        edge_list = set()
        if n >= 2:
            tree = KDTree(coords)
            k = min(8, n - 1)
            if k > 0:
                _, indices = tree.query(coords, k=k + 1)
                for i in range(n):
                    for j in indices[i, 1:]:
                        edge_list.add((i, j))
                        edge_list.add((j, i))
        if not edge_list:
            edge_list = [(i, i) for i in range(n)]

        edge_index = np.array(sorted(edge_list)).T

        return Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pos=torch.tensor(coords, dtype=torch.float),
            n_nodes=n
        )

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx], self.cncm_features[idx], self.targets[idx]


# ============================================================
# Part D: 训练与评估
# ============================================================
def train_epoch_mt(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    loss_components = defaultdict(float)

    for data, cncm, targets in loader:
        data = data.to(device)
        cncm = cncm.to(device)
        targets_device = {
            'class_labels': targets['class_labels'].to(device),
            'mitosis_labels': targets['mitosis_labels'].to(device),
            'tubule_scores': targets['tubule_scores'].to(device)
        }

        optimizer.zero_grad()
        outputs = model(data, cncm)
        loss, components = loss_fn(outputs, targets_device, model)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


@torch.no_grad()
def evaluate_mt(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_cls_preds, all_cls_labels = [], []
    all_mit_preds, all_mit_labels = [], []
    all_tub_preds, all_tub_labels = [], []

    for data, cncm, targets in loader:
        data = data.to(device)
        cncm = cncm.to(device)
        targets_device = {
            'class_labels': targets['class_labels'].to(device),
            'mitosis_labels': targets['mitosis_labels'].to(device),
            'tubule_scores': targets['tubule_scores'].to(device)
        }

        outputs = model(data, cncm)
        loss, _ = loss_fn(outputs, targets_device, model)
        total_loss += loss.item()

        # Task 1
        cls_pred = outputs['classification'].argmax(dim=1).cpu().numpy()
        all_cls_preds.extend(cls_pred)
        all_cls_labels.extend(targets['class_labels'].numpy())

        # Task 2
        mit_pred = (torch.sigmoid(outputs['mitosis'].squeeze()) > 0.5).cpu().numpy().astype(int)
        all_mit_preds.extend(mit_pred)
        all_mit_labels.extend(targets['mitosis_labels'].numpy().astype(int))

        # Task 3
        all_tub_preds.extend(outputs['tubule'].squeeze().cpu().numpy())
        all_tub_labels.extend(targets['tubule_scores'].numpy())

    results = {
        'cls_acc': accuracy_score(all_cls_labels, all_cls_preds),
        'cls_f1': f1_score(all_cls_labels, all_cls_preds, average='macro', zero_division=0),
        'mit_auc': roc_auc_score(all_mit_labels, all_mit_preds) if len(set(all_mit_labels)) > 1 else 0.5,
        'mit_f1': f1_score(all_mit_labels, all_mit_preds, zero_division=0),
        'tub_rmse': np.sqrt(mean_squared_error(all_tub_labels, all_tub_preds)),
        'tub_corr': np.corrcoef(all_tub_labels, all_tub_preds)[0, 1] if len(all_tub_labels) > 1 else 0
    }
    return total_loss / len(loader), results


# ============================================================
# Part E: 可解释性分析
# ============================================================
def plot_niche_analysis(cell_df, neighborhood_analyzer):
    """可视化细胞邻域分析"""
    # 选择一张示例图像
    sample_fname = cell_df['filename'].value_counts().index[0]
    sample_cells = cell_df[cell_df['filename'] == sample_fname].to_dict('records')

    cncm = neighborhood_analyzer.compute_neighborhood(sample_cells)
    niche_entropy = neighborhood_analyzer.compute_niche_entropy(sample_cells)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    coords = np.array([[c['x_center_abs'], c['y_center_abs']] for c in sample_cells])
    classes = np.array([c['class_id'] for c in sample_cells])

    colors = ['#FF0000', '#FF8C00', '#1E90FF', '#32CD32', '#9370DB', '#FFD700']

    # 1. 原始细胞分布
    for cls_id in range(6):
        mask = classes == cls_id
        if mask.sum() > 0:
            axes[0].scatter(coords[mask, 0], coords[mask, 1],
                          c=colors[cls_id], s=5, alpha=0.6, label=CLASS_NAMES[cls_id])
    axes[0].set_title(f'Cell Distribution\n{sample_fname}')
    axes[0].invert_yaxis()
    axes[0].legend(markerscale=3, fontsize=6)

    # 2. 邻域熵
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=niche_entropy,
                         cmap='RdYlGn', s=8, alpha=0.7)
    axes[1].set_title('Niche Entropy (Microenvironment Diversity)')
    axes[1].invert_yaxis()
    plt.colorbar(sc, ax=axes[1])

    # 3. 邻域组成热力图 (前20个细胞)
    n_show = min(20, len(sample_cells))
    comp_data = cncm[:n_show, :N_CLASSES]
    sns.heatmap(comp_data, ax=axes[2], cmap='YlOrRd',
                xticklabels=[CLASS_NAMES[i] for i in range(N_CLASSES)],
                yticklabels=[f'Cell {i}' for i in range(n_show)])
    axes[2].set_title('Neighborhood Composition\n(First 20 cells)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'niche_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: niche_analysis.png")


def plot_multitask_results(history, test_results):
    """可视化多任务训练结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 训练曲线
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=1.5)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=1.5)
    axes[0, 0].set_title('Total Loss'), axes[0, 0].legend()

    axes[0, 1].plot(history['cls_f1'], label='Classification F1', linewidth=1.5)
    axes[0, 1].set_title('Task 1: Cell Classification F1')

    axes[0, 2].plot(history['mit_auc'], label='Mitosis AUC', linewidth=1.5)
    axes[0, 2].set_title('Task 2: Mitosis Detection AUC')

    axes[1, 0].plot(history['tub_rmse'], label='Tubule RMSE', linewidth=1.5, color='red')
    axes[1, 0].set_title('Task 3: Tubule Score RMSE')

    axes[1, 1].plot(history['sigma_cls'], label='σ_cls', linewidth=1)
    axes[1, 1].plot(history['sigma_mit'], label='σ_mit', linewidth=1)
    axes[1, 1].plot(history['sigma_tub'], label='σ_tub', linewidth=1)
    axes[1, 1].set_title('Task Uncertainty (σ)'), axes[1, 1].legend()

    # 测试结果条形图
    metrics = ['cls_acc', 'cls_f1', 'mit_auc', 'mit_f1', 'tub_corr']
    values = [test_results.get(m, 0) for m in metrics]
    axes[1, 2].barh(metrics, values, color='steelblue', edgecolor='black')
    axes[1, 2].set_title('Test Set Metrics')
    for i, v in enumerate(values):
        axes[1, 2].text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'multitask_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: multitask_results.png")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("方案三: MT-GAT - 多任务图注意力网络")
    print("=" * 60)

    cell_df = pd.read_csv(OUTPUT_DIR / 'all_cells.csv')
    print(f"Loaded {len(cell_df)} cells")

    with open(OUTPUT_DIR / 'data_split.json') as f:
        split = json.load(f)

    # 邻域分析器
    neighborhood_analyzer = CellularNeighborhoodAnalyzer(
        radii=[50, 100, 200, 300, 400]
    )

    # 计算CNCM特征维度
    cncm_dim = len(neighborhood_analyzer.radii) * (N_CLASSES + 1) + 3
    print(f"CNCM feature dimension: {cncm_dim}")

    # 构建数据集
    print("\nBuilding datasets...")
    train_ds = MultiTaskCellDataset(cell_df, split['train_cases'], neighborhood_analyzer)
    val_ds = MultiTaskCellDataset(cell_df, split['val_cases'], neighborhood_analyzer)
    test_ds = MultiTaskCellDataset(cell_df, split['test_cases'], neighborhood_analyzer)

    from torch_geometric.loader import DataLoader as PyGLoader
    from torch_geometric.data import Batch

    def collate_fn(batch):
        graphs, cncms, targets = zip(*batch)
        batched_graph = Batch.from_data_list(graphs)
        batched_cncm = torch.cat(cncms, dim=0)
        batched_targets = {
            'class_labels': torch.cat([t['class_labels'] for t in targets]),
            'mitosis_labels': torch.cat([t['mitosis_labels'] for t in targets]),
            'tubule_scores': torch.cat([t['tubule_scores'] for t in targets])
        }
        return batched_graph, batched_cncm, batched_targets

    train_loader = PyGLoader(
        list(zip(train_ds.graphs, train_ds.cncm_features, train_ds.targets)),
        batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    val_loader = PyGLoader(
        list(zip(val_ds.graphs, val_ds.cncm_features, val_ds.targets)),
        batch_size=4, shuffle=False, collate_fn=collate_fn
    )
    test_loader = PyGLoader(
        list(zip(test_ds.graphs, test_ds.cncm_features, test_ds.targets)),
        batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    # 模型
    model = MultiTaskGAT(
        node_dim=15, cncm_dim=cncm_dim,
        hidden_dim=128, n_classes=N_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    loss_fn = MultiTaskLoss()

    # 训练
    print("\nTraining Multi-Task GAT...")
    best_val_f1 = 0
    history = defaultdict(list)

    for epoch in range(100):
        train_loss, components = train_epoch_mt(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_results = evaluate_mt(model, val_loader, loss_fn, DEVICE)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['cls_f1'].append(val_results['cls_f1'])
        history['mit_auc'].append(val_results['mit_auc'])
        history['tub_rmse'].append(val_results['tub_rmse'])
        history['sigma_cls'].append(components.get('sigma_cls', 0))
        history['sigma_mit'].append(components.get('sigma_mit', 0))
        history['sigma_tub'].append(components.get('sigma_tub', 0))

        if val_results['cls_f1'] > best_val_f1:
            best_val_f1 = val_results['cls_f1']
            torch.save(model.state_dict(), OUTPUT_DIR / 'mtgat_best.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Cls F1: {val_results['cls_f1']:.4f} | "
                  f"Mit AUC: {val_results['mit_auc']:.4f} | "
                  f"Tub RMSE: {val_results['tub_rmse']:.4f}")

    # 测试
    model.load_state_dict(torch.load(OUTPUT_DIR / 'mtgat_best.pt'))
    test_loss, test_results = evaluate_mt(model, test_loader, loss_fn, DEVICE)

    print("\n" + "=" * 40)
    print("MULTI-TASK TEST RESULTS")
    print("=" * 40)
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")

    # 可视化
    plot_niche_analysis(cell_df, neighborhood_analyzer)
    plot_multitask_results(history, test_results)

    # 保存结果
    with open(OUTPUT_DIR / 'mtgat_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in test_results.items()}, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
