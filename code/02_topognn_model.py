"""
方案一：基于细胞图神经网络(GNN) + 拓扑数据分析(TDA)的肿瘤微环境空间异质性量化
=============================================================================

核心创新：
  1. 构建 Cell Spatial Graph: 细胞核 → 图节点，空间邻近关系 → 边
  2. 提取多维度节点/边特征
  3. GNN图级分类 + 拓扑特征(Persistent Homology)融合
  4. 拓扑重要性可视化

依赖: torch, torch_geometric, gudhi (或 ripser), networkx
"""

import os, json, warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import (
    GATv2Conv, SAGPooling,
    global_mean_pool, global_max_pool
)
from torch_geometric.utils import to_networkx

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# Part A: 细胞空间图构建
# ============================================================
class CellSpatialGraph:
    """
    从YOLO标注构建细胞空间图

    节点特征 (dim=15):
      - 细胞类型 one-hot (6)
      - 归一化坐标 (2)
      - 归一化尺寸 (2)
      - 面积 (1)
      - 局部密度 (1)
      - 距图像中心距离 (1)
      - 纵横比 (1)
      - 紧密度 (1)

    边连接策略:
      - Delaunay三角剖分 (保留天然空间拓扑)
      - k-NN (k=8) 补充长程连接
    """

    def __init__(self, k_nn: int = 8, delaunay: bool = True,
                 max_dist: float = 300.0):
        self.k_nn = k_nn
        self.use_delaunay = delaunay
        self.max_dist = max_dist

    def extract_node_features(self, cells: List) -> np.ndarray:
        n = len(cells)
        features = np.zeros((n, 15), dtype=np.float32)

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

        coords = np.array([[c['x_center_abs'], c['y_center_abs']] for c in cells])
        if n >= 2:
            tree = KDTree(coords)
            k = min(self.k_nn, n)
            dists, _ = tree.query(coords, k=k + 1)
            local_density = 1.0 / (np.mean(dists[:, 1:], axis=1) + 1e-6) if k > 1 else np.ones(n)
            features[:, 11] = local_density / (local_density.max() + 1e-6)

        img_center = np.array([IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2])
        dist_to_center = np.linalg.norm(coords - img_center, axis=1)
        features[:, 12] = dist_to_center / (dist_to_center.max() + 1e-6)

        return features

    def build_edges(self, coords: np.ndarray) -> np.ndarray:
        edge_list = set()
        n = len(coords)

        if self.use_delaunay and n >= 4:
            try:
                tri = Delaunay(coords)
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i + 1, 3):
                            u, v = simplex[i], simplex[j]
                            dist = np.linalg.norm(coords[u] - coords[v])
                            if dist <= self.max_dist:
                                edge_list.add((u, v))
                                edge_list.add((v, u))
            except Exception:
                pass

        if n >= 2:
            tree = KDTree(coords)
            k = min(self.k_nn, n - 1)
            if k > 0:
                dists, indices = tree.query(coords, k=k + 1)
                for i in range(n):
                    for j_idx in range(1, k + 1):
                        j = indices[i, j_idx]
                        if dists[i, j_idx] <= self.max_dist:
                            edge_list.add((i, j))
                            edge_list.add((j, i))

        if not edge_list:
            edge_list = [(i, i) for i in range(n)]

        return np.array(sorted(edge_list)).T

    def extract_edge_features(self, cells: List, edge_index: np.ndarray) -> np.ndarray:
        n_edges = edge_index.shape[1]
        edge_attr = np.zeros((n_edges, 5), dtype=np.float32)

        for e in range(n_edges):
            u, v = edge_index[0, e], edge_index[1, e]
            dx = cells[u]['x_center_abs'] - cells[v]['x_center_abs']
            dy = cells[u]['y_center_abs'] - cells[v]['y_center_abs']
            dist = np.sqrt(dx**2 + dy**2)
            edge_attr[e, 0] = dist / self.max_dist
            edge_attr[e, 1] = np.arctan2(dy, dx) / np.pi
            edge_attr[e, 2] = 1.0 if cells[u]['class_id'] == cells[v]['class_id'] else 0.0
            a_u, a_v = cells[u]['area_abs'], cells[v]['area_abs']
            edge_attr[e, 3] = min(a_u, a_v) / max(a_u, a_v) if max(a_u, a_v) > 0 else 1.0
            type_interaction = cells[u]['class_id'] * N_CLASSES + cells[v]['class_id']
            edge_attr[e, 4] = type_interaction / 35.0

        return edge_attr

    def build_graph(self, cells: List) -> Data:
        x = self.extract_node_features(cells)
        coords = np.array([[c['x_center_abs'], c['y_center_abs']] for c in cells])
        edge_index = self.build_edges(coords)
        edge_attr = self.extract_edge_features(cells, edge_index)

        class_counts = defaultdict(int)
        for c in cells:
            class_counts[c['class_id']] += 1
        dominant_class = max(class_counts, key=class_counts.get)
        total = len(cells)
        class_ratio = np.array([class_counts[i] / total for i in range(N_CLASSES)])

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor(dominant_class, dtype=torch.long),
            pos=torch.tensor(coords, dtype=torch.float),
            class_ratio=torch.tensor(class_ratio, dtype=torch.float),
            n_nodes=len(cells)
        )
        return data


# ============================================================
# Part B: 拓扑数据分析 (TDA)
# ============================================================
class TopologicalFeatureExtractor:
    """
    基于Persistent Homology的拓扑特征提取

    H0: 连通分量 → 反映细胞聚集模式
    H1: 环状结构 → 反映腺管/小叶等中空结构

    输出:
      - Persistence Image (H0): 50×50 → 2500
      - Persistence Image (H1): 50×50 → 2500
      - Betti Curve (H0): 50
      - Betti Curve (H1): 50
      - 统计特征: 12
      → 总维度: 5112
    """

    def __init__(self, max_dim: int = 1, resolution: int = 50, sigma: float = 0.05):
        self.max_dim = max_dim
        self.resolution = resolution
        self.sigma = sigma

    def compute_persistence(self, coords: np.ndarray) -> Dict:
        n = len(coords)
        if n < 3:
            return {'h0_diagram': [], 'h1_diagram': [], 'n_points': n}

        dist_matrix = squareform(pdist(coords))

        # H0 via Union-Find
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist_matrix[i, j], i, j))
        edges.sort()

        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        h0_diagram = []
        birth_times = {}
        for dist, i, j in edges:
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                birth_i = birth_times.get(root_i, 0)
                birth_j = birth_times.get(root_j, 0)
                if birth_i < birth_j:
                    h0_diagram.append((birth_i, dist))
                    birth_times[root_j] = birth_i
                else:
                    h0_diagram.append((birth_j, dist))
                    birth_times[root_i] = birth_j
                union(i, j)
        for root in set(find(i) for i in range(n)):
            if root in birth_times:
                h0_diagram.append((birth_times[root], float('inf')))

        # H1 via Delaunay
        h1_diagram = []
        if n >= 4:
            try:
                tri = Delaunay(coords)
                for simplex in tri.simplices:
                    pts = coords[simplex]
                    a, b, c = pts[0], pts[1], pts[2]
                    ab, bc, ca = np.linalg.norm(a-b), np.linalg.norm(b-c), np.linalg.norm(c-a)
                    s = (ab + bc + ca) / 2
                    area = np.sqrt(max(0, s*(s-ab)*(s-bc)*(s-ca)))
                    if area > 0:
                        circum_r = (ab*bc*ca) / (4*area)
                        center = np.mean(pts, axis=0)
                        min_dist = float('inf')
                        for k, pt in enumerate(coords):
                            if k not in simplex:
                                min_dist = min(min_dist, np.linalg.norm(pt-center))
                        if min_dist > circum_r:
                            h1_diagram.append((circum_r, min_dist))
            except Exception:
                pass

        return {'h0_diagram': h0_diagram, 'h1_diagram': h1_diagram, 'n_points': n}

    def persistence_image(self, diagram: List[Tuple]) -> np.ndarray:
        if not diagram:
            return np.zeros((self.resolution, self.resolution))
        finite = [(b, d) for b, d in diagram if d < float('inf') and d - b > 1e-6]
        if not finite:
            return np.zeros((self.resolution, self.resolution))

        births = np.array([p[0] for p in finite])
        persists = np.array([p[1]-p[0] for p in finite])
        max_b = births.max() if len(births) > 0 else 1
        max_p = persists.max() if len(persists) > 0 else 1
        if max_b == 0: max_b = 1
        if max_p == 0: max_p = 1

        image = np.zeros((self.resolution, self.resolution))
        for b, d in finite:
            pers = d - b
            bx = int(b / max_b * (self.resolution - 1))
            py = int(pers / max_p * (self.resolution - 1))
            bx = min(bx, self.resolution - 1)
            py = min(py, self.resolution - 1)
            sigma_px = max(1, int(self.sigma * self.resolution))
            for xi in range(max(0, bx-sigma_px), min(self.resolution, bx+sigma_px+1)):
                for yi in range(max(0, py-sigma_px), min(self.resolution, py+sigma_px+1)):
                    image[xi, yi] += np.exp(-((xi-bx)**2+(yi-py)**2)/(2*sigma_px**2))
        if image.max() > 0:
            image /= image.max()
        return image

    def betti_curve(self, diagram: List[Tuple], n_bins: int = 50) -> np.ndarray:
        if not diagram:
            return np.zeros(n_bins)
        finite = [(b, d) for b, d in diagram if d < float('inf')]
        if not finite:
            return np.zeros(n_bins)
        max_val = max(max(d for _, d in finite), max(b for b, _ in finite))
        if max_val == 0: max_val = 1
        bins = np.linspace(0, max_val, n_bins)
        curve = np.array([sum(1 for b, d in finite if b <= t < d) for t in bins])
        return curve

    def _diagram_stats(self, diagram: List[Tuple]) -> np.ndarray:
        if not diagram:
            return np.zeros(6)
        finite = [(b, d) for b, d in diagram if d < float('inf')]
        if not finite:
            return np.zeros(6)
        persists = np.array([d-b for b, d in finite])
        births = np.array([b for b, _ in finite])
        deaths = np.array([d for _, d in finite])
        return np.array([
            np.mean(persists), np.std(persists) if len(persists)>1 else 0,
            np.max(persists), len(finite),
            np.mean(births), np.mean(deaths)
        ])

    def extract_features(self, coords: np.ndarray) -> np.ndarray:
        result = self.compute_persistence(coords)
        pi_h0 = self.persistence_image(result['h0_diagram'])
        pi_h1 = self.persistence_image(result['h1_diagram'])
        bc_h0 = self.betti_curve(result['h0_diagram'])
        bc_h1 = self.betti_curve(result['h1_diagram'])
        stats_h0 = self._diagram_stats(result['h0_diagram'])
        stats_h1 = self._diagram_stats(result['h1_diagram'])
        return np.concatenate([pi_h0.flatten(), pi_h1.flatten(), bc_h0, bc_h1, stats_h0, stats_h1])


# ============================================================
# Part C: GNN模型
# ============================================================
class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3):
        super().__init__()
        self.conv = GATv2Conv(in_dim, out_dim // heads, heads=heads, edge_dim=5, dropout=dropout)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        return x


class TopoGNN(nn.Module):
    """拓扑增强图神经网络"""

    def __init__(self, node_dim=15, topo_dim=5112, hidden_dim=128,
                 num_classes=6, dropout=0.3):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GATBlock(node_dim, hidden_dim, heads=4, dropout=dropout),
            GATBlock(hidden_dim, hidden_dim, heads=4, dropout=dropout),
            GATBlock(hidden_dim, hidden_dim * 2, heads=4, dropout=dropout),
        ])
        self.pool = SAGPooling(hidden_dim * 2, ratio=0.5)

        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, data, topo_features):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        gnn_feat = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        topo_feat = self.topo_encoder(topo_features)
        fused = self.fusion(torch.cat([gnn_feat, topo_feat], dim=1))
        return self.classifier(fused), fused


class AblationGNN(nn.Module):
    """消融: 纯GNN"""
    def __init__(self, node_dim=15, hidden_dim=128, num_classes=6, dropout=0.3):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GATBlock(node_dim, hidden_dim, heads=4, dropout=dropout),
            GATBlock(hidden_dim, hidden_dim, heads=4, dropout=dropout),
            GATBlock(hidden_dim, hidden_dim * 2, heads=4, dropout=dropout),
        ])
        self.pool = SAGPooling(hidden_dim * 2, ratio=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(x), x


# ============================================================
# Part D: 数据集
# ============================================================
class CellGraphDataset(Dataset):
    def __init__(self, cell_df, case_list, graph_builder, topo_extractor):
        super().__init__()
        self.image_groups = cell_df[cell_df['case_id'].isin(case_list)].groupby('filename')
        self.filenames = list(self.image_groups.groups.keys())
        self.graphs, self.topo_features, self.labels = [], [], []

        print(f"Building graphs for {len(self.filenames)} images...")
        for i, fname in enumerate(self.filenames):
            group = self.image_groups.get_group(fname)
            cells = group.to_dict('records')
            graph = graph_builder.build_graph(cells)
            self.graphs.append(graph)
            coords = np.array([[c['x_center_abs'], c['y_center_abs']] for c in cells])
            topo_feat = topo_extractor.extract_features(coords)
            self.topo_features.append(torch.tensor(topo_feat, dtype=torch.float))
            class_counts = defaultdict(int)
            for c in cells:
                class_counts[c['class_id']] += 1
            self.labels.append(max(class_counts, key=class_counts.get))
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(self.filenames)}")

    def len(self): return len(self.graphs)
    def get(self, idx): return self.graphs[idx], self.topo_features[idx], self.labels[idx]


# ============================================================
# 训练与评估
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, use_topo=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, topo_feat, labels in loader:
        data = data.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        optimizer.zero_grad()
        if use_topo:
            logits, _ = model(data, topo_feat.to(device))
        else:
            logits, _ = model(data)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_topo=True):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for data, topo_feat, labels in loader:
        data = data.to(device)
        labels_t = torch.tensor(labels, dtype=torch.long).to(device)
        if use_topo:
            logits, _ = model(data, topo_feat.to(device))
        else:
            logits, _ = model(data)
        total_loss += criterion(logits, labels_t).item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def main():
    print("=" * 60)
    print("方案一: TopoGNN - 拓扑增强图神经网络")
    print("=" * 60)

    cell_df = pd.read_csv(OUTPUT_DIR / 'all_cells.csv')
    print(f"Loaded {len(cell_df)} cells from {cell_df['filename'].nunique()} images")

    with open(OUTPUT_DIR / 'data_split.json') as f:
        split = json.load(f)

    graph_builder = CellSpatialGraph(k_nn=8, delaunay=True, max_dist=300.0)
    topo_extractor = TopologicalFeatureExtractor(max_dim=1, resolution=50)

    print("\nBuilding datasets...")
    train_ds = CellGraphDataset(cell_df, split['train_cases'], graph_builder, topo_extractor)
    val_ds = CellGraphDataset(cell_df, split['val_cases'], graph_builder, topo_extractor)
    test_ds = CellGraphDataset(cell_df, split['test_cases'], graph_builder, topo_extractor)

    from torch_geometric.loader import DataLoader as PyGLoader
    from torch_geometric.data import Batch

    def collate(batch):
        graphs, topos, labels = zip(*batch)
        return Batch.from_data_list(graphs), torch.stack(topos), list(labels)

    train_loader = PyGLoader(list(zip(train_ds.graphs, train_ds.topo_features, train_ds.labels)),
                              batch_size=8, shuffle=True, collate_fn=collate)
    val_loader = PyGLoader(list(zip(val_ds.graphs, val_ds.topo_features, val_ds.labels)),
                            batch_size=8, shuffle=False, collate_fn=collate)
    test_loader = PyGLoader(list(zip(test_ds.graphs, test_ds.topo_features, test_ds.labels)),
                             batch_size=8, shuffle=False, collate_fn=collate)

    # 训练 TopoGNN
    print("\n" + "=" * 40)
    print("Training TopoGNN (GNN + TDA)")
    print("=" * 40)
    model = TopoGNN(node_dim=15, topo_dim=5112, hidden_dim=128, num_classes=N_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(100):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, use_topo=True)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE, use_topo=True)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), OUTPUT_DIR / 'topognn_best.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

    # 测试
    model.load_state_dict(torch.load(OUTPUT_DIR / 'topognn_best.pt'))
    test_loss, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion, DEVICE, use_topo=True)
    print(f"\nTopoGNN Test Results: Acc={test_acc:.4f}, F1={test_f1:.4f}")
    present_classes = sorted(set(int(l) for l in labels) | set(int(p) for p in preds))
    print(classification_report(labels, preds, labels=present_classes,
          target_names=[CLASS_NAMES[i] for i in present_classes], zero_division=0))

    # 消融实验: 纯GNN
    print("\n" + "=" * 40)
    print("Ablation: GNN-only (without TDA)")
    print("=" * 40)
    model_ab = AblationGNN(node_dim=15, hidden_dim=128, num_classes=N_CLASSES).to(DEVICE)
    optimizer_ab = torch.optim.AdamW(model_ab.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_ab = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ab, T_max=100)

    best_val_f1_ab = 0
    for epoch in range(100):
        train_loss, train_acc = train_epoch(model_ab, train_loader, optimizer_ab, criterion, DEVICE, use_topo=False)
        val_loss, val_acc, val_f1, _, _ = evaluate(model_ab, val_loader, criterion, DEVICE, use_topo=False)
        scheduler_ab.step()
        if val_f1 > best_val_f1_ab:
            best_val_f1_ab = val_f1
            torch.save(model_ab.state_dict(), OUTPUT_DIR / 'gnn_only_best.pt')
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

    model_ab.load_state_dict(torch.load(OUTPUT_DIR / 'gnn_only_best.pt'))
    test_loss_ab, test_acc_ab, test_f1_ab, preds_ab, labels_ab = evaluate(
        model_ab, test_loader, criterion, DEVICE, use_topo=False)
    print(f"\nGNN-only Test Results: Acc={test_acc_ab:.4f}, F1={test_f1_ab:.4f}")

    # 结果对比
    print("\n" + "=" * 40)
    print("ABLATION RESULTS")
    print("=" * 40)
    print(f"{'Model':<20} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 44)
    print(f"{'TopoGNN (GNN+TDA)':<20} {test_acc:<12.4f} {test_f1:<12.4f}")
    print(f"{'GNN-only':<20} {test_acc_ab:<12.4f} {test_f1_ab:<12.4f}")
    print(f"{'Improvement':<20} {test_acc-test_acc_ab:<+.4f}      {test_f1-test_f1_ab:<+.4f}")

    # 保存结果
    results = {
        'topognn': {'test_acc': test_acc, 'test_f1': test_f1},
        'gnn_only': {'test_acc': test_acc_ab, 'test_f1': test_f1_ab},
        'improvement': {'acc_delta': test_acc - test_acc_ab, 'f1_delta': test_f1 - test_f1_ab}
    }
    with open(OUTPUT_DIR / 'topognn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train', linewidth=1.5)
    axes[0].plot(history['val_loss'], label='Val', linewidth=1.5)
    axes[0].set_xlabel('Epoch'), axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves'), axes[0].legend()
    axes[1].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
    axes[1].axhline(y=best_val_f1, color='red', linestyle='--', label=f'Best: {best_val_f1:.4f}')
    axes[1].set_xlabel('Epoch'), axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1'), axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'topognn_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
