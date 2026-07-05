"""
方案一 v4 Part B: P0-1 图结构消融实验 (Full vs NoEdges vs Random)
运行: python3 02e_v4_partB.py
"""
import os, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score, f1_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
FIG_DIR = OUTPUT_DIR / 'v4_figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
IMAGE_SIZE = (1360, 1024); N_CLASSES = 6
print(f"Device: {DEVICE}")

# ============================================================
# 图构建器 (支持三种模式)
# ============================================================
class CellSpatialGraph:
    def __init__(self, k_nn=8, delaunay=True, max_dist=300.0):
        self.k_nn, self.use_delaunay, self.max_dist = k_nn, delaunay, max_dist

    def extract_node_features(self, cells):
        n = len(cells); features = np.zeros((n, 15), dtype=np.float32)
        for i, cell in enumerate(cells):
            oh = np.zeros(N_CLASSES); oh[cell['class_id']] = 1.0
            features[i,:6] = oh
            features[i,6] = cell['x_center']; features[i,7] = cell['y_center']
            features[i,8] = cell['width']; features[i,9] = cell['height']
            features[i,10] = np.log1p(cell['area_abs']) / 15.0
            w, h = cell['width_abs'], cell['height_abs']
            features[i,13] = min(w,h)/max(w,h) if max(w,h)>0 else 1.0
            features[i,14] = cell['area_abs']/(w*h) if (w*h)>0 else 1.0
        coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        if n >= 2:
            tree = KDTree(coords); k = min(self.k_nn, n)
            dists, _ = tree.query(coords, k=k+1)
            ld = 1.0/(np.mean(dists[:,1:], axis=1)+1e-6) if k>1 else np.ones(n)
            features[:,11] = ld/(ld.max()+1e-6)
        img_c = np.array([IMAGE_SIZE[0]/2, IMAGE_SIZE[1]/2])
        dc = np.linalg.norm(coords-img_c, axis=1)
        features[:,12] = dc/(dc.max()+1e-6)
        return features

    def build_edges(self, coords):
        el = set(); n = len(coords)
        if self.use_delaunay and n >= 4:
            try:
                tri = Delaunay(coords)
                for s in tri.simplices:
                    for i in range(3):
                        for j in range(i+1,3):
                            u,v = s[i],s[j]
                            if np.linalg.norm(coords[u]-coords[v]) <= self.max_dist:
                                el.add((u,v)); el.add((v,u))
            except: pass
        if n >= 2:
            tree = KDTree(coords); k = min(self.k_nn, n-1)
            if k > 0:
                _, indices = tree.query(coords, k=k+1)
                for i in range(n):
                    for j in indices[i,1:]:
                        if np.linalg.norm(coords[i]-coords[j]) <= self.max_dist:
                            el.add((i,j)); el.add((j,i))
        if not el: el = {(i,i) for i in range(n)}
        return np.array(sorted(el)).T

    def build_random_edges(self, n_nodes, n_edges_per_node=8):
        el = set()
        for i in range(n_nodes):
            targets = np.random.choice(n_nodes, min(n_edges_per_node, n_nodes), replace=False)
            for j in targets:
                if i != j: el.add((i,j)); el.add((j,i))
        if not el: el = {(i,i) for i in range(n_nodes)}
        return np.array(sorted(el)).T

    def build_graph(self, cells, mode='full'):
        x = self.extract_node_features(cells)
        coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        if mode == 'full':
            ei = self.build_edges(coords)
        elif mode == 'random':
            ei = self.build_random_edges(len(cells))
        else:
            ei = np.array([[i, i] for i in range(len(cells))]).T
        return Data(x=torch.tensor(x), edge_index=torch.tensor(ei, dtype=torch.long),
                    pos=torch.tensor(coords), n_nodes=len(cells))

# ============================================================
# GNN模型
# ============================================================
class SimpleGNN(nn.Module):
    BACKBONES = {'gcn': GCNConv, 'gat': GATv2Conv, 'gin': GINConv, 'sage': SAGEConv}
    def __init__(self, backbone='gat', node_dim=15, hidden_dim=128, n_tasks=3, dropout=0.3):
        super().__init__()
        self.bb = backbone; cc = self.BACKBONES[backbone]
        if backbone == 'gin':
            mlp1 = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.conv1 = cc(mlp1)
            mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim*2))
            self.conv2 = cc(mlp2)
        elif backbone == 'gat':
            self.conv1 = cc(node_dim, hidden_dim//4, heads=4, dropout=dropout)
            self.conv2 = cc(hidden_dim, hidden_dim*2//4, heads=4, dropout=dropout)
        else:
            self.conv1 = cc(node_dim, hidden_dim)
            self.conv2 = cc(hidden_dim, hidden_dim*2)
        self.bn1 = nn.BatchNorm1d(hidden_dim); self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tasks))
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, ei))); x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, ei)))
        return self.classifier(x)

# ============================================================
# 节点标签 + 数据集
# ============================================================
def build_node_labels(cell_df, radius=200):
    records = []
    for fname, group in cell_df.groupby('filename'):
        cells = group.to_dict('records')
        coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        cids = np.array([c['class_id'] for c in cells])
        tm = KDTree(coords[cids==0]) if (cids==0).sum()>0 else None
        ta = KDTree(coords[cids==1]) if (cids==1).sum()>0 else None
        tt = KDTree(coords[cids==4]) if (cids==4).sum()>0 else None
        for i, cell in enumerate(cells):
            nm=0; na=0; nt=0
            if tm is not None: d,_=tm.query(coords[i:i+1],k=1); nm=int(d[0]<radius)
            if ta is not None: d,_=ta.query(coords[i:i+1],k=1); na=int(d[0]<radius)
            if tt is not None: d,_=tt.query(coords[i:i+1],k=1); nt=int(d[0]<radius)
            records.append({'filename':fname,'case_id':cell.get('case_id',fname.split('-')[0]),
                            'cell_idx':i,'class_id':cell['class_id'],
                            'near_mitosis':nm,'near_apoptosis':na,'near_tubule':nt})
    return pd.DataFrame(records)

class AblationGraphDataset(Dataset):
    def __init__(self, cell_df, node_df, case_list, gb, mode='full'):
        super().__init__()
        self.node_df = node_df[node_df['case_id'].isin(case_list)].copy()
        self.graphs = {}; self.node_labels = {}
        ig = cell_df[cell_df['case_id'].isin(case_list)].groupby('filename')
        fnames = sorted(ig.groups.keys())
        for fname in fnames:
            group = ig.get_group(fname); cells = group.to_dict('records')
            self.graphs[fname] = gb.build_graph(cells, mode=mode)
            subset = self.node_df[self.node_df['filename']==fname]
            labels = torch.zeros((len(cells), 3), dtype=torch.float)
            for _, row in subset.iterrows():
                labels[row['cell_idx']] = torch.tensor(
                    [row['near_mitosis'],row['near_apoptosis'],row['near_tubule']], dtype=torch.float)
            self.node_labels[fname] = labels
        self.fnames = fnames
    def len(self): return len(self.fnames)
    def get(self, idx):
        fn = self.fnames[idx]
        return self.graphs[fn], self.node_labels[fn], fn

def train_gnn_ablation(model, train_fns, val_fns, test_fns, dataset, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    pw = torch.tensor([5.0, 3.0, 2.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_vauc = 0
    for epoch in range(epochs):
        model.train(); tl = 0
        for fn in train_fns:
            g, labels, _ = dataset[dataset.fnames.index(fn)]
            g = g.to(DEVICE); labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(g)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step(); tl += loss.item()
        tl /= len(train_fns)
        model.eval(); vp, vl = [], []
        with torch.no_grad():
            for fn in val_fns:
                g, labels, _ = dataset[dataset.fnames.index(fn)]
                g = g.to(DEVICE)
                logits = model(g)
                vp.append(torch.sigmoid(logits).cpu().numpy()); vl.append(labels.numpy())
        vp = np.vstack(vp); vl = np.vstack(vl)
        try: vauc = roc_auc_score(vl, vp, average='macro')
        except: vauc = 0.5
        if vauc > best_vauc:
            best_vauc = vauc
            torch.save(model.state_dict(), OUTPUT_DIR/f'ablation_{model.bb}_best.pt')
        if (epoch+1) % 20 == 0:
            print(f"      Epoch {epoch+1}: Loss={tl:.4f}, Val AUC={vauc:.4f}")

    model.load_state_dict(torch.load(OUTPUT_DIR/f'ablation_{model.bb}_best.pt'))
    model.eval(); tp, tl2 = [], []
    with torch.no_grad():
        for fn in test_fns:
            g, labels, _ = dataset[dataset.fnames.index(fn)]
            g = g.to(DEVICE)
            logits = model(g)
            tp.append(torch.sigmoid(logits).cpu().numpy()); tl2.append(labels.numpy())
    tp = np.vstack(tp); tl2 = np.vstack(tl2)
    tn = ['Mitosis_Nearby', 'Apoptosis_Nearby', 'Tubule_Nearby']
    res = {}
    for i, name in enumerate(tn):
        try: auc = roc_auc_score(tl2[:,i], tp[:,i])
        except: auc = float('nan')
        f1 = f1_score(tl2[:,i], (tp[:,i]>0.5).astype(int), zero_division=0)
        res[name] = {'AUC': auc, 'F1': f1}
    return res

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("方案一 v4 Part B: Graph Structure Ablation")
    print("="*60)

    cell_df = pd.read_csv(OUTPUT_DIR/'all_cells.csv')
    gb = CellSpatialGraph(k_nn=8, delaunay=True, max_dist=300.0)
    node_df = build_node_labels(cell_df, radius=200)

    # 按病例划分
    case_ids = sorted(set(cell_df['case_id'].unique()))
    np.random.seed(SEED); np.random.shuffle(case_ids)
    n_cases = len(case_ids)
    train_cases = case_ids[:int(0.7*n_cases)]
    val_cases = case_ids[int(0.7*n_cases):int(0.85*n_cases)]
    test_cases = case_ids[int(0.85*n_cases):]
    print(f"Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")
    all_cases_list = list(train_cases)+list(val_cases)+list(test_cases)

    # 构建三种数据集
    print("\nBuilding datasets for 3 graph modes...")
    ds_full = AblationGraphDataset(cell_df, node_df, all_cases_list, gb, mode='full')
    ds_noedge = AblationGraphDataset(cell_df, node_df, all_cases_list, gb, mode='no_edges')
    ds_random = AblationGraphDataset(cell_df, node_df, all_cases_list, gb, mode='random')

    # 按病例分配文件名
    def get_case_fns(ds, case_list):
        return [f for f in ds.fnames if any(c in case_list for c in [f.split('-')[0]])]

    train_fns = get_case_fns(ds_full, train_cases)
    val_fns = get_case_fns(ds_full, val_cases)
    test_fns = get_case_fns(ds_full, test_cases)
    print(f"Graphs: train={len(train_fns)}, val={len(val_fns)}, test={len(test_fns)}")

    # 消融实验: GCN+GAT 两个backbone × 3种图模式
    all_results = {}
    modes = {'Full Graph': ds_full, 'No Edges (MLP)': ds_noedge, 'Random Edges': ds_random}
    backbones = ['gcn', 'gat']

    for bb in backbones:
        all_results[bb] = {}
        for mode_name, ds in modes.items():
            print(f"\n  {bb.upper()} + {mode_name}...")
            model = SimpleGNN(backbone=bb, node_dim=15, hidden_dim=128, n_tasks=3).to(DEVICE)
            res = train_gnn_ablation(model, train_fns, val_fns, test_fns, ds, epochs=50)
            all_results[bb][mode_name] = res
            for tn, m in res.items():
                print(f"    {tn}: AUC={m['AUC']:.4f}, F1={m['F1']:.4f}")

    # 保存
    with open(OUTPUT_DIR/'v4_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ====== Visualization ======
    print("\nGenerating ablation figures...")
    tn = ['Apoptosis_Nearby', 'Tubule_Nearby']  # skip Mitosis (all nan)
    mode_names = ['Full Graph', 'No Edges (MLP)', 'Random Edges']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, task in enumerate(tn):
        x = np.arange(len(mode_names)); width = 0.35
        for bi, bb in enumerate(backbones):
            vals = [all_results[bb][m][task]['AUC'] for m in mode_names]
            vals = [v if not np.isnan(v) else 0 for v in vals]
            axes[idx].bar(x + bi*width, vals, width, label=bb.upper(),
                         color=['#3498db','#e74c3c'][bi], edgecolor='black')
        axes[idx].set_xticks(x + width/2); axes[idx].set_xticklabels(mode_names, fontsize=8)
        axes[idx].set_title(f'{task}'); axes[idx].set_ylabel('AUC')
        axes[idx].set_ylim(0, 1.1); axes[idx].legend(fontsize=8)
        # Add delta annotations
        for bi, bb in enumerate(backbones):
            full_val = all_results[bb]['Full Graph'][task]['AUC']
            noedge_val = all_results[bb]['No Edges (MLP)'][task]['AUC']
            if not np.isnan(full_val) and not np.isnan(noedge_val):
                delta = full_val - noedge_val
                axes[idx].annotate(f'Δ={delta:+.3f}', xy=(0.5, max(full_val, noedge_val)+0.03),
                                   ha='center', fontsize=9, fontweight='bold',
                                   color='green' if delta>0 else 'red')
    plt.suptitle('Graph Structure Ablation: Full vs No Edges vs Random', fontsize=13, y=1.01)
    plt.tight_layout(); plt.savefig(FIG_DIR/'v4_graph_ablation.png', dpi=200, bbox_inches='tight'); plt.close()

    # Heatmap summary
    fig, ax = plt.subplots(figsize=(10, 5))
    heatmap_data = []
    row_labels = []
    for bb in backbones:
        for mode in mode_names:
            row_labels.append(f'{bb.upper()}+{mode}')
            row = []
            for task in tn:
                v = all_results[bb][mode][task]['AUC']
                row.append(0 if np.isnan(v) else v)
            heatmap_data.append(row)
    heatmap_data = np.array(heatmap_data)
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                xticklabels=tn, yticklabels=row_labels, ax=ax, cbar_kws={'label': 'AUC'})
    ax.set_title('Graph Structure Ablation Heatmap')
    plt.tight_layout(); plt.savefig(FIG_DIR/'v4_ablation_heatmap.png', dpi=200, bbox_inches='tight'); plt.close()

    print(f"\nDone! Results in {OUTPUT_DIR} and {FIG_DIR}")

if __name__ == '__main__':
    main()
