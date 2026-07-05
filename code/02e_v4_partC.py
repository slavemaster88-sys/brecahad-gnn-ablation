"""
方案一 v4 Part C: P0补充实验
  P0-3: Fine-tuned ResNet50 (在BreCAHAD上微调)
  P0-6: 多尺度图构建敏感性分析 (k=4,8,16,32)
  P0-7: 图结构贡献度量化 (ΔAUC + 统计检验)

运行: python3 02e_v4_partC.py
"""
import os, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import models, transforms
from PIL import Image
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import GATv2Conv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
FIG_DIR = OUTPUT_DIR / 'v4_figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/BreCAHAD/images')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
IMAGE_SIZE = (1360, 1024); N_CLASSES = 6
print(f"Device: {DEVICE}")

# ============================================================
# Data
# ============================================================
cell_df = pd.read_csv(OUTPUT_DIR/'all_cells.csv')
with open(OUTPUT_DIR/'data_split.json') as f: split = json.load(f)

# Build image-level features
records = []
for fname, group in cell_df.groupby('filename'):
    cells = group.to_dict('records')
    areas = np.array([c['area_abs'] for c in cells])
    widths = np.array([c['width_abs'] for c in cells])
    heights = np.array([c['height_abs'] for c in cells])
    cids = np.array([c['class_id'] for c in cells])
    n = len(cells); coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
    nn_m, nn_s = 0, 0
    if n >= 2:
        tree = KDTree(coords); dists, _ = tree.query(coords, k=min(5,n))
        if dists.shape[1] > 1: nn_m = np.mean(dists[:,1]); nn_s = np.std(dists[:,1])
    records.append({'filename':fname, 'case_id':group['case_id'].iloc[0],
        'cell_density':n/(IMAGE_SIZE[0]*IMAGE_SIZE[1])*1e6,
        'area_mean':np.mean(areas),'area_std':np.std(areas),'area_median':np.median(areas),
        'area_q25':np.percentile(areas,25),'area_q75':np.percentile(areas,75),
        'area_skew':float(pd.Series(areas).skew()) if n>2 else 0,
        'width_mean':np.mean(widths),'width_std':np.std(widths),
        'height_mean':np.mean(heights),'height_std':np.std(heights),
        'aspect_ratio_mean':np.mean([min(w,h)/max(w,h) if max(w,h)>0 else 1 for w,h in zip(widths,heights)]),
        'nn_dist_mean':nn_m,'nn_dist_std':nn_s,
        'has_mitosis':int(0 in cids),'has_apoptosis':int(1 in cids),'has_tubule':int(4 in cids)})
img_df = pd.DataFrame(records)

# ============================================================
# P0-3: Fine-tuned ResNet50 on BreCAHAD
# ============================================================
class BreCAHADImageDataset(TorchDataset):
    def __init__(self, filenames, img_df, images_dir, transform=None):
        self.filenames = filenames; self.img_df = img_df.set_index('filename')
        self.images_dir = images_dir; self.transform = transform
        self.labels = []
        for f in filenames:
            row = self.img_df.loc[f]
            self.labels.append(torch.tensor(
                [row['has_mitosis'], row['has_apoptosis'], row['has_tubule']], dtype=torch.float))
    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        img = Image.open(self.images_dir / f"{self.filenames[idx]}.jpg").convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

def finetune_resnet50(common, img_df, images_dir, epochs=15):
    """在BreCAHAD上微调ResNet50，多标签分类"""
    print("  Fine-tuning ResNet50 on BreCAHAD...")

    # 按病例划分
    case_ids = img_df.set_index('filename').loc[common]['case_id'].values
    unique_cases = sorted(set(case_ids))
    np.random.seed(SEED); np.random.shuffle(unique_cases)
    n_cases = len(unique_cases)
    train_cases = unique_cases[:int(0.7*n_cases)]
    val_cases = unique_cases[int(0.7*n_cases):int(0.85*n_cases)]

    train_fns = [f for f, c in zip(common, case_ids) if c in train_cases]
    val_fns = [f for f, c in zip(common, case_ids) if c in val_cases]
    print(f"    Train: {len(train_fns)} images, Val: {len(val_fns)} images")

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = BreCAHADImageDataset(train_fns, img_df, images_dir, train_transform)
    val_ds = BreCAHADImageDataset(val_fns, img_df, images_dir, val_transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 3))
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0, 3.0, 2.0]).to(DEVICE))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train(); tl = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step(); tl += loss.item()
        scheduler.step()

        model.eval(); vl = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                vl += criterion(model(imgs), labels).item()
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), OUTPUT_DIR/'finetuned_resnet50.pt')
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={tl/len(train_loader):.4f}, Val Loss={vl/len(val_loader):.4f}")

    # 提取微调后的特征
    model.load_state_dict(torch.load(OUTPUT_DIR/'finetuned_resnet50.pt'))
    model.fc = nn.Identity(); model.eval()

    extract_transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    ft_feats = {}
    with torch.no_grad():
        for fname in common:
            img = Image.open(images_dir / f"{fname}.jpg").convert('RGB')
            tensor = extract_transform(img).unsqueeze(0).to(DEVICE)
            ft_feats[fname] = model(tensor).squeeze().cpu().numpy()
    return ft_feats

# ============================================================
# P0-6: 多尺度图构建敏感性分析
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

    def build_graph(self, cells):
        x = self.extract_node_features(cells)
        coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        ei = self.build_edges(coords)
        return Data(x=torch.tensor(x), edge_index=torch.tensor(ei, dtype=torch.long),
                    pos=torch.tensor(coords), n_nodes=len(cells))

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

class MultiScaleDataset(PyGDataset):
    def __init__(self, cell_df, node_df, case_list, k_nn):
        super().__init__()
        gb = CellSpatialGraph(k_nn=k_nn, delaunay=True, max_dist=300.0)
        self.graphs = {}; self.node_labels = {}
        ig = cell_df[cell_df['case_id'].isin(case_list)].groupby('filename')
        fnames = sorted(ig.groups.keys())
        for fname in fnames:
            group = ig.get_group(fname); cells = group.to_dict('records')
            self.graphs[fname] = gb.build_graph(cells)
            subset = node_df[node_df['filename']==fname]
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

class SimpleGAT(nn.Module):
    def __init__(self, node_dim=15, hidden_dim=128, n_tasks=3, dropout=0.3):
        super().__init__()
        self.conv1 = GATv2Conv(node_dim, hidden_dim//4, heads=4, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim*2//4, heads=4, dropout=dropout)
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

def train_multiscale_gnn(dataset, train_fns, val_fns, test_fns, epochs=40):
    model = SimpleGAT(node_dim=15, hidden_dim=128, n_tasks=3).to(DEVICE)
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
            loss = criterion(model(g), labels)
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
        if vauc > best_vauc: best_vauc = vauc

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
    print("方案一 v4 Part C: P0补充实验")
    print("="*60)

    # Common data
    common = sorted(img_df['filename'].values)
    case_ids = img_df.set_index('filename').loc[common]['case_id'].values
    unique_cases = sorted(set(case_ids))
    np.random.seed(SEED); np.random.shuffle(unique_cases)
    n_cases = len(unique_cases)
    train_cases = unique_cases[:int(0.7*n_cases)]
    val_cases = unique_cases[int(0.7*n_cases):int(0.85*n_cases)]
    test_cases = unique_cases[int(0.85*n_cases):]

    # ====== P0-3: Fine-tuned ResNet50 ======
    print("\n[1/3] P0-3: Fine-tuned ResNet50 on BreCAHAD...")
    try:
        ft_feats = finetune_resnet50(common, img_df, IMAGES_DIR, epochs=15)
        X_ft = np.array([ft_feats[f] for f in common])

        # GroupKFold evaluation
        tasks = {'has_mitosis':'Mitosis','has_apoptosis':'Apoptosis','has_tubule':'Tubule'}
        print("  Evaluating fine-tuned features:")
        ft_results = []
        for tk, tn in tasks.items():
            y = img_df.set_index('filename').loc[common][tk].values
            X_s = StandardScaler().fit_transform(X_ft)
            gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            preds = []; trues = []
            for train_idx, test_idx in gkf.split(X_s, y, case_ids):
                X_tr, X_te = X_s[train_idx], X_s[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced', n_jobs=-1)
                rf.fit(X_tr, y_tr)
                preds.extend(rf.predict_proba(X_te)[:,1]); trues.extend(y_te)
            auc = roc_auc_score(trues, np.array(preds))
            print(f"    {tn}: Fine-tuned ResNet50 AUC={auc:.3f}")
            ft_results.append({'task': tn, 'Finetuned_ResNet50_AUC': auc})
        pd.DataFrame(ft_results).to_csv(OUTPUT_DIR/'v4_finetuned_resnet.csv', index=False)
    except Exception as e:
        print(f"  Fine-tuning failed: {e}")

    # ====== P0-6: 多尺度图构建敏感性 ======
    print("\n[2/3] P0-6: Multi-scale Graph Construction (k=4,8,16,32)...")
    node_df = build_node_labels(cell_df, radius=200)
    all_cases_list = list(train_cases)+list(val_cases)+list(test_cases)

    k_values = [4, 8, 16, 32]
    multiscale_results = {}

    for k in k_values:
        print(f"  Building graphs with k={k}...")
        ds = MultiScaleDataset(cell_df, node_df, all_cases_list, k_nn=k)

        def get_case_fns(ds, case_list):
            return [f for f in ds.fnames if any(c in case_list for c in [f.split('-')[0]])]

        train_fns = get_case_fns(ds, train_cases)
        val_fns = get_case_fns(ds, val_cases)
        test_fns = get_case_fns(ds, test_cases)

        res = train_multiscale_gnn(ds, train_fns, val_fns, test_fns, epochs=40)
        multiscale_results[f'k={k}'] = res
        for tn, m in res.items():
            print(f"    {tn}: AUC={m['AUC']:.4f}, F1={m['F1']:.4f}")

    with open(OUTPUT_DIR/'v4_multiscale_results.json', 'w') as f:
        json.dump(multiscale_results, f, indent=2)

    # ====== P0-7: 图结构贡献度量化 + 统计检验 ======
    print("\n[3/3] P0-7: Graph Structure Contribution Quantification...")

    # 加载消融结果
    with open(OUTPUT_DIR/'v4_ablation_results.json') as f:
        ablation = json.load(f)

    # 计算每个任务/backbone的图结构贡献
    print("\n  Graph Structure Contribution (ΔAUC = Full - MLP):")
    contributions = []
    for bb in ['gcn', 'gat']:
        for task in ['Mitosis_Nearby', 'Apoptosis_Nearby', 'Tubule_Nearby']:
            full_auc = ablation[bb]['Full Graph'][task]['AUC']
            mlp_auc = ablation[bb]['No Edges (MLP)'][task]['AUC']
            rand_auc = ablation[bb]['Random Edges'][task]['AUC']
            if not np.isnan(full_auc) and not np.isnan(mlp_auc):
                delta_full = full_auc - mlp_auc
                delta_rand = rand_auc - mlp_auc
                print(f"    {bb.upper()} {task}: Full-MLP Δ={delta_full:+.3f}, Rand-MLP Δ={delta_rand:+.3f}")
                contributions.append({
                    'backbone': bb, 'task': task,
                    'Full_AUC': full_auc, 'MLP_AUC': mlp_auc, 'Random_AUC': rand_auc,
                    'Delta_Full': delta_full, 'Delta_Random': delta_rand
                })

    df_contrib = pd.DataFrame(contributions)
    df_contrib.to_csv(OUTPUT_DIR/'v4_graph_contribution.csv', index=False)

    # Wilcoxon signed-rank test: Full vs MLP across tasks
    full_aucs = df_contrib['Full_AUC'].values
    mlp_aucs = df_contrib['MLP_AUC'].values
    stat, p_val = wilcoxon(full_aucs, mlp_aucs)
    print(f"\n  Wilcoxon test (Full vs MLP): W={stat:.1f}, p={p_val:.4f}")
    print(f"  Mean ΔAUC: {np.mean(df_contrib['Delta_Full']):.3f} ± {np.std(df_contrib['Delta_Full']):.3f}")

    # ====== Visualizations ======
    # Fig: Multi-scale sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tn = ['Apoptosis_Nearby', 'Tubule_Nearby']
    for idx, task in enumerate(tn):
        ks = [4, 8, 16, 32]
        aucs = [multiscale_results[f'k={k}'][task]['AUC'] for k in ks]
        f1s = [multiscale_results[f'k={k}'][task]['F1'] for k in ks]
        aucs = [a if not np.isnan(a) else 0 for a in aucs]
        f1s = [f if not np.isnan(f) else 0 for f in f1s]

        ax1 = axes[idx]; ax2 = ax1.twinx()
        line1, = ax1.plot(ks, aucs, 'o-', color='#3498db', linewidth=2, markersize=8, label='AUC')
        line2, = ax2.plot(ks, f1s, 's--', color='#e74c3c', linewidth=2, markersize=8, label='F1')
        ax1.set_xlabel('k-NN Parameter'); ax1.set_ylabel('AUC', color='#3498db')
        ax2.set_ylabel('F1', color='#e74c3c')
        ax1.set_title(f'{task} - Multi-scale Sensitivity')
        ax1.set_xticks(ks)
        lines = [line1, line2]; labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower right', fontsize=8)
    plt.suptitle('GNN Sensitivity to Graph Construction (k-NN)', fontsize=13, y=1.01)
    plt.tight_layout(); plt.savefig(FIG_DIR/'v4_multiscale_sensitivity.png', dpi=200, bbox_inches='tight'); plt.close()

    # Fig: Graph contribution bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_contrib)); width = 0.35
    bars1 = ax.bar(x - width/2, df_contrib['Delta_Full'], width, label='Full Graph - MLP',
                   color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, df_contrib['Delta_Random'], width, label='Random Graph - MLP',
                   color='#f39c12', edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    labels = [f"{r['backbone'].upper()}\n{r['task'].replace('_Nearby','')}" for _, r in df_contrib.iterrows()]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('ΔAUC'); ax.set_title(f'Graph Structure Contribution (Wilcoxon p={p_val:.4f})')
    ax.legend(fontsize=9)
    # Annotate values
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.3f}', ha='center', fontsize=7, fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG_DIR/'v4_graph_contribution.png', dpi=200, bbox_inches='tight'); plt.close()

    print(f"\nDone! Results in {OUTPUT_DIR} and {FIG_DIR}")

if __name__ == '__main__':
    main()
