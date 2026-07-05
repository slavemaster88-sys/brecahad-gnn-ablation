"""
方案一 Final v5: P0-9(节点GNN+CI) + P0-10(Mitosis分布) + P1-1(生物学解释)
运行: python3 02f_v5_compute.py
"""
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, f1_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import GATv2Conv
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; IMAGE_SIZE = (1360, 1024); N_CLASSES = 6
print(f"Device: {DEVICE}")

TOPO_DIM_NAMES = [
    'H0_MeanPersistence','H0_StdPersistence','H0_MaxPersistence',
    'H0_NumFeatures','H0_MeanBirth','H0_MeanDeath',
    'H1_MeanPersistence','H1_StdPersistence','H1_MaxPersistence',
    'H1_NumFeatures','H1_MeanBirth','H1_MeanDeath',
    'Betti0_Integral','Betti1_Integral',
    'MS_r50_Components','MS_r50_Cycles','MS_r100_Components','MS_r100_Cycles',
    'MS_r200_Components','MS_r200_Cycles','MS_r300_Components','MS_r300_Cycles',
    'MS_r400_Components','MS_r400_Cycles',
    'TotalPersistence_H0','TotalPersistence_H1',
]

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
        'has_mitosis':int(0 in cids),'has_apoptosis':int(1 in cids),'has_tubule':int(4 in cids),
        'n_mitosis':int((cids==0).sum()),'n_apoptosis':int((cids==1).sum()),
        'n_tubule':int((cids==4).sum()),'n_cells':n})
img_df = pd.DataFrame(records)
morph_feats = ['cell_density','area_mean','area_std','area_median','area_q25','area_q75','area_skew',
               'width_mean','width_std','height_mean','height_std','aspect_ratio_mean','nn_dist_mean','nn_dist_std']

# Topo features
class CompactTopoFeatures:
    def __init__(self, radii=[50,100,200,300,400]): self.radii = radii
    def compute_persistence(self, coords):
        n = len(coords)
        if n < 3: return {'h0_diagram':[], 'h1_diagram':[], 'n_points':n}
        dm = squareform(pdist(coords))
        edges = [(dm[i,j], i, j) for i in range(n) for j in range(i+1,n)]; edges.sort()
        parent = list(range(n))
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(x,y):
            px,py=find(x),find(y)
            if px!=py: parent[px]=py; return True
            return False
        h0d=[]; bt={}
        for dist,i,j in edges:
            ri,rj=find(i),find(j)
            if ri!=rj:
                bi=bt.get(ri,0); bj=bt.get(rj,0)
                h0d.append((min(bi,bj),dist))
                bt[ri if bi>=bj else rj]=min(bi,bj); union(i,j)
        for root in set(find(i) for i in range(n)):
            if root in bt: h0d.append((bt[root],float('inf')))
        h1d=[]
        if n>=4:
            try:
                tri=Delaunay(coords)
                for s in tri.simplices:
                    pts=coords[s]; a,b,c=pts[0],pts[1],pts[2]
                    ab,bc,ca=np.linalg.norm(a-b),np.linalg.norm(b-c),np.linalg.norm(c-a)
                    ss=(ab+bc+ca)/2; area=np.sqrt(max(0,ss*(ss-ab)*(ss-bc)*(ss-ca)))
                    if area>0:
                        cr=(ab*bc*ca)/(4*area); center=np.mean(pts,axis=0)
                        md=min((np.linalg.norm(pt-center) for k,pt in enumerate(coords) if k not in s),default=float('inf'))
                        if md>cr: h1d.append((cr,md))
            except: pass
        return {'h0_diagram':h0d,'h1_diagram':h1d,'n_points':n}
    def _diagram_stats(self, d):
        if not d: return np.zeros(6)
        f=[(b,d) for b,d in d if d<float('inf')]
        if not f: return np.zeros(6)
        p=np.array([d-b for b,d in f]); b=np.array([b for b,_ in f]); de=np.array([d for _,d in f])
        return np.array([np.mean(p), np.std(p) if len(p)>1 else 0,
                         np.max(p), len(f), np.mean(b), np.mean(de)])
    def _betti_integral(self, d, nb=50):
        if not d: return 0.0
        f=[(b,d) for b,d in d if d<float('inf')]
        if not f: return 0.0
        mv=max(max(d for _,d in f),max(b for b,_ in f))
        if mv==0: return 0.0
        bins=np.linspace(0,mv,nb)
        curve=np.array([sum(1 for b,d in f if b<=t<d) for t in bins])
        return np.trapz(curve,bins)/(nb*mv)
    def _multiscale_features(self, coords):
        n=len(coords); feats=[]
        for r in self.radii:
            dm=squareform(pdist(coords)); adj=dm<=r
            visited=set(); nc=0
            for i in range(n):
                if i not in visited:
                    nc+=1; stack=[i]; visited.add(i)
                    while stack:
                        v=stack.pop()
                        for u in range(n):
                            if u not in visited and adj[v,u]: visited.add(u); stack.append(u)
            feats.append(nc); ne=np.sum(adj)/2
            feats.append(min(max(0,ne-n+nc),50))
        return np.array(feats)
    def extract_features(self, coords):
        r=self.compute_persistence(coords)
        h0s=self._diagram_stats(r['h0_diagram']); h1s=self._diagram_stats(r['h1_diagram'])
        h0i=self._betti_integral(r['h0_diagram']); h1i=self._betti_integral(r['h1_diagram'])
        ms=self._multiscale_features(coords)
        feats=np.concatenate([h0s,h1s,[h0i,h1i],ms])
        n=r['n_points']
        if n>0:
            tp0=sum(d-b for b,d in r['h0_diagram'] if d<float('inf'))
            tp1=sum(d-b for b,d in r['h1_diagram'] if d<float('inf'))
            feats=np.append(feats,[tp0/max(n,1),tp1/max(n,1)])
        return feats.astype(np.float64)

# Graph builder
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

# GNN
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

class NodeDataset(PyGDataset):
    def __init__(self, cell_df, node_df, case_list, gb):
        super().__init__()
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

def bootstrap_gnn_ci(model, test_fns, dataset, n_bootstrap=500):
    model.eval()
    all_preds = []; all_labels = []
    with torch.no_grad():
        for fn in test_fns:
            g, labels, _ = dataset[dataset.fnames.index(fn)]
            g = g.to(DEVICE)
            logits = model(g)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
    all_preds = np.vstack(all_preds); all_labels = np.vstack(all_labels)

    tn = ['Mitosis_Nearby', 'Apoptosis_Nearby', 'Tubule_Nearby']
    results = {}
    for i, name in enumerate(tn):
        yt = all_labels[:,i]; yp = all_preds[:,i]
        valid = ~np.isnan(yt)
        yt = yt[valid]; yp = yp[valid]
        if len(np.unique(yt)) < 2:
            results[name] = {'AUC': float('nan'), 'CI_low': float('nan'), 'CI_high': float('nan'),
                            'F1': 0.0, 'n_pos': int(yt.sum()), 'n_neg': int(len(yt)-yt.sum())}
            continue
        aucs = []
        n = len(yt)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            try: aucs.append(roc_auc_score(yt[idx], yp[idx]))
            except: aucs.append(0.5)
        auc = roc_auc_score(yt, yp)
        ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
        f1 = f1_score(yt, (yp>0.5).astype(int), zero_division=0)
        results[name] = {'AUC': auc, 'CI_low': ci_low, 'CI_high': ci_high,
                        'F1': f1, 'n_pos': int(yt.sum()), 'n_neg': int(len(yt)-yt.sum())}
    return results

def train_node_gnn(dataset, train_fns, val_fns, test_fns, epochs=50):
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
    model.eval()
    return bootstrap_gnn_ci(model, test_fns, dataset, n_bootstrap=500)

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("方案一 v5: P0-9 + P0-10 + P1-1")
    print("="*60)

    # Topo features
    print("\n[Prep] Extracting features...")
    te = CompactTopoFeatures(radii=[50,100,200,300,400])
    topo_feats = {}
    for fname, group in cell_df.groupby('filename'):
        cells = group.to_dict('records')
        coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        topo_feats[fname] = te.extract_features(coords)
    common = sorted(set(topo_feats) & set(img_df['filename'].values))
    case_ids = img_df.set_index('filename').loc[common]['case_id'].values

    # Case split
    unique_cases = sorted(set(case_ids))
    np.random.seed(SEED); np.random.shuffle(unique_cases)
    n_cases = len(unique_cases)
    train_cases = unique_cases[:int(0.7*n_cases)]
    val_cases = unique_cases[int(0.7*n_cases):int(0.85*n_cases)]
    test_cases = unique_cases[int(0.85*n_cases):]
    print(f"  Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")

    # ====== P0-9 ======
    print("\n[P0-9] Node-level GNN with Bootstrap 95% CI...")
    gb = CellSpatialGraph(k_nn=8, delaunay=True, max_dist=300.0)
    node_df = build_node_labels(cell_df, radius=200)
    all_cases_list = list(train_cases)+list(val_cases)+list(test_cases)
    dataset = NodeDataset(cell_df, node_df, all_cases_list, gb)

    def get_case_fns(ds, cl):
        return [f for f in ds.fnames if any(c in cl for c in [f.split('-')[0]])]
    train_fns = get_case_fns(dataset, train_cases)
    val_fns = get_case_fns(dataset, val_cases)
    test_fns = get_case_fns(dataset, test_cases)
    print(f"  Graphs: train={len(train_fns)}, val={len(val_fns)}, test={len(test_fns)}")

    all_runs = []
    for run in range(3):
        print(f"  Run {run+1}/3...")
        np.random.seed(SEED+run); torch.manual_seed(SEED+run)
        res = train_node_gnn(dataset, train_fns, val_fns, test_fns, epochs=50)
        all_runs.append(res)

    tn = ['Mitosis_Nearby', 'Apoptosis_Nearby', 'Tubule_Nearby']
    final_gnn = {}
    print("\n  Final GNN Results (mean of 3 runs, 95% CI):")
    for name in tn:
        aucs = [r[name]['AUC'] for r in all_runs if not np.isnan(r[name]['AUC'])]
        f1s = [r[name]['F1'] for r in all_runs]
        ci_lows = [r[name]['CI_low'] for r in all_runs if not np.isnan(r[name]['CI_low'])]
        ci_highs = [r[name]['CI_high'] for r in all_runs if not np.isnan(r[name]['CI_high'])]
        if aucs:
            ma, sa = np.mean(aucs), np.std(aucs)
            mf, sf = np.mean(f1s), np.std(f1s)
            mcl, mch = np.mean(ci_lows), np.mean(ci_highs)
        else:
            ma, sa, mf, sf, mcl, mch = float('nan'), 0, 0, 0, float('nan'), float('nan')
        final_gnn[name] = {'AUC_mean': ma, 'AUC_std': sa, 'CI_low': mcl, 'CI_high': mch,
                           'F1_mean': mf, 'F1_std': sf,
                           'n_pos': all_runs[0][name].get('n_pos',0),
                           'n_neg': all_runs[0][name].get('n_neg',0)}
        print(f"    {name}: AUC={ma:.3f}±{sa:.3f} [95%CI: {mcl:.3f}-{mch:.3f}], "
              f"F1={mf:.3f}±{sf:.3f}, pos={all_runs[0][name].get('n_pos','?')}")

    clean = {k: {kk: float(vv) if not isinstance(vv,str) else vv for kk,vv in v.items()} for k,v in final_gnn.items()}
    with open(OUTPUT_DIR/'v5_final_gnn_ci.json','w') as f: json.dump(clean, f, indent=2)

    # ====== P0-10 ======
    print("\n[P0-10] Mitosis_Nearby distribution analysis...")
    test_nodes = node_df[node_df['case_id'].isin(test_cases)]
    dist_data = []
    for case in sorted(test_cases):
        cn = test_nodes[test_nodes['case_id']==case]
        nm = int(cn['near_mitosis'].sum()); na = int(cn['near_apoptosis'].sum())
        nt = int(cn['near_tubule'].sum()); ntot = len(cn)
        print(f"  {case}: M={nm}/{ntot}, A={na}/{ntot}, T={nt}/{ntot}")
        dist_data.append({'case':case,'n_mitosis':nm,'n_apoptosis':na,'n_tubule':nt,'n_total':ntot})
    total_mit = test_nodes['near_mitosis'].sum()
    total = len(test_nodes)
    print(f"  Overall: Mitosis={int(total_mit)}/{total} ({total_mit/total*100:.1f}%)")
    pd.DataFrame(dist_data).to_csv(OUTPUT_DIR/'v5_mitosis_distribution.csv', index=False)

    # ====== P1-1 ======
    print("\n[P1-1] Biological interpretation of topological features...")
    X_topo = np.array([topo_feats[f] for f in common])
    y_mit = img_df.set_index('filename').loc[common]['has_mitosis'].values
    y_tub = img_df.set_index('filename').loc[common]['has_tubule'].values

    interps = []
    for i, name in enumerate(TOPO_DIM_NAMES):
        pm = mannwhitneyu(X_topo[y_mit==1,i], X_topo[y_mit==0,i])[1] if (y_mit==1).sum()>=3 and (y_mit==0).sum()>=3 else 1.0
        pt = mannwhitneyu(X_topo[y_tub==1,i], X_topo[y_tub==0,i])[1] if (y_tub==1).sum()>=3 and (y_tub==0).sum()>=3 else 1.0
        interps.append({'dimension':name,'mitosis_p':pm,'tubule_p':pt,
                        'mitosis_sig':pm<0.05,'tubule_sig':pt<0.05})

    df_interp = pd.DataFrame(interps)
    df_interp.to_csv(OUTPUT_DIR/'v5_topo_biology.csv', index=False)

    top_mit = df_interp.nsmallest(5, 'mitosis_p')
    top_tub = df_interp.nsmallest(5, 'tubule_p')
    print("  Top 5 for Mitosis:")
    for _,r in top_mit.iterrows(): print(f"    {r['dimension']}: p={r['mitosis_p']:.2e}")
    print("  Top 5 for Tubule:")
    for _,r in top_tub.iterrows(): print(f"    {r['dimension']}: p={r['tubule_p']:.2e}")

    print(f"\nDone! Results in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
