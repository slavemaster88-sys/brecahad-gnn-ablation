"""
方案一 v4 Part A: P0-2(GroupKFold+CI) + P0-4(Patch Baseline) + P0-5(Gudhi H1)
运行: python3 02e_v4_partA.py
"""
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
FIG_DIR = OUTPUT_DIR / 'v4_figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/BreCAHAD/images')
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
IMAGE_SIZE = (1360, 1024); N_CLASSES = 6

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

# ============================================================
# Data
# ============================================================
cell_df = pd.read_csv(OUTPUT_DIR/'all_cells.csv')
with open(OUTPUT_DIR/'data_split.json') as f: split = json.load(f)

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
morph_feats = ['cell_density','area_mean','area_std','area_median','area_q25','area_q75','area_skew',
               'width_mean','width_std','height_mean','height_std','aspect_ratio_mean','nn_dist_mean','nn_dist_std']

# ============================================================
# Topo Features
# ============================================================
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

# ============================================================
# P0-5: Gudhi H1 Validation
# ============================================================
def validate_h1_with_gudhi(cell_df, te, n_samples=30):
    print("  Validating H1 with gudhi RipsComplex...")
    try:
        import gudhi
        filenames = sorted(cell_df['filename'].unique())
        sample_fns = np.random.choice(filenames, min(n_samples, len(filenames)), replace=False)
        correlations = []
        for fname in sample_fns:
            cells = cell_df[cell_df['filename']==fname].to_dict('records')
            coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
            our_result = te.compute_persistence(coords)
            our_h1_persist = [d-b for b,d in our_result['h1_diagram'] if d<float('inf')]
            if len(coords) >= 4:
                rips = gudhi.RipsComplex(points=coords, max_edge_length=500.0)
                st = rips.create_simplex_tree(max_dimension=2)
                st.compute_persistence()
                gudhi_h1 = st.persistence_intervals_in_dimension(1)
                gudhi_h1_persist = [d-b for b,d in gudhi_h1 if d<float('inf')]
                if len(our_h1_persist) > 0 and len(gudhi_h1_persist) > 0:
                    correlations.append((np.mean(our_h1_persist), np.mean(gudhi_h1_persist)))
        if len(correlations) >= 5:
            corr = np.corrcoef(np.array(correlations).T)[0,1]
            print(f"    H1 persistence correlation (ours vs gudhi): r={corr:.4f} (n={len(correlations)})")
            return corr
        else:
            print("    Insufficient samples for H1 validation")
            return None
    except ImportError:
        print("    gudhi not installed, skipping H1 validation")
        return None

# ============================================================
# P0-2: GroupKFold + Bootstrap CI
# ============================================================
def bootstrap_auc_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try: aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
        except: aucs.append(0.5)
    return np.percentile(aucs, [100*alpha/2, 100*(1-alpha/2)])

def group_kfold_eval(X, y, groups, model=None):
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced', n_jobs=-1)
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = []; trues = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:,1]
        preds.extend(proba); trues.extend(y_test)
    preds = np.array(preds); trues = np.array(trues)
    auc = roc_auc_score(trues, preds)
    ci_low, ci_high = bootstrap_auc_ci(trues, preds)
    return auc, ci_low, ci_high

# ============================================================
# P0-4: Patch Baseline
# ============================================================
class PatchBaseline:
    def __init__(self, patch_size=128, stride=128):
        self.patch_size = patch_size; self.stride = stride
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity(); self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    @torch.no_grad()
    def extract_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        patch_feats = []
        for y in range(0, h-self.patch_size+1, self.stride):
            for x in range(0, w-self.patch_size+1, self.stride):
                patch = img.crop((x, y, x+self.patch_size, y+self.patch_size))
                tensor = self.transform(patch).unsqueeze(0)
                feat = self.model(tensor).squeeze().numpy()
                patch_feats.append(feat)
        if not patch_feats: return np.zeros(512)
        return np.mean(patch_feats, axis=0)

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("方案一 v4 Part A: GroupKFold + Patch + Gudhi")
    print("="*60)

    # Topo features
    print("\n[1/4] Extracting topological features...")
    te = CompactTopoFeatures(radii=[50,100,200,300,400])
    topo_feats = {}
    for fname, group in cell_df.groupby('filename'):
        cells = group.to_dict('records')
        coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        topo_feats[fname] = te.extract_features(coords)
    topo_dim = len(list(topo_feats.values())[0])
    print(f"  Topo dim: {topo_dim}")

    common = sorted(set(topo_feats) & set(img_df['filename'].values))
    X_topo = np.array([topo_feats[f] for f in common])
    X_trad = img_df.set_index('filename').loc[common][morph_feats].values
    X_comb = np.concatenate([X_trad, X_topo], axis=1)
    case_ids = img_df.set_index('filename').loc[common]['case_id'].values
    print(f"  Images: {len(common)}, Cases: {len(set(case_ids))}")

    # P0-5
    print("\n[2/4] P0-5: Gudhi H1 Validation...")
    h1_corr = validate_h1_with_gudhi(cell_df, te, n_samples=30)

    # P0-2
    print("\n[3/4] P0-2: Case-level GroupKFold + Bootstrap 95% CI...")
    tasks = {'has_mitosis':'Mitosis','has_apoptosis':'Apoptosis','has_tubule':'Tubule'}
    cv_results = []

    for tk, tn in tasks.items():
        y = img_df.set_index('filename').loc[common][tk].values
        Xt_s = StandardScaler().fit_transform(X_trad)
        Xp_s = StandardScaler().fit_transform(X_topo)
        Xc_s = StandardScaler().fit_transform(X_comb)

        auc_t, ci_t_l, ci_t_h = group_kfold_eval(Xt_s, y, case_ids)
        auc_p, ci_p_l, ci_p_h = group_kfold_eval(Xp_s, y, case_ids)
        auc_c, ci_c_l, ci_c_h = group_kfold_eval(Xc_s, y, case_ids)

        print(f"  {tn}:")
        print(f"    Traditional: AUC={auc_t:.3f} [95%CI: {ci_t_l:.3f}-{ci_t_h:.3f}]")
        print(f"    Topological: AUC={auc_p:.3f} [95%CI: {ci_p_l:.3f}-{ci_p_h:.3f}]")
        print(f"    Combined:    AUC={auc_c:.3f} [95%CI: {ci_c_l:.3f}-{ci_c_h:.3f}]")

        cv_results.append({
            'task': tn,
            'Trad_AUC': auc_t, 'Trad_CI_low': ci_t_l, 'Trad_CI_high': ci_t_h,
            'Topo_AUC': auc_p, 'Topo_CI_low': ci_p_l, 'Topo_CI_high': ci_p_h,
            'Comb_AUC': auc_c, 'Comb_CI_low': ci_c_l, 'Comb_CI_high': ci_c_h,
        })

    df_cv = pd.DataFrame(cv_results)
    df_cv.to_csv(OUTPUT_DIR/'v4_groupkfold_results.csv', index=False)

    # P0-4
    print("\n[4/4] P0-4: Patch Baseline (ResNet18 sliding window)...")
    try:
        patch_extractor = PatchBaseline(patch_size=128, stride=128)
        patch_feats = {}
        for i, fname in enumerate(common):
            img_path = IMAGES_DIR / f"{fname}.jpg"
            if img_path.exists():
                patch_feats[fname] = patch_extractor.extract_features(img_path)
            if (i+1) % 30 == 0: print(f"  {i+1}/{len(common)}")
        X_patch = np.array([patch_feats[f] for f in common])
        patch_dim = X_patch.shape[1]
        print(f"  Patch features: {patch_dim}d")

        patch_results = []
        print("  Patch baseline GroupKFold evaluation:")
        for tk, tn in tasks.items():
            y = img_df.set_index('filename').loc[common][tk].values
            Xp_s = StandardScaler().fit_transform(X_patch)
            auc_patch, ci_l, ci_h = group_kfold_eval(Xp_s, y, case_ids)
            print(f"    {tn}: Patch AUC={auc_patch:.3f} [95%CI: {ci_l:.3f}-{ci_h:.3f}]")
            patch_results.append({'task': tn, 'Patch_AUC': auc_patch, 'Patch_CI_low': ci_l, 'Patch_CI_high': ci_h})
        pd.DataFrame(patch_results).to_csv(OUTPUT_DIR/'v4_patch_baseline.csv', index=False)
    except Exception as e:
        print(f"  Patch baseline failed: {e}")

    # ====== Visualization: GroupKFold comparison ======
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (tk, tn) in enumerate(tasks.items()):
        row = df_cv[df_cv['task']==tn].iloc[0]
        methods = ['Traditional', 'Topological', 'Combined']
        vals = [row['Trad_AUC'], row['Topo_AUC'], row['Comb_AUC']]
        ci_lows = [row['Trad_CI_low'], row['Topo_CI_low'], row['Comb_CI_low']]
        ci_highs = [row['Trad_CI_high'], row['Topo_CI_high'], row['Comb_CI_high']]
        errors = [[v-l for v,l in zip(vals,ci_lows)], [h-v for v,h in zip(vals,ci_highs)]]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        axes[idx].bar(methods, vals, color=colors, edgecolor='black', yerr=errors, capsize=5)
        axes[idx].set_title(f'{tn} Detection (GroupKFold)'); axes[idx].set_ylabel('AUC')
        axes[idx].set_ylim(0, 1.1); axes[idx].tick_params(axis='x', rotation=15)
        for i, (v, l, h) in enumerate(zip(vals, ci_lows, ci_highs)):
            axes[idx].text(i, v+0.03, f'{v:.3f}\n[{l:.3f}-{h:.3f}]', ha='center', fontsize=7)
    plt.suptitle('Case-Level Cross-Validation with 95% Bootstrap CI', fontsize=13, y=1.01)
    plt.tight_layout(); plt.savefig(FIG_DIR/'v4_groupkfold.png', dpi=200, bbox_inches='tight'); plt.close()

    print(f"\nDone! Results in {OUTPUT_DIR} and {FIG_DIR}")

if __name__ == '__main__':
    main()
