"""
方案一 v3 Part A: 修复硬伤 + 精简拓扑特征 + 特征对比实验
运行: python3 02c_v3_part1_features.py
"""

import os, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = {0:'Mitosis',1:'Apoptosis',2:'Tumor_nuclei',3:'Non_tumor_nuclei',4:'Tubule',5:'Non_tubule'}
IMAGE_SIZE = (1360,1024); SEED=42
np.random.seed(SEED)

# ============================================================
# Part 1: 无标签泄露传统特征
# ============================================================
def build_clean_features(cell_df):
    records = []
    for fname, group in cell_df.groupby('filename'):
        cells = group.to_dict('records')
        areas = np.array([c['area_abs'] for c in cells])
        widths = np.array([c['width_abs'] for c in cells])
        heights = np.array([c['height_abs'] for c in cells])
        class_ids = np.array([c['class_id'] for c in cells])
        n = len(cells)
        coords = np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])

        nn_mean, nn_std = 0, 0
        if n >= 2:
            tree = KDTree(coords); dists, _ = tree.query(coords, k=min(5,n))
            if dists.shape[1] > 1:
                nn_mean = np.mean(dists[:,1]); nn_std = np.std(dists[:,1])

        records.append({
            'filename': fname, 'case_id': group['case_id'].iloc[0],
            'cell_density': n/(IMAGE_SIZE[0]*IMAGE_SIZE[1])*1e6,
            'area_mean': np.mean(areas), 'area_std': np.std(areas),
            'area_median': np.median(areas),
            'area_q25': np.percentile(areas,25), 'area_q75': np.percentile(areas,75),
            'area_skew': float(pd.Series(areas).skew()) if n>2 else 0,
            'width_mean': np.mean(widths), 'width_std': np.std(widths),
            'height_mean': np.mean(heights), 'height_std': np.std(heights),
            'aspect_ratio_mean': np.mean([min(w,h)/max(w,h) if max(w,h)>0 else 1 for w,h in zip(widths,heights)]),
            'nn_dist_mean': nn_mean, 'nn_dist_std': nn_std,
            'has_mitosis': int(0 in class_ids), 'has_apoptosis': int(1 in class_ids),
            'has_tubule': int(4 in class_ids), 'has_non_tubule': int(5 in class_ids),
            'n_cells': n,
        })
    df = pd.DataFrame(records)
    morph_feats = ['cell_density','area_mean','area_std','area_median',
                   'area_q25','area_q75','area_skew',
                   'width_mean','width_std','height_mean','height_std',
                   'aspect_ratio_mean','nn_dist_mean','nn_dist_std']
    return df, morph_feats

# ============================================================
# Part 2: 精简拓扑特征 (Compact, ~24维)
# ============================================================
class CompactTopoFeatures:
    def __init__(self, radii=[50,100,200,300,400]):
        self.radii = radii

    def compute_persistence(self, coords):
        n=len(coords)
        if n<3: return {'h0_diagram':[],'h1_diagram':[],'n_points':n}
        dm=squareform(pdist(coords))
        edges=[(dm[i,j],i,j) for i in range(n) for j in range(i+1,n)]; edges.sort()
        parent=list(range(n))
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

    def _diagram_stats(self, diagram):
        if not diagram: return np.zeros(6)
        finite=[(b,d) for b,d in diagram if d<float('inf')]
        if not finite: return np.zeros(6)
        persists=np.array([d-b for b,d in finite])
        births=np.array([b for b,_ in finite]); deaths=np.array([d for _,d in finite])
        return np.array([np.mean(persists), np.std(persists) if len(persists)>1 else 0,
                         np.max(persists), len(finite), np.mean(births), np.mean(deaths)])

    def _betti_integral(self, diagram, n_bins=50):
        if not diagram: return 0.0
        finite=[(b,d) for b,d in diagram if d<float('inf')]
        if not finite: return 0.0
        max_val=max(max(d for _,d in finite),max(b for b,_ in finite))
        if max_val==0: return 0.0
        bins=np.linspace(0,max_val,n_bins)
        curve=np.array([sum(1 for b,d in finite if b<=t<d) for t in bins])
        return np.trapz(curve,bins)/(n_bins*max_val)

    def _multiscale_features(self, coords):
        n=len(coords); feats=[]
        for r in self.radii:
            dm=squareform(pdist(coords)); adj=dm<=r
            visited=set(); n_comp=0
            for i in range(n):
                if i not in visited:
                    n_comp+=1; stack=[i]; visited.add(i)
                    while stack:
                        v=stack.pop()
                        for u in range(n):
                            if u not in visited and adj[v,u]:
                                visited.add(u); stack.append(u)
            feats.append(n_comp)
            n_edges=np.sum(adj)/2
            n_cycles=max(0,n_edges-n+n_comp)
            feats.append(min(n_cycles,50))
        return np.array(feats)

    def extract_features(self, coords):
        r=self.compute_persistence(coords)
        h0s=self._diagram_stats(r['h0_diagram'])
        h1s=self._diagram_stats(r['h1_diagram'])
        h0i=self._betti_integral(r['h0_diagram'])
        h1i=self._betti_integral(r['h1_diagram'])
        ms=self._multiscale_features(coords)
        feats=np.concatenate([h0s,h1s,[h0i,h1i],ms])
        n=r['n_points']
        if n>0:
            tp_h0=sum(d-b for b,d in r['h0_diagram'] if d<float('inf'))
            tp_h1=sum(d-b for b,d in r['h1_diagram'] if d<float('inf'))
            feats=np.append(feats,[tp_h0/max(n,1),tp_h1/max(n,1)])
        return feats

# ============================================================
# Experiment 1: TDA统计检验
# ============================================================
def exp1_tda_stats(cell_df, topo_ext):
    print("="*60)
    print("实验1 v3: 精简拓扑特征区分力统计检验")
    print("="*60)
    topo_feats={}
    for fname,group in cell_df.groupby('filename'):
        cells=group.to_dict('records')
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        topo_feats[fname]=topo_ext.extract_features(coords)
    img_labels={}
    for fname,group in cell_df.groupby('filename'):
        cids=group['class_id'].values
        img_labels[fname]={'has_mitosis':int(0 in cids),'has_apoptosis':int(1 in cids),'has_tubule':int(4 in cids)}
    topo_dim=len(list(topo_feats.values())[0])
    print(f"Topo feature dim: {topo_dim}")
    results=[]
    for label_name in ['has_mitosis','has_apoptosis','has_tubule']:
        pos,neg=[],[]
        for fname,feat in topo_feats.items():
            if fname in img_labels:
                if img_labels[fname][label_name]==1: pos.append(feat)
                else: neg.append(feat)
        pos_arr,neg_arr=np.array(pos),np.array(neg)
        p_values=np.array([mannwhitneyu(pos_arr[:,d],neg_arr[:,d],alternative='two-sided')[1]
                           if pos_arr[:,d].std()>0 and neg_arr[:,d].std()>0 else 1.0
                           for d in range(topo_dim)])
        n_sig=(p_values<0.05).sum(); n_bonf=(p_values<0.05/topo_dim).sum()
        top_idx=np.argsort(p_values)[:5]
        results.append({'label':label_name,'n_pos':len(pos),'n_neg':len(neg),
                        'sig_p05':n_sig,'sig_bonf':n_bonf,'min_p':p_values.min(),
                        'median_p':np.median(p_values),'total_dims':topo_dim,
                        'top5_dims':str(top_idx.tolist()),'top5_pvals':str(p_values[top_idx].tolist())})
        print(f"  {label_name}: {n_sig}/{topo_dim} sig(p<0.05), {n_bonf}(Bonferroni), min_p={p_values.min():.2e}")
    pd.DataFrame(results).to_csv(OUTPUT_DIR/'experiment1_v3_tda_stats.csv',index=False)
    return topo_feats

# ============================================================
# Experiment 2: 无泄露特征对比
# ============================================================
def exp2_clean_features(cell_df, topo_feats):
    print("\n"+"="*60)
    print("实验2 v3: 无标签泄露特征对比")
    print("="*60)
    img_df, morph_feats = build_clean_features(cell_df)
    common=sorted(set(topo_feats.keys())&set(img_df['filename'].values))
    X_topo=np.array([topo_feats[f] for f in common])
    topo_dim=X_topo.shape[1]
    X_trad=img_df.set_index('filename').loc[common][morph_feats].values
    X_trad_s=StandardScaler().fit_transform(X_trad)
    X_topo_s=StandardScaler().fit_transform(X_topo)
    X_comb_s=np.concatenate([X_trad_s,X_topo_s],axis=1)
    print(f"Trad: {X_trad.shape[1]}d (no leakage), Topo: {topo_dim}d, Comb: {X_comb_s.shape[1]}d")

    tasks={'has_mitosis':'Mitosis','has_apoptosis':'Apoptosis','has_tubule':'Tubule'}
    results=[]; cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for tk,tn in tasks.items():
        y=img_df.set_index('filename').loc[common][tk].values
        print(f"\n--- {tn} (pos={y.mean():.1%}) ---")
        for fn,X in [('Traditional(clean)',X_trad_s),('Topological',X_topo_s),('Combined',X_comb_s)]:
            rf=RandomForestClassifier(n_estimators=100,random_state=SEED,class_weight='balanced')
            rf_f1=cross_val_score(rf,X,y,cv=cv,scoring='f1')
            rf_auc=cross_val_score(rf,X,y,cv=cv,scoring='roc_auc')
            results.append({'task':tn,'feature':fn,
                            'RF_F1':rf_f1.mean(),'RF_F1_std':rf_f1.std(),
                            'RF_AUC':rf_auc.mean(),'RF_AUC_std':rf_auc.std()})
            print(f"  {fn:25s} | F1={rf_f1.mean():.4f}±{rf_f1.std():.4f} AUC={rf_auc.mean():.4f}±{rf_auc.std():.4f}")
    df=pd.DataFrame(results); df.to_csv(OUTPUT_DIR/'experiment2_v3_clean_features.csv',index=False)
    return df

# ============================================================
# Experiment 3: 拓扑消融 H0 vs H1 vs H0+H1
# ============================================================
def exp3_topo_ablation(cell_df):
    """拓扑特征消融：仅H0 vs 仅H1 vs H0+H1"""
    print("\n"+"="*60)
    print("实验3: 拓扑特征消融 H0 vs H1 vs H0+H1")
    print("="*60)
    img_df, morph_feats = build_clean_features(cell_df)

    # 分别提取H0-only, H1-only, H0+H1特征
    topo_full=CompactTopoFeatures()
    feats_h0={}; feats_h1={}; feats_full={}
    for fname,group in cell_df.groupby('filename'):
        cells=group.to_dict('records')
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        r=topo_full.compute_persistence(coords)
        # H0 only: h0 stats(6) + h0 integral(1) + multiscale components(5) + total persistence(1) = 13
        h0s=topo_full._diagram_stats(r['h0_diagram'])
        h0i=topo_full._betti_integral(r['h0_diagram'])
        ms=topo_full._multiscale_features(coords)
        n=r['n_points']
        tp_h0=sum(d-b for b,d in r['h0_diagram'] if d<float('inf'))/max(n,1) if n>0 else 0
        feats_h0[fname]=np.concatenate([h0s,[h0i],ms[[0,2,4,6,8]],[tp_h0]])  # 13维
        # H1 only: h1 stats(6) + h1 integral(1) + cycle features(5) + total persistence(1) = 13
        h1s=topo_full._diagram_stats(r['h1_diagram'])
        h1i=topo_full._betti_integral(r['h1_diagram'])
        tp_h1=sum(d-b for b,d in r['h1_diagram'] if d<float('inf'))/max(n,1) if n>0 else 0
        feats_h1[fname]=np.concatenate([h1s,[h1i],ms[[1,3,5,7,9]],[tp_h1]])  # 13维
        # Full
        feats_full[fname]=topo_full.extract_features(coords)

    common=sorted(set(feats_full.keys())&set(img_df['filename'].values))
    X_h0=np.array([feats_h0[f] for f in common])
    X_h1=np.array([feats_h1[f] for f in common])
    X_full=np.array([feats_full[f] for f in common])
    print(f"Dims: H0={X_h0.shape[1]}, H1={X_h1.shape[1]}, H0+H1={X_full.shape[1]}")

    tasks={'has_mitosis':'Mitosis','has_apoptosis':'Apoptosis','has_tubule':'Tubule'}
    results=[]; cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for tk,tn in tasks.items():
        y=img_df.set_index('filename').loc[common][tk].values
        print(f"\n--- {tn} ---")
        for fn,X in [('H0 only',X_h0),('H1 only',X_h1),('H0+H1 (full)',X_full)]:
            X_s=StandardScaler().fit_transform(X)
            rf=RandomForestClassifier(n_estimators=100,random_state=SEED,class_weight='balanced')
            rf_auc=cross_val_score(rf,X_s,y,cv=cv,scoring='roc_auc')
            results.append({'task':tn,'topo_type':fn,'RF_AUC':rf_auc.mean(),'RF_AUC_std':rf_auc.std()})
            print(f"  {fn:20s} | AUC={rf_auc.mean():.4f}±{rf_auc.std():.4f}")
    df=pd.DataFrame(results); df.to_csv(OUTPUT_DIR/'experiment3_topo_ablation.csv',index=False)
    return df

# ============================================================
# Main
# ============================================================
def main():
    print("方案一 v3: 修复硬伤 + 精简拓扑特征")
    cell_df=pd.read_csv(OUTPUT_DIR/'all_cells.csv')
    topo_ext=CompactTopoFeatures(radii=[50,100,200,300,400])
    topo_feats=exp1_tda_stats(cell_df, topo_ext)
    exp2_clean_features(cell_df, topo_feats)
    exp3_topo_ablation(cell_df)

    # Summary visualization
    df2=pd.read_csv(OUTPUT_DIR/'experiment2_v3_clean_features.csv')
    df3=pd.read_csv(OUTPUT_DIR/'experiment3_topo_ablation.csv')

    fig,axes=plt.subplots(1,2,figsize=(16,6))
    # Feature comparison
    tasks_list=['Mitosis','Apoptosis','Tubule']
    feat_types=['Traditional(clean)','Topological','Combined']
    x=np.arange(len(tasks_list)); width=0.25
    colors=['#3498db','#e74c3c','#2ecc71']
    for i,ft in enumerate(feat_types):
        vals=[df2[(df2['task']==t)&(df2['feature']==ft)]['RF_AUC'].values[0] for t in tasks_list]
        axes[0].bar(x+i*width,vals,width,label=ft,color=colors[i],edgecolor='black')
    axes[0].set_xticks(x+width); axes[0].set_xticklabels(tasks_list)
    axes[0].set_ylabel('RF AUC (5-fold CV)'); axes[0].set_title('Feature Comparison (No Label Leakage)')
    axes[0].legend(fontsize=8); axes[0].set_ylim(0,1.05)

    # Topo ablation
    topo_types=['H0 only','H1 only','H0+H1 (full)']
    x2=np.arange(len(tasks_list)); width2=0.25
    colors2=['#9b59b6','#f39c12','#1abc9c']
    for i,tt in enumerate(topo_types):
        vals=[df3[(df3['task']==t)&(df3['topo_type']==tt)]['RF_AUC'].values[0] for t in tasks_list]
        axes[1].bar(x2+i*width2,vals,width2,label=tt,color=colors2[i],edgecolor='black')
    axes[1].set_xticks(x2+width2); axes[1].set_xticklabels(tasks_list)
    axes[1].set_ylabel('RF AUC (5-fold CV)'); axes[1].set_title('Topological Feature Ablation')
    axes[1].legend(fontsize=8); axes[1].set_ylim(0,1.05)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/'v3_feature_comparison.png',dpi=150); plt.close()

    print(f"\nDone! Results in {OUTPUT_DIR}")

if __name__=='__main__':
    main()
