"""
方案一 Final Part A (Optimized): P0-2 + P1-3 + P2-6/8
优化: 预计算所有CV预测，permutation只在标签上排列（避免重复训练）
运行: python3 02d_final_partA.py
"""
import os, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, Delaunay
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
FIG_DIR = OUTPUT_DIR / 'final_figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/BreCAHAD/images')
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
IMAGE_SIZE=(1360,1024)

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
# Data Loading
# ============================================================
cell_df = pd.read_csv(OUTPUT_DIR/'all_cells.csv')
with open(OUTPUT_DIR/'data_split.json') as f: split = json.load(f)

# Build clean features
records=[]
for fname,group in cell_df.groupby('filename'):
    cells=group.to_dict('records')
    areas=np.array([c['area_abs'] for c in cells])
    widths=np.array([c['width_abs'] for c in cells])
    heights=np.array([c['height_abs'] for c in cells])
    cids=np.array([c['class_id'] for c in cells])
    n=len(cells); coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
    nn_m,nn_s=0,0
    if n>=2:
        tree=KDTree(coords); dists,_=tree.query(coords,k=min(5,n))
        if dists.shape[1]>1: nn_m=np.mean(dists[:,1]); nn_s=np.std(dists[:,1])
    records.append({'filename':fname,'case_id':group['case_id'].iloc[0],
        'cell_density':n/(IMAGE_SIZE[0]*IMAGE_SIZE[1])*1e6,
        'area_mean':np.mean(areas),'area_std':np.std(areas),'area_median':np.median(areas),
        'area_q25':np.percentile(areas,25),'area_q75':np.percentile(areas,75),
        'area_skew':float(pd.Series(areas).skew()) if n>2 else 0,
        'width_mean':np.mean(widths),'width_std':np.std(widths),
        'height_mean':np.mean(heights),'height_std':np.std(heights),
        'aspect_ratio_mean':np.mean([min(w,h)/max(w,h) if max(w,h)>0 else 1 for w,h in zip(widths,heights)]),
        'nn_dist_mean':nn_m,'nn_dist_std':nn_s,
        'has_mitosis':int(0 in cids),'has_apoptosis':int(1 in cids),'has_tubule':int(4 in cids)})
img_df=pd.DataFrame(records)
morph_feats=['cell_density','area_mean','area_std','area_median','area_q25','area_q75','area_skew',
             'width_mean','width_std','height_mean','height_std','aspect_ratio_mean','nn_dist_mean','nn_dist_std']

# CompactTopoFeatures
class CompactTopoFeatures:
    def __init__(self,radii=[50,100,200,300,400]): self.radii=radii
    def compute_persistence(self,coords):
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
    def _diagram_stats(self,d):
        if not d: return np.zeros(6)
        f=[(b,d) for b,d in d if d<float('inf')]
        if not f: return np.zeros(6)
        p=np.array([d-b for b,d in f]); b=np.array([b for b,_ in f]); de=np.array([d for _,d in f])
        return np.array([np.mean(p),np.std(p) if len(p)>1 else 0,np.max(p),len(f),np.mean(b),np.mean(de)])
    def _betti_integral(self,d,nb=50):
        if not d: return 0.0
        f=[(b,d) for b,d in d if d<float('inf')]
        if not f: return 0.0
        mv=max(max(d for _,d in f),max(b for b,_ in f))
        if mv==0: return 0.0
        bins=np.linspace(0,mv,nb)
        curve=np.array([sum(1 for b,d in f if b<=t<d) for t in bins])
        return np.trapz(curve,bins)/(nb*mv)
    def _multiscale_features(self,coords):
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
    def extract_features(self,coords):
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

te = CompactTopoFeatures(radii=[50,100,200,300,400])

# ============================================================
# P0-2: Optimized Permutation Test (pre-compute CV preds)
# ============================================================
def permutation_test_fast(X1, X2, y, n_perm=500):
    """预计算CV预测，只在标签上排列"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf1 = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced', n_jobs=-1)
    rf2 = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced', n_jobs=-1)

    # 预计算CV预测
    pred1 = cross_val_predict(rf1, X1, y, cv=cv, method='predict_proba')[:,1]
    pred2 = cross_val_predict(rf2, X2, y, cv=cv, method='predict_proba')[:,1]
    obs_auc1 = roc_auc_score(y, pred1)
    obs_auc2 = roc_auc_score(y, pred2)
    obs_diff = obs_auc2 - obs_auc1

    # Permutation: shuffle labels, recompute AUC from pre-computed preds
    diffs = []
    for _ in range(n_perm):
        pidx = np.random.permutation(len(y))
        try: a1 = roc_auc_score(y[pidx], pred1)
        except: a1 = 0.5
        try: a2 = roc_auc_score(y[pidx], pred2)
        except: a2 = 0.5
        diffs.append(a2 - a1)

    diffs = np.array(diffs)
    p_val = (np.sum(np.abs(diffs) >= np.abs(obs_diff)) + 1) / (n_perm + 1)
    return obs_diff, p_val, obs_auc1, obs_auc2

# ============================================================
# P1-3: ResNet50
# ============================================================
class ResNetFE:
    def __init__(self):
        self.model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc=nn.Identity(); self.model.eval()
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    @torch.no_grad()
    def extract(self,path):
        img=Image.open(path).convert('RGB')
        return self.model(self.transform(img).unsqueeze(0)).squeeze().numpy()

# ============================================================
# P2-6: Wasserstein Distance
# ============================================================
def wasserstein_distance(diag1,diag2):
    f1=np.array([(b,d) for b,d in diag1 if d<float('inf')])
    f2=np.array([(b,d) for b,d in diag2 if d<float('inf')])
    if len(f1)==0 and len(f2)==0: return 0.0
    if len(f1)==0: return np.mean(np.abs(f2[:,1]-f2[:,0]))
    if len(f2)==0: return np.mean(np.abs(f1[:,1]-f1[:,0]))
    p1=np.sort(f1[:,1]-f1[:,0]); p2=np.sort(f2[:,1]-f2[:,0])
    ml=max(len(p1),len(p2))
    p1=np.pad(p1,(0,ml-len(p1)),constant_values=0)
    p2=np.pad(p2,(0,ml-len(p2)),constant_values=0)
    return np.sum(np.abs(p1-p2))/ml

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("方案一 Final Part A: P0-2 + P1-3 + P2-6/8 (Optimized)")
    print("="*60)

    # Extract topo features
    print("\n[1/5] Extracting topological features...")
    topo_feats={}
    for fname,group in cell_df.groupby('filename'):
        cells=group.to_dict('records')
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        topo_feats[fname]=te.extract_features(coords)
    topo_dim=len(list(topo_feats.values())[0])
    print(f"  Topo dim: {topo_dim}")

    common=sorted(set(topo_feats)&set(img_df['filename'].values))
    X_topo=np.array([topo_feats[f] for f in common])
    X_trad=img_df.set_index('filename').loc[common][morph_feats].values
    X_comb=np.concatenate([X_trad,X_topo],axis=1)

    # ====== P1-3: ResNet50 ======
    print("\n[2/5] P1-3: ResNet50 Baseline...")
    try:
        extractor=ResNetFE()
        resnet_feats={}
        for i,fname in enumerate(common):
            img_path=IMAGES_DIR/f"{fname}.jpg"
            if img_path.exists():
                resnet_feats[fname]=extractor.extract(img_path)
            if (i+1)%30==0: print(f"  {i+1}/{len(common)}")
        X_resnet=np.array([resnet_feats[f] for f in common])
        has_resnet=True
        print(f"  ResNet50: {X_resnet.shape[1]}d")
    except Exception as e:
        print(f"  ResNet50 failed: {e}, skipping")
        has_resnet=False; X_resnet=None

    # ====== P0-2: Statistical Tests ======
    print("\n[3/5] P0-2: Permutation Tests (fast, n=500)...")
    tasks={'has_mitosis':'Mitosis','has_apoptosis':'Apoptosis','has_tubule':'Tubule'}
    stat_results=[]

    for tk,tn in tasks.items():
        y=img_df.set_index('filename').loc[common][tk].values
        Xt_s=StandardScaler().fit_transform(X_trad)
        Xp_s=StandardScaler().fit_transform(X_topo)
        Xc_s=StandardScaler().fit_transform(X_comb)

        d1,p1,a1,a2=permutation_test_fast(Xt_s,Xp_s,y)
        s1="***" if p1<0.001 else ("**" if p1<0.01 else ("*" if p1<0.05 else "ns"))
        print(f"  {tn}: Trad({a1:.3f}) vs Topo({a2:.3f}) Δ={d1:+.4f} p={p1:.4f} {s1}")

        d2,p2,_,a3=permutation_test_fast(Xt_s,Xc_s,y)
        s2="***" if p2<0.001 else ("**" if p2<0.01 else ("*" if p2<0.05 else "ns"))
        print(f"  {tn}: Trad({a1:.3f}) vs Comb({a3:.3f}) Δ={d2:+.4f} p={p2:.4f} {s2}")

        row={'task':tn,'Trad_AUC':a1,'Topo_AUC':a2,'Comb_AUC':a3,
             'Trad_vs_Topo_p':p1,'Trad_vs_Comb_p':p2}

        if has_resnet:
            Xr_s=StandardScaler().fit_transform(X_resnet)
            d3,p3,_,a4=permutation_test_fast(Xt_s,Xr_s,y)
            s3="***" if p3<0.001 else ("**" if p3<0.01 else ("*" if p3<0.05 else "ns"))
            print(f"  {tn}: Trad({a1:.3f}) vs ResNet({a4:.3f}) Δ={d3:+.4f} p={p3:.4f} {s3}")
            d4,p4,_,_=permutation_test_fast(Xr_s,Xc_s,y)
            s4="***" if p4<0.001 else ("**" if p4<0.01 else ("*" if p4<0.05 else "ns"))
            print(f"  {tn}: ResNet({a4:.3f}) vs Comb({a3:.3f}) Δ={d4:+.4f} p={p4:.4f} {s4}")
            row['ResNet_AUC']=a4; row['Trad_vs_ResNet_p']=p3; row['ResNet_vs_Comb_p']=p4
        stat_results.append(row)

    df_stat=pd.DataFrame(stat_results)
    df_stat.to_csv(OUTPUT_DIR/'final_statistical_tests.csv',index=False)
    print(f"\n  Results saved to final_statistical_tests.csv")

    # ====== P2-6: Topological Distance ======
    print("\n[4/5] P2-6: Topological Distance Matrix...")
    diagrams={}
    for fname in common:
        cells=cell_df[cell_df['filename']==fname].to_dict('records')
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        diagrams[fname]=te.compute_persistence(coords)

    n=len(common); dist_mat=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            d_h0=wasserstein_distance(diagrams[common[i]]['h0_diagram'],diagrams[common[j]]['h0_diagram'])
            d_h1=wasserstein_distance(diagrams[common[i]]['h1_diagram'],diagrams[common[j]]['h1_diagram'])
            dist_mat[i,j]=d_h0+d_h1; dist_mat[j,i]=dist_mat[i,j]
        if (i+1)%30==0: print(f"  {i+1}/{n}")

    # P2-8: 病例聚类
    print("\n[5/5] P2-8: Case Clustering via Topological Distance...")
    case_ids=img_df.set_index('filename').loc[common]['case_id'].values
    unique_cases=sorted(set(case_ids))
    case_dist=np.zeros((len(unique_cases),len(unique_cases)))
    for ci,c1 in enumerate(unique_cases):
        for cj,c2 in enumerate(unique_cases):
            mask_i=np.array([i for i,c in enumerate(case_ids) if c==c1])
            mask_j=np.array([j for j,c in enumerate(case_ids) if c==c2])
            case_dist[ci,cj]=np.mean(dist_mat[np.ix_(mask_i,mask_j)])

    clustering=AgglomerativeClustering(n_clusters=3,metric='precomputed',linkage='average')
    case_clusters=clustering.fit_predict(case_dist)

    case_mitosis={}
    for c in unique_cases:
        mask=case_ids==c
        case_mitosis[c]=img_df.set_index('filename').loc[common]['has_mitosis'].values[mask].mean()
    ref_labels=pd.cut(pd.Series(case_mitosis),bins=3,labels=[0,1,2]).astype(int)
    ari=adjusted_rand_score(ref_labels,case_clusters)
    np.fill_diagonal(case_dist,0)
    sil=silhouette_score(case_dist,case_clusters,metric='precomputed')
    print(f"  Clusters: {dict(zip(unique_cases,case_clusters))}")
    print(f"  Silhouette: {sil:.4f}, ARI (vs mitosis ratio): {ari:.4f}")

    # ====== Visualizations ======
    # Fig 1: Statistical comparison
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for idx,(tk,tn) in enumerate(tasks.items()):
        row=df_stat[df_stat['task']==tn].iloc[0]
        methods=['Traditional','Topological','Combined']
        if has_resnet: methods.append('ResNet50')
        vals=[row['Trad_AUC'],row['Topo_AUC'],row['Comb_AUC']]
        if has_resnet: vals.append(row['ResNet_AUC'])
        colors=['#3498db','#e74c3c','#2ecc71','#f39c12'][:len(methods)]
        axes[idx].bar(methods,vals,color=colors,edgecolor='black')
        axes[idx].set_title(f'{tn} Detection'); axes[idx].set_ylabel('AUC (5-fold CV)')
        axes[idx].set_ylim(0,1.05); axes[idx].tick_params(axis='x',rotation=20)
        p1=row['Trad_vs_Topo_p']; p2=row['Trad_vs_Comb_p']
        s1="***" if p1<0.001 else ("**" if p1<0.01 else ("*" if p1<0.05 else "ns"))
        s2="***" if p2<0.001 else ("**" if p2<0.01 else ("*" if p2<0.05 else "ns"))
        axes[idx].text(1,vals[1]+0.02,f'T vs Topo: {s1}',ha='center',fontsize=8,color='#e74c3c')
        axes[idx].text(2,vals[2]+0.02,f'T vs Comb: {s2}',ha='center',fontsize=8,color='#2ecc71')
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig1_statistical_comparison.png',dpi=200); plt.close()

    # Fig 2: Topological distance
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    sns.heatmap(case_dist,ax=axes[0],cmap='YlOrRd',square=True,
                xticklabels=unique_cases,yticklabels=unique_cases)
    axes[0].set_title('Inter-Case Topological Distance (Wasserstein)')
    tsne=TSNE(n_components=2,metric='precomputed',init='random',random_state=SEED,perplexity=max(2,min(5,len(unique_cases)-1)))
    case_2d=tsne.fit_transform(case_dist)
    for ci,c in enumerate(unique_cases):
        axes[1].scatter(case_2d[ci,0],case_2d[ci,1],s=200,c=f'C{case_clusters[ci]}',
                       edgecolors='black',linewidth=1.5)
        axes[1].annotate(c,(case_2d[ci,0],case_2d[ci,1]),fontsize=8,ha='center',va='center')
    axes[1].set_title(f'Case Clustering (t-SNE)\nSilhouette={sil:.3f}, ARI={ari:.3f}')
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig2_topological_distance.png',dpi=200); plt.close()

    # Fig 3: Topo feature importance
    fig,ax=plt.subplots(figsize=(14,6))
    rf=RandomForestClassifier(n_estimators=100,random_state=SEED,class_weight='balanced')
    y_all=img_df.set_index('filename').loc[common]['has_mitosis'].values
    rf.fit(StandardScaler().fit_transform(X_topo),y_all)
    importances=rf.feature_importances_
    top_idx=np.argsort(importances)[-15:][::-1]
    ax.barh([TOPO_DIM_NAMES[i] for i in top_idx],importances[top_idx],color='steelblue',edgecolor='black')
    ax.set_xlabel('Feature Importance'); ax.set_title('Top 15 Topological Features for Mitosis Detection')
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig3_topo_importance.png',dpi=200); plt.close()

    print(f"\nDone! Figures in {FIG_DIR}")
    print(f"Results in {OUTPUT_DIR}")

if __name__=='__main__':
    main()
