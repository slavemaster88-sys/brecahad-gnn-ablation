"""
方案一增强版：TopoGNN — 有意义的实验设计 (Part 1/2)
=====================================================

修正策略：原版使用"主要细胞类型"作为标签导致trivial任务
改为：
  1. 多标签分类：预测图像中是否存在 Mitosis / Apoptosis / Tubule
  2. 拓扑特征 vs 传统特征对比实验
  3. TopoGNN多标签图分类 + 消融

运行: python3 02b_topognn_v2_part1.py
"""

import os, json, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATv2Conv, SAGPooling, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.data import Batch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Config
DATA_ROOT = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/BreCAHAD')
OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

CLASS_NAMES = {0:'Mitosis',1:'Apoptosis',2:'Tumor_nuclei',3:'Non_tumor_nuclei',4:'Tubule',5:'Non_tubule'}
N_CLASSES = 6
IMAGE_SIZE = (1360,1024)

# ============================================================
# Label builder
# ============================================================
def build_meaningful_labels(cell_df):
    records = []
    for fname, group in cell_df.groupby('filename'):
        class_ids = group['class_id'].values
        class_counts = defaultdict(int)
        for c in class_ids: class_counts[c] += 1
        total = len(class_ids)
        records.append({
            'filename': fname,
            'case_id': group['case_id'].iloc[0],
            'n_cells': total,
            'has_mitosis': int(0 in class_counts),
            'has_apoptosis': int(1 in class_counts),
            'has_tubule': int(4 in class_counts),
            'has_non_tubule': int(5 in class_counts),
            'cell_density': total/(IMAGE_SIZE[0]*IMAGE_SIZE[1])*1e6,
        })
    return pd.DataFrame(records)

# ============================================================
# Graph builder (compact)
# ============================================================
class CellSpatialGraph:
    def __init__(self, k_nn=8, delaunay=True, max_dist=300.0):
        self.k_nn, self.use_delaunay, self.max_dist = k_nn, delaunay, max_dist

    def extract_node_features(self, cells):
        n = len(cells)
        features = np.zeros((n,15), dtype=np.float32)
        for i, cell in enumerate(cells):
            cls_oh = np.zeros(N_CLASSES); cls_oh[cell['class_id']]=1.0
            features[i,:6]=cls_oh
            features[i,6]=cell['x_center']; features[i,7]=cell['y_center']
            features[i,8]=cell['width']; features[i,9]=cell['height']
            features[i,10]=np.log1p(cell['area_abs'])/15.0
            w,h=cell['width_abs'],cell['height_abs']
            features[i,13]=min(w,h)/max(w,h) if max(w,h)>0 else 1.0
            features[i,14]=cell['area_abs']/(w*h) if (w*h)>0 else 1.0
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        if n>=2:
            tree=KDTree(coords); k=min(self.k_nn,n)
            dists,_=tree.query(coords,k=k+1)
            ld=1.0/(np.mean(dists[:,1:],axis=1)+1e-6) if k>1 else np.ones(n)
            features[:,11]=ld/(ld.max()+1e-6)
        img_c=np.array([IMAGE_SIZE[0]/2,IMAGE_SIZE[1]/2])
        dc=np.linalg.norm(coords-img_c,axis=1)
        features[:,12]=dc/(dc.max()+1e-6)
        return features

    def build_edges(self, coords):
        edge_list=set(); n=len(coords)
        if self.use_delaunay and n>=4:
            try:
                tri=Delaunay(coords)
                for s in tri.simplices:
                    for i in range(3):
                        for j in range(i+1,3):
                            u,v=s[i],s[j]
                            if np.linalg.norm(coords[u]-coords[v])<=self.max_dist:
                                edge_list.add((u,v)); edge_list.add((v,u))
            except: pass
        if n>=2:
            tree=KDTree(coords); k=min(self.k_nn,n-1)
            if k>0:
                _,indices=tree.query(coords,k=k+1)
                for i in range(n):
                    for j in indices[i,1:]:
                        if np.linalg.norm(coords[i]-coords[j])<=self.max_dist:
                            edge_list.add((i,j)); edge_list.add((j,i))
        if not edge_list: edge_list={(i,i) for i in range(n)}
        return np.array(sorted(edge_list)).T

    def build_graph(self, cells):
        x=self.extract_node_features(cells)
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        ei=self.build_edges(coords)
        return Data(x=torch.tensor(x), edge_index=torch.tensor(ei,dtype=torch.long),
                    pos=torch.tensor(coords), n_nodes=len(cells))

# ============================================================
# TDA extractor (compact)
# ============================================================
class TopologicalFeatureExtractor:
    def __init__(self, resolution=50, sigma=0.05):
        self.resolution, self.sigma = resolution, sigma

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

    def persistence_image(self, diagram):
        if not diagram: return np.zeros((self.resolution,self.resolution))
        finite=[(b,d) for b,d in diagram if d<float('inf') and d-b>1e-6]
        if not finite: return np.zeros((self.resolution,self.resolution))
        births=np.array([p[0] for p in finite]); persists=np.array([p[1]-p[0] for p in finite])
        max_b=births.max() if len(births)>0 else 1; max_p=persists.max() if len(persists)>0 else 1
        if max_b==0: max_b=1
        if max_p==0: max_p=1
        image=np.zeros((self.resolution,self.resolution))
        for b,d in finite:
            pers=d-b; bx=int(b/max_b*(self.resolution-1)); py=int(pers/max_p*(self.resolution-1))
            bx=min(bx,self.resolution-1); py=min(py,self.resolution-1)
            sp=max(1,int(self.sigma*self.resolution))
            for xi in range(max(0,bx-sp),min(self.resolution,bx+sp+1)):
                for yi in range(max(0,py-sp),min(self.resolution,py+sp+1)):
                    image[xi,yi]+=np.exp(-((xi-bx)**2+(yi-py)**2)/(2*sp**2))
        if image.max()>0: image/=image.max()
        return image

    def betti_curve(self, diagram, n_bins=50):
        if not diagram: return np.zeros(n_bins)
        finite=[(b,d) for b,d in diagram if d<float('inf')]
        if not finite: return np.zeros(n_bins)
        max_val=max(max(d for _,d in finite),max(b for b,_ in finite))
        if max_val==0: max_val=1
        bins=np.linspace(0,max_val,n_bins)
        return np.array([sum(1 for b,d in finite if b<=t<d) for t in bins])

    def _diagram_stats(self, diagram):
        if not diagram: return np.zeros(6)
        finite=[(b,d) for b,d in diagram if d<float('inf')]
        if not finite: return np.zeros(6)
        persists=np.array([d-b for b,d in finite])
        births=np.array([b for b,_ in finite]); deaths=np.array([d for _,d in finite])
        return np.array([np.mean(persists), np.std(persists) if len(persists)>1 else 0,
                         np.max(persists), len(finite), np.mean(births), np.mean(deaths)])

    def extract_features(self, coords):
        r=self.compute_persistence(coords)
        pi0=self.persistence_image(r['h0_diagram']); pi1=self.persistence_image(r['h1_diagram'])
        bc0=self.betti_curve(r['h0_diagram']); bc1=self.betti_curve(r['h1_diagram'])
        s0=self._diagram_stats(r['h0_diagram']); s1=self._diagram_stats(r['h1_diagram'])
        return np.concatenate([pi0.flatten(),pi1.flatten(),bc0,bc1,s0,s1])

# ============================================================
# GNN Models
# ============================================================
class GATBlock(nn.Module):
    def __init__(self,in_dim,out_dim,heads=4,dropout=0.3):
        super().__init__()
        self.conv=GATv2Conv(in_dim,out_dim//heads,heads=heads,dropout=dropout)
        self.bn=nn.BatchNorm1d(out_dim); self.dropout=nn.Dropout(dropout)
    def forward(self,x,ei):
        x=self.conv(x,ei); x=self.bn(x); x=F.elu(x); x=self.dropout(x); return x

class TopoGNN_MultiLabel(nn.Module):
    def __init__(self,node_dim=15,topo_dim=5112,hidden_dim=128,n_tasks=3,dropout=0.3):
        super().__init__()
        self.gnn_layers=nn.ModuleList([
            GATBlock(node_dim,hidden_dim,heads=4,dropout=dropout),
            GATBlock(hidden_dim,hidden_dim,heads=4,dropout=dropout),
            GATBlock(hidden_dim,hidden_dim*2,heads=4,dropout=dropout),
        ])
        self.pool=SAGPooling(hidden_dim*2,ratio=0.5)
        self.topo_encoder=nn.Sequential(
            nn.Linear(topo_dim,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(256,128),
        )
        self.fusion=nn.Sequential(
            nn.Linear(hidden_dim*4+128,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(dropout),
        )
        self.classifiers=nn.ModuleList([nn.Linear(128,1) for _ in range(n_tasks)])

    def forward(self,data,topo_features):
        x,ei,batch=data.x,data.edge_index,data.batch
        for layer in self.gnn_layers: x=layer(x,ei)
        x,ei,_,batch,_,_=self.pool(x,ei,None,batch)
        gf=torch.cat([global_mean_pool(x,batch),global_max_pool(x,batch)],dim=1)
        tf=self.topo_encoder(topo_features)
        fused=self.fusion(torch.cat([gf,tf],dim=1))
        return torch.cat([clf(fused) for clf in self.classifiers],dim=1)

class GraphDataset(Dataset):
    def __init__(self,cell_df,case_list,gb,te,img_labels_dict):
        super().__init__()
        self.ig=cell_df[cell_df['case_id'].isin(case_list)].groupby('filename')
        self.filenames=[f for f in self.ig.groups.keys() if f in img_labels_dict]
        self.graphs,self.topo_feats,self.labels=[],[],[]
        print(f"Building graphs for {len(self.filenames)} images...")
        for i,fname in enumerate(self.filenames):
            group=self.ig.get_group(fname); cells=group.to_dict('records')
            self.graphs.append(gb.build_graph(cells))
            coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
            self.topo_feats.append(torch.tensor(te.extract_features(coords),dtype=torch.float))
            lbl=img_labels_dict[fname]
            self.labels.append(torch.tensor([lbl['has_mitosis'],lbl['has_apoptosis'],lbl['has_tubule']],dtype=torch.float))
            if (i+1)%30==0: print(f"  {i+1}/{len(self.filenames)}")
    def len(self): return len(self.graphs)
    def get(self,idx): return self.graphs[idx],self.topo_feats[idx],self.labels[idx]

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("方案一 v2: TopoGNN 多标签图分类 + TDA区分力验证")
    print("="*60)

    cell_df=pd.read_csv(OUTPUT_DIR/'all_cells.csv')
    img_labels=build_meaningful_labels(cell_df)
    img_labels_dict=img_labels.set_index('filename').to_dict('index')

    print(f"\nLabel distribution:")
    for col in ['has_mitosis','has_apoptosis','has_tubule']:
        pos=img_labels[col].sum()
        print(f"  {col}: {pos}/{len(img_labels)} ({pos/len(img_labels)*100:.1f}%)")

    graph_builder=CellSpatialGraph(k_nn=8,delaunay=True,max_dist=300.0)
    topo_ext=TopologicalFeatureExtractor(resolution=50)

    # ============ Experiment 1: TDA discriminative power ============
    print("\n"+"="*60)
    print("实验1: 拓扑特征统计检验 (Mann-Whitney U)")
    print("="*60)

    topo_features={}
    for fname,group in cell_df.groupby('filename'):
        cells=group.to_dict('records')
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        topo_features[fname]=topo_ext.extract_features(coords)

    results1=[]
    for label_name in ['has_mitosis','has_apoptosis','has_tubule']:
        pos_feat=[]; neg_feat=[]
        for fname,feat in topo_features.items():
            if fname in img_labels_dict:
                if img_labels_dict[fname][label_name]==1: pos_feat.append(feat)
                else: neg_feat.append(feat)
        pos_arr=np.array(pos_feat); neg_arr=np.array(neg_feat)
        n_dims=pos_arr.shape[1]
        p_values=np.array([mannwhitneyu(pos_arr[:,d],neg_arr[:,d],alternative='two-sided')[1]
                           if pos_arr[:,d].std()>0 and neg_arr[:,d].std()>0 else 1.0
                           for d in range(n_dims)])
        n_sig=(p_values<0.05).sum(); n_bonf=(p_values<0.05/n_dims).sum()
        results1.append({'label':label_name,'n_pos':len(pos_feat),'n_neg':len(neg_feat),
                         'sig_p05':n_sig,'sig_bonf':n_bonf,'min_p':p_values.min(),
                         'median_p':np.median(p_values),'total_dims':n_dims})
        print(f"  {label_name}: {n_sig}/{n_dims} sig dims (p<0.05), {n_bonf} (Bonferroni), min_p={p_values.min():.2e}")

    df1=pd.DataFrame(results1)
    df1.to_csv(OUTPUT_DIR/'experiment1_tda_discrimination.csv',index=False)

    # ============ Experiment 2: Feature comparison ============
    print("\n"+"="*60)
    print("实验2: 拓扑特征 vs 传统特征 分类性能")
    print("="*60)

    trad_features={}
    for fname,group in cell_df.groupby('filename'):
        cells=group.to_dict('records')
        areas=np.array([c['area_abs'] for c in cells])
        widths=np.array([c['width_abs'] for c in cells])
        heights=np.array([c['height_abs'] for c in cells])
        cids=np.array([c['class_id'] for c in cells])
        trad_features[fname]=np.array([
            len(cells), np.mean(areas),np.std(areas),
            np.mean(widths),np.std(widths), np.mean(heights),np.std(heights),
            np.percentile(areas,25),np.percentile(areas,75),
            np.sum(cids==0),np.sum(cids==1), np.sum(cids==2)/len(cells),
            np.sum(cids==4),np.sum(cids==5),
        ])

    common=sorted(set(topo_features)&set(trad_features)&set(img_labels_dict))
    X_trad=np.array([trad_features[f] for f in common])
    X_topo=np.array([topo_features[f] for f in common])
    pca=PCA(n_components=50); X_topo_r=pca.fit_transform(X_topo)
    X_comb_r=np.concatenate([X_trad,X_topo_r],axis=1)
    print(f"Trad: {X_trad.shape[1]}d, Topo: {X_topo.shape[1]}d→{X_topo_r.shape[1]}d (PCA), Comb: {X_comb_r.shape[1]}d")

    tasks={'has_mitosis':'Mitosis','has_apoptosis':'Apoptosis','has_tubule':'Tubule'}
    results2=[]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    for tk,tn in tasks.items():
        y=np.array([img_labels_dict[f][tk] for f in common])
        print(f"\n--- {tn} (pos={y.mean():.1%}) ---")
        X_trad_s=StandardScaler().fit_transform(X_trad)
        X_topo_s=StandardScaler().fit_transform(X_topo_r)
        X_comb_s=StandardScaler().fit_transform(X_comb_r)

        for fn,X in [('Traditional',X_trad_s),('Topological',X_topo_s),('Combined',X_comb_s)]:
            rf=RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')
            rf_f1=cross_val_score(rf,X,y,cv=cv,scoring='f1')
            rf_auc=cross_val_score(rf,X,y,cv=cv,scoring='roc_auc')
            gb=GradientBoostingClassifier(n_estimators=100,random_state=42)
            gb_f1=cross_val_score(gb,X,y,cv=cv,scoring='f1')
            gb_auc=cross_val_score(gb,X,y,cv=cv,scoring='roc_auc')
            results2.append({'task':tn,'feature':fn,
                             'RF_F1':rf_f1.mean(),'RF_F1_std':rf_f1.std(),
                             'RF_AUC':rf_auc.mean(),'RF_AUC_std':rf_auc.std(),
                             'GB_F1':gb_f1.mean(),'GB_F1_std':gb_f1.std(),
                             'GB_AUC':gb_auc.mean(),'GB_AUC_std':gb_auc.std()})
            print(f"  {fn:15s} | RF F1={rf_f1.mean():.3f}±{rf_f1.std():.3f} AUC={rf_auc.mean():.3f}±{rf_auc.std():.3f} | GB F1={gb_f1.mean():.3f}±{gb_f1.std():.3f} AUC={gb_auc.mean():.3f}±{gb_auc.std():.3f}")

    df2=pd.DataFrame(results2)
    df2.to_csv(OUTPUT_DIR/'experiment2_feature_comparison.csv',index=False)

    # ============ Experiment 3: TopoGNN multilabel ============
    print("\n"+"="*60)
    print("实验3: TopoGNN 多标签图分类")
    print("="*60)

    with open(OUTPUT_DIR/'data_split.json') as f: split=json.load(f)

    train_ds=GraphDataset(cell_df,split['train_cases'],graph_builder,topo_ext,img_labels_dict)
    val_ds=GraphDataset(cell_df,split['val_cases'],graph_builder,topo_ext,img_labels_dict)
    test_ds=GraphDataset(cell_df,split['test_cases'],graph_builder,topo_ext,img_labels_dict)

    def collate(batch):
        graphs,topos,labels=zip(*batch)
        return Batch.from_data_list(graphs),torch.stack(topos),torch.stack(labels)

    train_loader=PyGLoader(list(zip(train_ds.graphs,train_ds.topo_feats,train_ds.labels)),
                            batch_size=8,shuffle=True,collate_fn=collate)
    val_loader=PyGLoader(list(zip(val_ds.graphs,val_ds.topo_feats,val_ds.labels)),
                          batch_size=8,shuffle=False,collate_fn=collate)
    test_loader=PyGLoader(list(zip(test_ds.graphs,test_ds.topo_feats,test_ds.labels)),
                           batch_size=8,shuffle=False,collate_fn=collate)

    print("\n--- TopoGNN (GNN + TDA) ---")
    model=TopoGNN_MultiLabel(node_dim=15,topo_dim=5112,hidden_dim=128,n_tasks=3).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0,2.0,1.5]).to(DEVICE))

    best_vauc=0; history={'loss':[],'val_auc':[]}
    for epoch in range(100):
        model.train(); tl=0
        for data,topo,labels in train_loader:
            data,topo,labels=data.to(DEVICE),topo.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad()
            loss=criterion(model(data,topo),labels)
            loss.backward(); optimizer.step(); tl+=loss.item()
        model.eval(); vp,vl=[],[]
        with torch.no_grad():
            for data,topo,labels in val_loader:
                data,topo=data.to(DEVICE),topo.to(DEVICE)
                vp.append(torch.sigmoid(model(data,topo)).cpu().numpy())
                vl.append(labels.numpy())
        vp=np.vstack(vp); vl=np.vstack(vl)
        try:
            vauc=roc_auc_score(vl,vp,average='macro')
        except ValueError:
            vauc=0.5
        history['loss'].append(tl/len(train_loader)); history['val_auc'].append(vauc)
        if vauc>best_vauc: best_vauc=vauc; torch.save(model.state_dict(),OUTPUT_DIR/'topognn_multilabel_best.pt')
        if (epoch+1)%20==0: print(f"  Epoch {epoch+1:3d} | Loss: {tl/len(train_loader):.4f} | Val AUC: {vauc:.4f}")

    model.load_state_dict(torch.load(OUTPUT_DIR/'topognn_multilabel_best.pt'))
    model.eval(); tp,tl2=[],[]
    with torch.no_grad():
        for data,topo,labels in test_loader:
            data,topo=data.to(DEVICE),topo.to(DEVICE)
            tp.append(torch.sigmoid(model(data,topo)).cpu().numpy())
            tl2.append(labels.numpy())
    tp=np.vstack(tp); tl2=np.vstack(tl2)

    task_names=['Mitosis','Apoptosis','Tubule']
    print("\nTopoGNN Test Results:")
    topognn_res={}
    for i,name in enumerate(task_names):
        try:
            auc=roc_auc_score(tl2[:,i],tp[:,i])
        except ValueError:
            auc=float('nan')
        f1=f1_score(tl2[:,i],(tp[:,i]>0.5).astype(int),zero_division=0)
        topognn_res[name]={'AUC':auc,'F1':f1}
        print(f"  {name}: AUC={auc:.4f}, F1={f1:.4f}")

    # Baseline RF
    print("\n--- Baseline: RF on topology PCA ---")
    train_topo=np.array([t.numpy() for t in train_ds.topo_feats])
    train_lbl=np.array([l.numpy() for l in train_ds.labels])
    test_topo=np.array([t.numpy() for t in test_ds.topo_feats])
    test_lbl=np.array([l.numpy() for l in test_ds.labels])
    pca2=PCA(n_components=50)
    train_topo_p=pca2.fit_transform(train_topo)
    test_topo_p=pca2.transform(test_topo)

    rf_res={}
    for i,name in enumerate(task_names):
        rf=RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')
        rf.fit(train_topo_p,train_lbl[:,i])
        rp=rf.predict_proba(test_topo_p)[:,1]
        try:
            auc=roc_auc_score(test_lbl[:,i],rp)
        except ValueError:
            auc=float('nan')
        f1=f1_score(test_lbl[:,i],(rp>0.5).astype(int),zero_division=0)
        rf_res[name]={'AUC':auc,'F1':f1}
        print(f"  {name}: AUC={auc:.4f}, F1={f1:.4f}")

    # Summary
    print("\n"+"="*60)
    print("FINAL COMPARISON: TopoGNN vs RF Baseline")
    print("="*60)
    print(f"{'Task':<15} {'Metric':<8} {'TopoGNN':<10} {'RF(Topo)':<10} {'Δ':<10}")
    print("-"*53)
    for name in task_names:
        for metric in ['AUC','F1']:
            tv=topognn_res[name][metric]; rv=rf_res[name][metric]
            print(f"{name:<15} {metric:<8} {tv:<10.4f} {rv:<10.4f} {tv-rv:<+.4f}")

    # Save
    final_results={'topognn':topognn_res,'rf_baseline':rf_res}
    with open(OUTPUT_DIR/'topognn_v2_results.json','w') as f: json.dump(final_results,f,indent=2)

    # Plot
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for i,name in enumerate(task_names):
        axes[i].plot(history['val_auc'],label='Val AUC',linewidth=2)
        axes[i].set_title(f'{name} Detection'); axes[i].set_xlabel('Epoch'); axes[i].set_ylabel('AUC')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/'topognn_v2_training.png',dpi=150); plt.close()

    # Feature importance bar chart
    fig,ax=plt.subplots(figsize=(12,6))
    tasks_list=['Mitosis','Apoptosis','Tubule']
    feat_types=['Traditional','Topological','Combined']
    x=np.arange(len(tasks_list)); width=0.25
    for i,ft in enumerate(feat_types):
        vals=[df2[(df2['task']==t)&(df2['feature']==ft)]['RF_AUC'].values[0] for t in tasks_list]
        ax.bar(x+i*width,vals,width,label=ft,edgecolor='black')
    ax.set_xticks(x+width); ax.set_xticklabels(tasks_list)
    ax.set_ylabel('RF AUC (5-fold CV)'); ax.set_title('Feature Type Comparison: Traditional vs Topological vs Combined')
    ax.legend(); ax.set_ylim(0,1)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/'feature_comparison_bars.png',dpi=150); plt.close()

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("Done!")

if __name__=='__main__':
    main()
