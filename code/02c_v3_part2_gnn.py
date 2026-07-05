"""
方案一 v3 Part B: 节点级GNN + Backbone对比 + 拓扑注意力可解释性
运行: python3 02c_v3_part2_gnn.py
"""

import os, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv
from torch_geometric.loader import DataLoader as PyGLoader

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

CLASS_NAMES = {0:'Mitosis',1:'Apoptosis',2:'Tumor_nuclei',3:'Non_tumor_nuclei',4:'Tubule',5:'Non_tubule'}
N_CLASSES=6; IMAGE_SIZE=(1360,1024); SEED=42
np.random.seed(SEED); torch.manual_seed(SEED)

# ============================================================
# Compact graph + topo builders (reuse from part1)
# ============================================================
class CellSpatialGraph:
    def __init__(self,k_nn=8,delaunay=True,max_dist=300.0):
        self.k_nn,self.use_delaunay,self.max_dist=k_nn,delaunay,max_dist
    def extract_node_features(self,cells):
        n=len(cells); features=np.zeros((n,15),dtype=np.float32)
        for i,cell in enumerate(cells):
            oh=np.zeros(N_CLASSES); oh[cell['class_id']]=1.0
            features[i,:6]=oh
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
    def build_edges(self,coords):
        el=set(); n=len(coords)
        if self.use_delaunay and n>=4:
            try:
                tri=Delaunay(coords)
                for s in tri.simplices:
                    for i in range(3):
                        for j in range(i+1,3):
                            u,v=s[i],s[j]
                            if np.linalg.norm(coords[u]-coords[v])<=self.max_dist:
                                el.add((u,v)); el.add((v,u))
            except: pass
        if n>=2:
            tree=KDTree(coords); k=min(self.k_nn,n-1)
            if k>0:
                _,indices=tree.query(coords,k=k+1)
                for i in range(n):
                    for j in indices[i,1:]:
                        if np.linalg.norm(coords[i]-coords[j])<=self.max_dist:
                            el.add((i,j)); el.add((j,i))
        if not el: el={(i,i) for i in range(n)}
        return np.array(sorted(el)).T
    def build_graph(self,cells):
        x=self.extract_node_features(cells)
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        ei=self.build_edges(coords)
        return Data(x=torch.tensor(x),edge_index=torch.tensor(ei,dtype=torch.long),
                    pos=torch.tensor(coords),n_nodes=len(cells))

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
    def _diagram_stats(self,diagram):
        if not diagram: return np.zeros(6)
        finite=[(b,d) for b,d in diagram if d<float('inf')]
        if not finite: return np.zeros(6)
        persists=np.array([d-b for b,d in finite])
        births=np.array([b for b,_ in finite]); deaths=np.array([d for _,d in finite])
        return np.array([np.mean(persists),np.std(persists) if len(persists)>1 else 0,
                         np.max(persists),len(finite),np.mean(births),np.mean(deaths)])
    def _betti_integral(self,diagram,n_bins=50):
        if not diagram: return 0.0
        finite=[(b,d) for b,d in diagram if d<float('inf')]
        if not finite: return 0.0
        max_val=max(max(d for _,d in finite),max(b for b,_ in finite))
        if max_val==0: return 0.0
        bins=np.linspace(0,max_val,n_bins)
        curve=np.array([sum(1 for b,d in finite if b<=t<d) for t in bins])
        return np.trapz(curve,bins)/(n_bins*max_val)
    def _multiscale_features(self,coords):
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
            n_edges=np.sum(adj)/2; n_cycles=max(0,n_edges-n+n_comp)
            feats.append(min(n_cycles,50))
        return np.array(feats)
    def extract_features(self,coords):
        r=self.compute_persistence(coords)
        h0s=self._diagram_stats(r['h0_diagram']); h1s=self._diagram_stats(r['h1_diagram'])
        h0i=self._betti_integral(r['h0_diagram']); h1i=self._betti_integral(r['h1_diagram'])
        ms=self._multiscale_features(coords)
        feats=np.concatenate([h0s,h1s,[h0i,h1i],ms])
        n=r['n_points']
        if n>0:
            tp_h0=sum(d-b for b,d in r['h0_diagram'] if d<float('inf'))
            tp_h1=sum(d-b for b,d in r['h1_diagram'] if d<float('inf'))
            feats=np.append(feats,[tp_h0/max(n,1),tp_h1/max(n,1)])
        return feats

# ============================================================
# Node-level labels
# ============================================================
def build_node_labels(cell_df, radius=200):
    records=[]
    for fname,group in cell_df.groupby('filename'):
        cells=group.to_dict('records')
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        cids=np.array([c['class_id'] for c in cells])
        t_mit=KDTree(coords[cids==0]) if (cids==0).sum()>0 else None
        t_apo=KDTree(coords[cids==1]) if (cids==1).sum()>0 else None
        t_tub=KDTree(coords[cids==4]) if (cids==4).sum()>0 else None
        for i,cell in enumerate(cells):
            nm=0; na=0; nt=0
            if t_mit is not None:
                d,_=t_mit.query(coords[i:i+1],k=1); nm=int(d[0]<radius)
            if t_apo is not None:
                d,_=t_apo.query(coords[i:i+1],k=1); na=int(d[0]<radius)
            if t_tub is not None:
                d,_=t_tub.query(coords[i:i+1],k=1); nt=int(d[0]<radius)
            records.append({'filename':fname,'case_id':cell.get('case_id',fname.split('-')[0]),
                            'cell_idx':i,'class_id':cell['class_id'],
                            'near_mitosis':nm,'near_apoptosis':na,'near_tubule':nt})
    return pd.DataFrame(records)

# ============================================================
# GNN Models
# ============================================================
class GNNEncoder(nn.Module):
    BACKBONES={'gcn':GCNConv,'gat':GATv2Conv,'gin':GINConv,'sage':SAGEConv}
    def __init__(self,backbone='gat',node_dim=15,hidden_dim=128,num_layers=3,dropout=0.3):
        super().__init__()
        self.backbone_name=backbone
        conv_cls=self.BACKBONES[backbone]
        self.layers=nn.ModuleList()
        in_dim=node_dim
        for i in range(num_layers):
            out_dim=hidden_dim if i<num_layers-1 else hidden_dim*2
            if backbone=='gin':
                mlp=nn.Sequential(nn.Linear(in_dim,out_dim),nn.ReLU(),nn.Linear(out_dim,out_dim))
                self.layers.append(conv_cls(mlp))
            elif backbone=='gat':
                self.layers.append(conv_cls(in_dim,out_dim//4,heads=4,dropout=dropout))
            else:
                self.layers.append(conv_cls(in_dim,out_dim))
            in_dim=out_dim
        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_dim if i<num_layers-1 else hidden_dim*2) for i in range(num_layers)])
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,ei):
        for i,(layer,bn) in enumerate(zip(self.layers,self.bns)):
            x=layer(x,ei); x=bn(x)
            if i<len(self.layers)-1: x=F.relu(x); x=self.dropout(x)
        return x

class TopologicalAttention(nn.Module):
    def __init__(self,topo_dim,hidden_dim=64):
        super().__init__()
        self.attention=nn.Sequential(nn.Linear(topo_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,topo_dim),nn.Sigmoid())
    def forward(self,topo):
        w=self.attention(topo); return topo*w, w

class TopoGNN_Node(nn.Module):
    def __init__(self,backbone='gat',node_dim=15,topo_dim=26,hidden_dim=128,n_tasks=3,dropout=0.3):
        super().__init__()
        self.backbone_name=backbone
        self.gnn=GNNEncoder(backbone=backbone,node_dim=node_dim,hidden_dim=hidden_dim,dropout=dropout)
        self.topo_attn=TopologicalAttention(topo_dim,hidden_dim=64)
        fd=hidden_dim*2+topo_dim
        self.fusion=nn.Sequential(nn.Linear(fd,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                   nn.Linear(hidden_dim,hidden_dim//2),nn.ReLU())
        self.classifiers=nn.ModuleList([nn.Linear(hidden_dim//2,1) for _ in range(n_tasks)])
    def forward(self,data,topo):
        node_emb=self.gnn(data.x,data.edge_index)
        topo_w,attn_w=self.topo_attn(topo)
        fused=self.fusion(torch.cat([node_emb,topo_w],dim=1))
        return torch.cat([clf(fused) for clf in self.classifiers],dim=1), attn_w

class NodeDataset(Dataset):
    def __init__(self,cell_df,node_df,case_list,gb,te):
        super().__init__()
        self.node_df=node_df[node_df['case_id'].isin(case_list)].copy()
        self.graphs={}; self.topo_feats={}; self.node_map={}; self.labels=[]
        img_groups=cell_df[cell_df['case_id'].isin(case_list)].groupby('filename')
        filenames=sorted(img_groups.groups.keys())
        print(f"  Building {len(filenames)} graphs...")
        gidx=0
        for gi,fname in enumerate(filenames):
            group=img_groups.get_group(fname); cells=group.to_dict('records')
            self.graphs[fname]=gb.build_graph(cells)
            coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
            self.topo_feats[fname]=torch.tensor(te.extract_features(coords),dtype=torch.float)
            subset=self.node_df[self.node_df['filename']==fname]
            for _,row in subset.iterrows():
                self.node_map[gidx]=(fname,row['cell_idx'])
                self.labels.append([row['near_mitosis'],row['near_apoptosis'],row['near_tubule']])
                gidx+=1
            if (gi+1)%30==0: print(f"    {gi+1}/{len(filenames)}")
        self.n_samples=len(self.labels)
    def len(self): return self.n_samples
    def get(self,idx):
        fname,li=self.node_map[idx]
        return {'node_feat':self.graphs[fname].x[li],'topo_feat':self.topo_feats[fname],
                'label':torch.tensor(self.labels[idx],dtype=torch.float),'filename':fname}

# ============================================================
# Training & Evaluation
# ============================================================
def train_node_model(model,train_loader,val_loader,test_loader,backbone_name,topo_dim):
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    pw=torch.tensor([5.0,3.0,2.0]).to(DEVICE)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pw)
    best_vauc=0; history={'loss':[],'val_auc':[]}
    for epoch in range(50):
        model.train(); tl=0
        for node_feat,topo_feat,labels in train_loader:
            node_feat=node_feat.to(DEVICE); topo_feat=topo_feat.to(DEVICE); labels=labels.to(DEVICE)
            optimizer.zero_grad()
            logits,_=model(Data(x=node_feat,edge_index=torch.zeros((2,0),dtype=torch.long).to(DEVICE)),topo_feat)
            loss=criterion(logits,labels); loss.backward(); optimizer.step(); tl+=loss.item()
        model.eval(); vp,vl=[],[]
        with torch.no_grad():
            for node_feat,topo_feat,labels in val_loader:
                node_feat=node_feat.to(DEVICE); topo_feat=topo_feat.to(DEVICE)
                logits,_=model(Data(x=node_feat,edge_index=torch.zeros((2,0),dtype=torch.long).to(DEVICE)),topo_feat)
                vp.append(torch.sigmoid(logits).cpu().numpy()); vl.append(labels.numpy())
        vp=np.vstack(vp); vl=np.vstack(vl)
        try: vauc=roc_auc_score(vl,vp,average='macro')
        except: vauc=0.5
        history['loss'].append(tl/len(train_loader)); history['val_auc'].append(vauc)
        if vauc>best_vauc: best_vauc=vauc; torch.save(model.state_dict(),OUTPUT_DIR/f'nodes_{backbone_name}_best.pt')
    # Test
    model.load_state_dict(torch.load(OUTPUT_DIR/f'nodes_{backbone_name}_best.pt'))
    model.eval(); tp,tl2=[],[]
    with torch.no_grad():
        for node_feat,topo_feat,labels in test_loader:
            node_feat=node_feat.to(DEVICE); topo_feat=topo_feat.to(DEVICE)
            logits,_=model(Data(x=node_feat,edge_index=torch.zeros((2,0),dtype=torch.long).to(DEVICE)),topo_feat)
            tp.append(torch.sigmoid(logits).cpu().numpy()); tl2.append(labels.numpy())
    tp=np.vstack(tp); tl2=np.vstack(tl2)
    task_names=['Mitosis_Nearby','Apoptosis_Nearby','Tubule_Nearby']
    res={}
    for i,name in enumerate(task_names):
        try: auc=roc_auc_score(tl2[:,i],tp[:,i])
        except: auc=float('nan')
        f1=f1_score(tl2[:,i],(tp[:,i]>0.5).astype(int),zero_division=0)
        res[name]={'AUC':auc,'F1':f1}
    return res, history

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("方案一 v3 Part B: 节点级GNN + Backbone对比")
    print("="*60)
    cell_df=pd.read_csv(OUTPUT_DIR/'all_cells.csv')
    with open(OUTPUT_DIR/'data_split.json') as f: split=json.load(f)
    node_df=build_node_labels(cell_df,radius=200)
    print(f"Node labels: {len(node_df)} total")
    for col in ['near_mitosis','near_apoptosis','near_tubule']:
        p=node_df[col].sum(); print(f"  {col}: {p}/{len(node_df)} ({p/len(node_df)*100:.1f}%)")

    gb=CellSpatialGraph(k_nn=8,delaunay=True,max_dist=300.0)
    te=CompactTopoFeatures(radii=[50,100,200,300,400])
    topo_dim=len(te.extract_features(np.array([[0,0],[100,100],[200,200]])))
    print(f"Topo dim: {topo_dim}")

    train_ds=NodeDataset(cell_df,node_df,split['train_cases'],gb,te)
    val_ds=NodeDataset(cell_df,node_df,split['val_cases'],gb,te)
    test_ds=NodeDataset(cell_df,node_df,split['test_cases'],gb,te)
    print(f"Samples: train={train_ds.n_samples}, val={val_ds.n_samples}, test={test_ds.n_samples}")

    def collate(batch):
        nf=torch.stack([b['node_feat'] for b in batch])
        tf=torch.stack([b['topo_feat'] for b in batch])
        lb=torch.stack([b['label'] for b in batch])
        return nf,tf,lb

    from torch.utils.data import DataLoader as TorchLoader
    train_loader=TorchLoader([train_ds[i] for i in range(train_ds.n_samples)],batch_size=256,shuffle=True,collate_fn=collate)
    val_loader=TorchLoader([val_ds[i] for i in range(val_ds.n_samples)],batch_size=256,shuffle=False,collate_fn=collate)
    test_loader=TorchLoader([test_ds[i] for i in range(test_ds.n_samples)],batch_size=256,shuffle=False,collate_fn=collate)

    # Backbone comparison
    print("\n--- Backbone Comparison ---")
    backbone_results={}
    for bb in ['gcn','gat','gin','sage']:
        print(f"\n  Training {bb.upper()}...")
        model=TopoGNN_Node(backbone=bb,node_dim=15,topo_dim=topo_dim,hidden_dim=128,n_tasks=3).to(DEVICE)
        res,hist=train_node_model(model,train_loader,val_loader,test_loader,bb,topo_dim)
        backbone_results[bb]=res
        for tn,metrics in res.items():
            print(f"    {tn}: AUC={metrics['AUC']:.4f}, F1={metrics['F1']:.4f}")

    # Summary
    print("\n"+"="*60)
    print("BACKBONE COMPARISON SUMMARY")
    print("="*60)
    task_names=['Mitosis_Nearby','Apoptosis_Nearby','Tubule_Nearby']
    for tn in task_names:
        print(f"\n{tn}:")
        for bb in ['gcn','gat','gin','sage']:
            m=backbone_results[bb][tn]
            print(f"  {bb.upper():6s}: AUC={m['AUC']:.4f}, F1={m['F1']:.4f}")

    # Save
    with open(OUTPUT_DIR/'v3_backbone_comparison.json','w') as f:
        json.dump(backbone_results,f,indent=2)

    # Visualization
    fig,axes=plt.subplots(1,2,figsize=(16,5))
    x=np.arange(len(task_names)); width=0.2
    colors=['#3498db','#e74c3c','#2ecc71','#f39c12']
    for i,bb in enumerate(['gcn','gat','gin','sage']):
        vals=[backbone_results[bb][tn]['AUC'] for tn in task_names]
        axes[0].bar(x+i*width,vals,width,label=bb.upper(),color=colors[i],edgecolor='black')
    axes[0].set_xticks(x+1.5*width); axes[0].set_xticklabels(task_names,fontsize=8)
    axes[0].set_ylabel('AUC'); axes[0].set_title('GNN Backbone Comparison (Node-level)')
    axes[0].legend(fontsize=8)

    for i,bb in enumerate(['gcn','gat','gin','sage']):
        vals=[backbone_results[bb][tn]['F1'] for tn in task_names]
        axes[1].bar(x+i*width,vals,width,label=bb.upper(),color=colors[i],edgecolor='black')
    axes[1].set_xticks(x+1.5*width); axes[1].set_xticklabels(task_names,fontsize=8)
    axes[1].set_ylabel('F1'); axes[1].set_title('GNN Backbone Comparison (F1)')
    axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/'v3_backbone_comparison.png',dpi=150); plt.close()

    print(f"\nDone! Results in {OUTPUT_DIR}")

if __name__=='__main__':
    main()
