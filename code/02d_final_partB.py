"""
方案一 Final Part B: P0-1修复(完整图GNN) + P1-4(注意力可解释性)
运行: python3 02d_final_partB.py
"""
import os, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader as TorchLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
FIG_DIR = OUTPUT_DIR / 'final_figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

CLASS_NAMES={0:'Mitosis',1:'Apoptosis',2:'Tumor_nuclei',3:'Non_tumor_nuclei',4:'Tubule',5:'Non_tubule'}
N_CLASSES=6; IMAGE_SIZE=(1360,1024); SEED=42
np.random.seed(SEED); torch.manual_seed(SEED)

TOPO_DIM_NAMES=[
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
# Graph + Topo builders
# ============================================================
class CellSpatialGraph:
    def __init__(self,k_nn=8,delaunay=True,max_dist=300.0):
        self.k_nn,self.use_delaunay,self.max_dist=k_nn,delaunay,max_dist
    def extract_node_features(self,cells):
        n=len(cells); features=np.zeros((n,15),dtype=np.float32)
        for i,cell in enumerate(cells):
            oh=np.zeros(N_CLASSES); oh[cell['class_id']]=1.0
            features[i,:6]=oh; features[i,6]=cell['x_center']; features[i,7]=cell['y_center']
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

# ============================================================
# P0-1修复: GNN with full graph structure
# ============================================================
class GNNEncoder(nn.Module):
    BACKBONES={'gcn':GCNConv,'gat':GATv2Conv,'gin':GINConv,'sage':SAGEConv}
    def __init__(self,backbone='gat',node_dim=15,hidden_dim=128,nl=3,dropout=0.3):
        super().__init__(); self.bb=backbone; cc=self.BACKBONES[backbone]
        self.layers=nn.ModuleList(); idim=node_dim
        for i in range(nl):
            odim=hidden_dim if i<nl-1 else hidden_dim*2
            if backbone=='gin':
                mlp=nn.Sequential(nn.Linear(idim,odim),nn.ReLU(),nn.Linear(odim,odim))
                self.layers.append(cc(mlp))
            elif backbone=='gat': self.layers.append(cc(idim,odim//4,heads=4,dropout=dropout))
            else: self.layers.append(cc(idim,odim))
            idim=odim
        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_dim if i<nl-1 else hidden_dim*2) for i in range(nl)])
        self.do=nn.Dropout(dropout)
    def forward(self,x,ei):
        for i,(l,bn) in enumerate(zip(self.layers,self.bns)):
            x=l(x,ei); x=bn(x)
            if i<len(self.layers)-1: x=F.relu(x); x=self.do(x)
        return x

class TopologicalAttention(nn.Module):
    def __init__(self,td,hd=64):
        super().__init__()
        self.attn=nn.Sequential(nn.Linear(td,hd),nn.Tanh(),nn.Linear(hd,td),nn.Sigmoid())
    def forward(self,t):
        w=self.attn(t); return t*w, w

class TopoGNN_Full(nn.Module):
    """完整图结构的节点级TopoGNN"""
    def __init__(self,backbone='gat',nd=15,td=26,hd=128,nt=3,dropout=0.3):
        super().__init__(); self.bb=backbone
        self.gnn=GNNEncoder(backbone=backbone,node_dim=nd,hidden_dim=hd,dropout=dropout)
        self.ta=TopologicalAttention(td,64)
        fd=hd*2+td
        self.fusion=nn.Sequential(nn.Linear(fd,hd),nn.BatchNorm1d(hd),nn.ReLU(),nn.Dropout(dropout),
                                   nn.Linear(hd,hd//2),nn.ReLU())
        self.clfs=nn.ModuleList([nn.Linear(hd//2,1) for _ in range(nt)])
    def forward(self,data,topo):
        ne=self.gnn(data.x,data.edge_index)
        tw,aw=self.ta(topo)
        # Broadcast topo features to all nodes
        tw=tw.expand(ne.size(0),-1)
        fused=self.fusion(torch.cat([ne,tw],dim=1))
        return torch.cat([c(fused) for c in self.clfs],dim=1), aw

# ============================================================
# Node labels + Dataset
# ============================================================
def build_node_labels(cell_df,radius=200):
    records=[]
    for fname,group in cell_df.groupby('filename'):
        cells=group.to_dict('records')
        coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
        cids=np.array([c['class_id'] for c in cells])
        tm=KDTree(coords[cids==0]) if (cids==0).sum()>0 else None
        ta=KDTree(coords[cids==1]) if (cids==1).sum()>0 else None
        tt=KDTree(coords[cids==4]) if (cids==4).sum()>0 else None
        for i,cell in enumerate(cells):
            nm=0; na=0; nt=0
            if tm is not None: d,_=tm.query(coords[i:i+1],k=1); nm=int(d[0]<radius)
            if ta is not None: d,_=ta.query(coords[i:i+1],k=1); na=int(d[0]<radius)
            if tt is not None: d,_=tt.query(coords[i:i+1],k=1); nt=int(d[0]<radius)
            records.append({'filename':fname,'case_id':cell.get('case_id',fname.split('-')[0]),
                            'cell_idx':i,'class_id':cell['class_id'],
                            'near_mitosis':nm,'near_apoptosis':na,'near_tubule':nt})
    return pd.DataFrame(records)

class FullGraphNodeDataset(Dataset):
    """每个样本=完整的图，返回所有节点的嵌入+标签"""
    def __init__(self,cell_df,node_df,case_list,gb,te):
        super().__init__()
        self.node_df=node_df[node_df['case_id'].isin(case_list)].copy()
        self.graphs={}; self.topo_feats={}; self.node_labels={}
        ig=cell_df[cell_df['case_id'].isin(case_list)].groupby('filename')
        fnames=sorted(ig.groups.keys())
        print(f"  Building {len(fnames)} full-structure graphs...")
        for gi,fname in enumerate(fnames):
            group=ig.get_group(fname); cells=group.to_dict('records')
            self.graphs[fname]=gb.build_graph(cells)
            coords=np.array([[c['x_center_abs'],c['y_center_abs']] for c in cells])
            self.topo_feats[fname]=torch.tensor(te.extract_features(coords),dtype=torch.float)
            subset=self.node_df[self.node_df['filename']==fname]
            labels=torch.zeros((len(cells),3),dtype=torch.float)
            for _,row in subset.iterrows():
                labels[row['cell_idx']]=torch.tensor([row['near_mitosis'],row['near_apoptosis'],row['near_tubule']],dtype=torch.float)
            self.node_labels[fname]=labels
            if (gi+1)%30==0: print(f"    {gi+1}/{len(fnames)}")
        self.fnames=fnames
    def len(self): return len(self.fnames)
    def get(self,idx):
        fn=self.fnames[idx]
        return self.graphs[fn],self.topo_feats[fn],self.node_labels[fn],fn

# ============================================================
# Training
# ============================================================
def train_full_gnn(model,train_fnames,val_fnames,test_fnames,dataset,epochs=50):
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    pw=torch.tensor([5.0,3.0,2.0]).to(DEVICE)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pw)
    best_vauc=0; all_attn_weights=[]

    for epoch in range(epochs):
        model.train(); tl=0
        for fn in train_fnames:
            g,tf,labels,_=dataset[dataset.fnames.index(fn)]
            g=g.to(DEVICE); tf=tf.to(DEVICE).unsqueeze(0); labels=labels.to(DEVICE)
            optimizer.zero_grad()
            logits,aw=model(g,tf)
            loss=criterion(logits,labels)
            loss.backward(); optimizer.step(); tl+=loss.item()
        tl/=len(train_fnames)

        model.eval(); vp,vl=[],[]
        with torch.no_grad():
            for fn in val_fnames:
                g,tf,labels,_=dataset[dataset.fnames.index(fn)]
                g=g.to(DEVICE); tf=tf.to(DEVICE).unsqueeze(0)
                logits,_=model(g,tf)
                vp.append(torch.sigmoid(logits).cpu().numpy()); vl.append(labels.numpy())
        vp=np.vstack(vp); vl=np.vstack(vl)
        try: vauc=roc_auc_score(vl,vp,average='macro')
        except: vauc=0.5
        if vauc>best_vauc:
            best_vauc=vauc; torch.save(model.state_dict(),OUTPUT_DIR/f'fullgnn_{model.bb}_best.pt')
        if (epoch+1)%10==0: print(f"    Epoch {epoch+1}: Loss={tl:.4f}, Val AUC={vauc:.4f}")

    # Test + collect attention weights
    model.load_state_dict(torch.load(OUTPUT_DIR/f'fullgnn_{model.bb}_best.pt'))
    model.eval(); tp,tl2=[],[]
    with torch.no_grad():
        for fn in test_fnames:
            g,tf,labels,_=dataset[dataset.fnames.index(fn)]
            g=g.to(DEVICE); tf=tf.to(DEVICE).unsqueeze(0)
            logits,aw=model(g,tf)
            tp.append(torch.sigmoid(logits).cpu().numpy()); tl2.append(labels.numpy())
            all_attn_weights.append(aw.cpu().numpy().mean(0))
    tp=np.vstack(tp); tl2=np.vstack(tl2)
    mean_attn=np.mean(all_attn_weights,axis=0)

    tn=['Mitosis_Nearby','Apoptosis_Nearby','Tubule_Nearby']
    res={}
    for i,name in enumerate(tn):
        try: auc=roc_auc_score(tl2[:,i],tp[:,i])
        except: auc=float('nan')
        f1=f1_score(tl2[:,i],(tp[:,i]>0.5).astype(int),zero_division=0)
        res[name]={'AUC':auc,'F1':f1}
    return res, mean_attn

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("方案一 Final Part B: Full-Graph GNN + Attention Viz")
    print("="*60)

    cell_df=pd.read_csv(OUTPUT_DIR/'all_cells.csv')
    with open(OUTPUT_DIR/'data_split.json') as f: split=json.load(f)

    gb=CellSpatialGraph(k_nn=8,delaunay=True,max_dist=300.0)
    te=CompactTopoFeatures(radii=[50,100,200,300,400])
    topo_dim=len(te.extract_features(np.array([[0,0],[100,100],[200,200]])))

    node_df=build_node_labels(cell_df,radius=200)
    print(f"Node labels: {len(node_df)} total")

    dataset=FullGraphNodeDataset(cell_df,node_df,split['train_cases']+split['val_cases']+split['test_cases'],gb,te)
    train_fns=[f for f in dataset.fnames if any(c in split['train_cases'] for c in [f.split('-')[0]])]
    val_fns=[f for f in dataset.fnames if any(c in split['val_cases'] for c in [f.split('-')[0]])]
    test_fns=[f for f in dataset.fnames if any(c in split['test_cases'] for c in [f.split('-')[0]])]
    print(f"Graphs: train={len(train_fns)}, val={len(val_fns)}, test={len(test_fns)}")

    # Train all backbones
    print("\n--- Training Full-Graph GNNs ---")
    all_results={}; all_attn={}
    for bb in ['gcn','gat','gin','sage']:
        print(f"\n  {bb.upper()} (with full graph structure)...")
        model=TopoGNN_Full(backbone=bb,nd=15,td=topo_dim,hd=128,nt=3).to(DEVICE)
        res,attn=train_full_gnn(model,train_fns,val_fns,test_fns,dataset,epochs=50)
        all_results[bb]=res; all_attn[bb]=attn
        for tn,m in res.items():
            print(f"    {tn}: AUC={m['AUC']:.4f}, F1={m['F1']:.4f}")

    # Summary
    print("\n"+"="*60)
    print("FULL-GRAPH GNN RESULTS")
    print("="*60)
    tn=['Mitosis_Nearby','Apoptosis_Nearby','Tubule_Nearby']
    for t in tn:
        print(f"\n{t}:")
        for bb in ['gcn','gat','gin','sage']:
            m=all_results[bb][t]
            print(f"  {bb.upper():6s}: AUC={m['AUC']:.4f}, F1={m['F1']:.4f}")

    with open(OUTPUT_DIR/'final_fullgnn_results.json','w') as f:
        json.dump(all_results,f,indent=2)

    # ====== P1-4: Attention Visualization ======
    print("\n"+"="*60)
    print("P1-4: Topological Attention Visualization")
    print("="*60)

    fig,axes=plt.subplots(2,2,figsize=(18,12))
    for idx,bb in enumerate(['gcn','gat','gin','sage']):
        ax=axes[idx//2,idx%2]
        attn=all_attn[bb]
        top_k=5; top_idx=np.argsort(attn)[-top_k:][::-1]
        names=[TOPO_DIM_NAMES[i] for i in top_idx]
        vals=attn[top_idx]
        colors=plt.cm.RdYlGn(vals/np.max(vals))
        ax.barh(names,vals,color=colors,edgecolor='black')
        ax.set_title(f'{bb.upper()} - Top {top_k} Attended Topological Features')
        ax.set_xlabel('Attention Weight')
        for i,(n,v) in enumerate(zip(names,vals)):
            ax.text(v+0.005,i,f'{v:.3f}',va='center',fontsize=8)
    plt.suptitle('Topological Attention: Which TDA Features Drive Predictions?',fontsize=14,y=1.01)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig4_topological_attention.png',dpi=200,bbox_inches='tight'); plt.close()

    # Cross-backbone attention agreement
    fig,ax=plt.subplots(figsize=(14,5))
    x=np.arange(len(TOPO_DIM_NAMES)); width=0.2
    for i,bb in enumerate(['gcn','gat','gin','sage']):
        ax.bar(x+i*width,all_attn[bb],width,label=bb.upper(),alpha=0.7)
    ax.set_xticks(x+1.5*width); ax.set_xticklabels(TOPO_DIM_NAMES,rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('Attention Weight'); ax.set_title('Cross-Backbone Topological Attention Agreement')
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig4b_attention_agreement.png',dpi=200,bbox_inches='tight'); plt.close()
    print(f"  Saved attention figures to {FIG_DIR}")

    # ====== Backbone comparison bar chart ======
    fig,axes=plt.subplots(1,2,figsize=(16,5))
    x=np.arange(len(tn)); width=0.2
    colors=['#3498db','#e74c3c','#2ecc71','#f39c12']
    for i,bb in enumerate(['gcn','gat','gin','sage']):
        vals=[all_results[bb][t]['AUC'] for t in tn]
        axes[0].bar(x+i*width,vals,width,label=bb.upper(),color=colors[i],edgecolor='black')
        vals_f1=[all_results[bb][t]['F1'] for t in tn]
        axes[1].bar(x+i*width,vals_f1,width,label=bb.upper(),color=colors[i],edgecolor='black')
    for ax in axes:
        ax.set_xticks(x+1.5*width); ax.set_xticklabels(tn,fontsize=8); ax.legend(fontsize=8)
    axes[0].set_ylabel('AUC'); axes[0].set_title('Full-Graph GNN: AUC by Backbone')
    axes[1].set_ylabel('F1'); axes[1].set_title('Full-Graph GNN: F1 by Backbone')
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig5_fullgnn_backbone.png',dpi=200); plt.close()

    print(f"\nDone! All results in {OUTPUT_DIR} and {FIG_DIR}")

if __name__=='__main__':
    main()
