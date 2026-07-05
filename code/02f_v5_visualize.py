"""
方案一 Final v5: 论文级综合可视化 (Figure 1-6)
运行: python3 02f_v5_visualize.py
"""
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
FIG_DIR = OUTPUT_DIR / 'v5_final_figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11})

# ============================================================
# Load all results
# ============================================================
df_cv = pd.read_csv(OUTPUT_DIR/'v4_groupkfold_results.csv')
df_contrib = pd.read_csv(OUTPUT_DIR/'v4_graph_contribution.csv')
df_interp = pd.read_csv(OUTPUT_DIR/'v5_topo_biology.csv')
with open(OUTPUT_DIR/'v5_final_gnn_ci.json') as f: gnn_ci = json.load(f)
with open(OUTPUT_DIR/'v4_ablation_results.json') as f: ablation = json.load(f)
with open(OUTPUT_DIR/'v4_multiscale_results.json') as f: multiscale = json.load(f)

# ============================================================
# Figure 1: Study Framework (conceptual)
# ============================================================
print("Generating Figure 1: Study Framework...")
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis('off')

# Title
ax.text(6, 9.7, 'Two-Level Spatial Graph Analysis of Tumor Microenvironment',
        ha='center', fontsize=18, fontweight='bold')

# Level 1 box
ax.add_patch(plt.Rectangle((0.3, 5.3), 5.4, 4.0, fill=True, facecolor='#e8f4fd', edgecolor='#3498db', linewidth=2.5, alpha=0.5))
ax.text(3.0, 9.0, 'Level 1: Image-Level Feature Comparison', ha='center', fontsize=14, fontweight='bold', color='#2471a3')
features_text = (
    'Traditional Morphology (14-d)\n'
    '  → Cell density, area stats, aspect ratio, NN distance\n\n'
    'Topological Features (26-d)\n'
    '  → H0/H1 persistence, Betti integrals, multiscale\n\n'
    'Deep Features\n'
    '  → ResNet50 (ImageNet), FT-ResNet50, Patch CNN\n\n'
    'Task: Binary classification of rare event presence\n'
    '  → Mitosis / Apoptosis / Tubule'
)
ax.text(3.0, 7.2, features_text, ha='center', va='center', fontsize=9, fontfamily='monospace')

# Arrow
ax.annotate('', xy=(9.0, 7.3), xytext=(5.9, 7.3),
           arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))

# Level 2 box
ax.add_patch(plt.Rectangle((6.3, 5.3), 5.4, 4.0, fill=True, facecolor='#fdebd0', edgecolor='#e74c3c', linewidth=2.5, alpha=0.5))
ax.text(9.0, 9.0, 'Level 2: Node-Level Graph Analysis', ha='center', fontsize=14, fontweight='bold', color='#c0392b')
gnn_text = (
    'Cell Spatial Graph Construction\n'
    '  → Delaunay triangulation + k-NN (k=8)\n'
    '  → Nodes: 15-d cell features\n'
    '  → Edges: spatial proximity\n\n'
    'Graph Neural Network (GATv2)\n'
    '  → 2-layer message passing\n'
    '  → Multi-task: 3 binary heads\n\n'
    'Ablation Study\n'
    '  → Full Graph vs MLP vs Random Edges\n\n'
    'Task: Predict rare event proximity\n'
    '  → Mitosis / Apoptosis / Tubule nearby'
)
ax.text(9.0, 7.2, gnn_text, ha='center', va='center', fontsize=9, fontfamily='monospace')

# Bottom summary
ax.text(6, 4.5, 'Key Finding: Spatial graph structure contributes ΔAUC = +0.336 (Wilcoxon p=0.031)',
        ha='center', fontsize=13, fontweight='bold', color='#27ae60',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f5e3', edgecolor='#27ae60'))

plt.tight_layout()
plt.savefig(FIG_DIR/'fig1_study_framework.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig1_study_framework.png")

# ============================================================
# Figure 2: Image-Level Feature Comparison (GroupKFold + CI)
# ============================================================
print("Generating Figure 2: Feature Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
tasks = ['Mitosis', 'Apoptosis', 'Tubule']
for idx, tn in enumerate(tasks):
    row = df_cv[df_cv['task']==tn].iloc[0]
    methods = ['Traditional', 'Topological', 'Combined']
    vals = [row['Trad_AUC'], row['Topo_AUC'], row['Comb_AUC']]
    ci_lows = [row['Trad_CI_low'], row['Topo_CI_low'], row['Comb_CI_low']]
    ci_highs = [row['Trad_CI_high'], row['Topo_CI_high'], row['Comb_CI_high']]
    errors = [[v-l for v,l in zip(vals,ci_lows)], [h-v for v,h in zip(vals,ci_highs)]]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[idx].bar(methods, vals, color=colors, edgecolor='black', linewidth=1, yerr=errors, capsize=6)
    axes[idx].set_title(f'{tn} Detection', fontweight='bold')
    axes[idx].set_ylabel('AUC (GroupKFold)')
    axes[idx].set_ylim(0, 1.1)
    axes[idx].tick_params(axis='x', rotation=15)
    for i, (v, l, h) in enumerate(zip(vals, ci_lows, ci_highs)):
        axes[idx].text(i, v+0.04, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        axes[idx].text(i, v+0.01, f'[{l:.2f}, {h:.2f}]', ha='center', fontsize=7, color='gray')
    axes[idx].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.suptitle('Image-Level Classification: Multi-View Feature Comparison\n(Case-Level GroupKFold, 95% Bootstrap CI)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR/'fig2_feature_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig2_feature_comparison.png")

# ============================================================
# Figure 3: Graph Structure Ablation (Main Result)
# ============================================================
print("Generating Figure 3: Graph Ablation...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
tn = ['Apoptosis_Nearby', 'Tubule_Nearby']
mode_names = ['Full Graph', 'No Edges\n(MLP)', 'Random\nEdges']
backbones = ['gcn', 'gat']

for idx, task in enumerate(tn):
    x = np.arange(len(mode_names)); width = 0.35
    for bi, bb in enumerate(backbones):
        vals = [ablation[bb][m][task]['AUC'] for m in ['Full Graph', 'No Edges (MLP)', 'Random Edges']]
        vals = [v if not np.isnan(v) else 0 for v in vals]
        axes[idx].bar(x + bi*width, vals, width, label=bb.upper(),
                     color=['#3498db','#e74c3c'][bi], edgecolor='black', linewidth=1)
    axes[idx].set_xticks(x + width/2)
    axes[idx].set_xticklabels(mode_names, fontsize=9)
    axes[idx].set_title(f'{task}', fontweight='bold')
    axes[idx].set_ylabel('AUC')
    axes[idx].set_ylim(0, 1.15)
    axes[idx].legend(fontsize=9, loc='lower right')

    # Δ annotations
    for bi, bb in enumerate(backbones):
        full_val = ablation[bb]['Full Graph'][task]['AUC']
        mlp_val = ablation[bb]['No Edges (MLP)'][task]['AUC']
        if not np.isnan(full_val) and not np.isnan(mlp_val):
            delta = full_val - mlp_val
            y_pos = max(full_val, mlp_val) + 0.06
            axes[idx].annotate(f'Δ={delta:+.3f}',
                              xy=(0.5 + bi*0.35, y_pos-0.02),
                              ha='center', fontsize=11, fontweight='bold',
                              color='#27ae60',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', alpha=0.8))

plt.suptitle('Graph Structure Ablation: The Critical Role of Spatial Connectivity\n(Full Graph vs MLP: Mean ΔAUC=+0.336, Wilcoxon p=0.031)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR/'fig3_graph_ablation.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig3_graph_ablation.png")

# ============================================================
# Figure 4: Multi-Scale Sensitivity + Graph Contribution
# ============================================================
print("Generating Figure 4: Sensitivity + Contribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Multi-scale sensitivity
tn2 = ['Apoptosis_Nearby', 'Tubule_Nearby']
ks = [4, 8, 16, 32]
for task in tn2:
    aucs = [multiscale[f'k={k}'][task]['AUC'] for k in ks]
    aucs = [a if not np.isnan(a) else 0 for a in aucs]
    axes[0].plot(ks, aucs, 'o-', linewidth=2, markersize=8, label=task.replace('_Nearby',''))
axes[0].set_xlabel('k-NN Parameter'); axes[0].set_ylabel('AUC')
axes[0].set_title('Multi-Scale Graph Construction Sensitivity', fontweight='bold')
axes[0].set_xticks(ks); axes[0].legend(fontsize=9)
axes[0].axvline(x=8, color='red', linestyle='--', alpha=0.5, label='Optimal k=8')

# Right: Graph contribution
x = np.arange(len(df_contrib)); width = 0.35
bars1 = axes[1].bar(x - width/2, df_contrib['Delta_Full'], width, label='Full - MLP',
                   color='#2ecc71', edgecolor='black', linewidth=1)
bars2 = axes[1].bar(x + width/2, df_contrib['Delta_Random'], width, label='Random - MLP',
                   color='#f39c12', edgecolor='black', linewidth=1)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
labels = [f"{r['backbone'].upper()}\n{r['task'].replace('_Nearby','')}" for _, r in df_contrib.iterrows()]
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7)
axes[1].set_ylabel('ΔAUC')
axes[1].set_title('Graph Structure Contribution (Wilcoxon p=0.031)', fontweight='bold')
axes[1].legend(fontsize=9)
for bar in bars1:
    h = bar.get_height()
    axes[1].text(bar.get_x()+bar.get_width()/2, h+0.008, f'{h:.3f}', ha='center', fontsize=7, fontweight='bold')

plt.suptitle('Robustness Analysis: Multi-Scale Sensitivity & Structure Contribution',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR/'fig4_sensitivity_contribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig4_sensitivity_contribution.png")

# ============================================================
# Figure 5: Node-Level GNN Final Results (with 95% CI)
# ============================================================
print("Generating Figure 5: Node-Level GNN Final...")
fig, ax = plt.subplots(figsize=(12, 6))
tn3 = ['Mitosis_Nearby', 'Apoptosis_Nearby', 'Tubule_Nearby']
x = np.arange(len(tn3)); width = 0.35

# GNN results
gnn_aucs = [gnn_ci[t]['AUC_mean'] for t in tn3]
gnn_cis = [[gnn_ci[t]['AUC_mean']-gnn_ci[t]['CI_low'] for t in tn3],
           [gnn_ci[t]['CI_high']-gnn_ci[t]['AUC_mean'] for t in tn3]]

# MLP results (from ablation)
mlp_aucs = [ablation['gat']['No Edges (MLP)'][t]['AUC'] for t in tn3]
mlp_aucs = [a if not np.isnan(a) else 0 for a in mlp_aucs]

bars1 = ax.bar(x - width/2, gnn_aucs, width, label='GAT + Full Graph',
              color='#2ecc71', edgecolor='black', linewidth=1, yerr=gnn_cis, capsize=6)
bars2 = ax.bar(x + width/2, mlp_aucs, width, label='MLP (No Graph)',
              color='#e74c3c', edgecolor='black', linewidth=1)

# Annotate AUC values
for i, (bar, auc, ci_l, ci_h) in enumerate(zip(bars1, gnn_aucs,
    [gnn_ci[t]['CI_low'] for t in tn3], [gnn_ci[t]['CI_high'] for t in tn3])):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
            f'{auc:.3f}\n[{ci_l:.2f}-{ci_h:.2f}]', ha='center', fontsize=8, fontweight='bold', color='#27ae60')
    delta = auc - mlp_aucs[i]
    ax.annotate(f'Δ={delta:+.3f}', xy=(i, max(auc, mlp_aucs[i])+0.12),
               ha='center', fontsize=10, fontweight='bold', color='#c0392b')

ax.set_xticks(x)
ax.set_xticklabels([t.replace('_Nearby','\nNearby') for t in tn3], fontsize=10)
ax.set_ylabel('AUC'); ax.set_ylim(0, 1.2)
ax.set_title('Node-Level GNN: Final Results with 95% Bootstrap CI\n(3-run average, case-level split)',
            fontweight='bold')
ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig(FIG_DIR/'fig5_node_gnn_final.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig5_node_gnn_final.png")

# ============================================================
# Figure 6: Topological Feature Biology Interpretation
# ============================================================
print("Generating Figure 6: Biology Interpretation...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Mitosis
top_mit = df_interp.nsmallest(10, 'mitosis_p')
axes[0].barh(top_mit['dimension'], -np.log10(top_mit['mitosis_p']),
            color=['#e74c3c' if s else '#95a5a6' for s in top_mit['mitosis_sig']],
            edgecolor='black', linewidth=0.5)
axes[0].axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='p=0.05')
axes[0].set_xlabel('-log10(p-value)'); axes[0].set_title('Mitosis: Topological Discriminators', fontweight='bold')
axes[0].legend(fontsize=8)

# Tubule
top_tub = df_interp.nsmallest(10, 'tubule_p')
axes[1].barh(top_tub['dimension'], -np.log10(top_tub['tubule_p']),
            color=['#3498db' if s else '#95a5a6' for s in top_tub['tubule_sig']],
            edgecolor='black', linewidth=0.5)
axes[1].axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='p=0.05')
axes[1].set_xlabel('-log10(p-value)'); axes[1].set_title('Tubule: Topological Discriminators', fontweight='bold')
axes[1].legend(fontsize=8)

plt.suptitle('Biological Interpretation of Topological Features\n(Mann-Whitney U test: positive vs negative images)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR/'fig6_topo_biology.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig6_topo_biology.png")

# ============================================================
# Summary table output
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print("\nTable 1: Image-Level Classification (GroupKFold, 95% CI)")
print("-"*70)
print(f"{'Task':<12} {'Traditional':>18} {'Topological':>18} {'Combined':>18}")
print("-"*70)
for _, row in df_cv.iterrows():
    print(f"{row['task']:<12} {row['Trad_AUC']:.3f} [{row['Trad_CI_low']:.2f}-{row['Trad_CI_high']:.2f}]  "
          f"{row['Topo_AUC']:.3f} [{row['Topo_CI_low']:.2f}-{row['Topo_CI_high']:.2f}]  "
          f"{row['Comb_AUC']:.3f} [{row['Comb_CI_low']:.2f}-{row['Comb_CI_high']:.2f}]")

print("\nTable 2: Graph Structure Ablation (ΔAUC = Full - MLP)")
print("-"*60)
print(f"{'Backbone':<10} {'Task':<20} {'Full':>8} {'MLP':>8} {'Δ':>8}")
print("-"*60)
for _, row in df_contrib.iterrows():
    print(f"{row['backbone'].upper():<10} {row['task']:<20} {row['Full_AUC']:.3f}  {row['MLP_AUC']:.3f}  {row['Delta_Full']:+.3f}")
print(f"\n  Mean ΔAUC = {df_contrib['Delta_Full'].mean():.3f} ± {df_contrib['Delta_Full'].std():.3f}")

print("\nTable 3: Node-Level GNN Final (3-run avg, 95% Bootstrap CI)")
print("-"*70)
print(f"{'Task':<22} {'AUC':>10} {'95% CI':>20} {'F1':>8} {'Pos/Neg':>12}")
print("-"*70)
for t in tn3:
    r = gnn_ci[t]
    print(f"{t:<22} {r['AUC_mean']:.3f}±{r['AUC_std']:.3f}  [{r['CI_low']:.3f}-{r['CI_high']:.3f}]  "
          f"{r['F1_mean']:.3f}  {r['n_pos']}/{r['n_neg']}")

print(f"\nAll figures saved to: {FIG_DIR}")
print("Ready for publication!")

if __name__ == '__main__':
    pass
