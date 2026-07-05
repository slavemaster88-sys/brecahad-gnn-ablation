"""
P1-6: 在Fig1框架图中嵌入真实Delaunay三角剖分示例
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
FIG_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/Pathology/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
cell_df = pd.read_csv(OUTPUT_DIR/'all_cells.csv')

# Pick a representative mitosis image
mitosis_files = cell_df[cell_df['class_id']==0]['filename'].unique()
print(f"Mitosis files: {mitosis_files[:5]}")
fname = mitosis_files[0]  # Case_1-01
cells = cell_df[cell_df['filename']==fname]

print(f"Using {fname}: {len(cells)} cells")

# Extract coordinates and class
coords = np.array([[c['x_center_abs'], c['y_center_abs']] for _, c in cells.iterrows()])
class_ids = cells['class_id'].values
class_names = {0: 'Mitosis', 1: 'Apoptosis', 2: 'Tumor', 3: 'Connective', 4: 'Tubule', 5: 'Inflammatory'}
class_colors = {0: '#e74c3c', 1: '#e67e22', 2: '#2ecc71', 3: '#3498db', 4: '#9b59b6', 5: '#f39c12'}

# Build Delaunay
tri = Delaunay(coords)
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i+1, 3):
            u, v = simplex[i], simplex[j]
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
edges = list(edges)
print(f"Delaunay edges: {len(edges)}")

# Create figure
fig = plt.figure(figsize=(18, 12))

# ============================================================
# Main framework layout
# ============================================================
gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25,
                       height_ratios=[1.2, 1, 0.8])

# ---- Row 0: Title bar ----
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
ax_title.axis('off')
ax_title.text(0.5, 0.5, 
              'Two-Level Ablation Framework for Spatial Graph Structure Analysis\n'
              'in Breast Histopathology Tumor Microenvironment Characterization',
              ha='center', va='center', fontsize=16, fontweight='bold',
              transform=ax_title.transAxes)

# ---- Row 1: Three columns (Input → Methods → Output) ----
# Column 1: Input
ax_input = fig.add_subplot(gs[1, 0])
ax_input.set_xlim(0, 1); ax_input.set_ylim(0, 1); ax_input.axis('off')
# Draw a box
rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.1",
                       facecolor='#e8f4f8', edgecolor='#2980b9', linewidth=2)
ax_input.add_patch(rect)
ax_input.text(0.5, 0.92, 'INPUT', ha='center', va='center', fontsize=13, fontweight='bold', color='#2980b9')
ax_input.text(0.5, 0.78, 'BreCAHAD Dataset', ha='center', va='center', fontsize=10, fontweight='bold')
ax_input.text(0.5, 0.68, '162 H&E Images', ha='center', va='center', fontsize=9, color='#555')
ax_input.text(0.5, 0.58, '23,496 Annotated Cells', ha='center', va='center', fontsize=9, color='#555')
ax_input.text(0.5, 0.48, '6 Cell Types', ha='center', va='center', fontsize=9, color='#555')
ax_input.text(0.5, 0.38, '17 Cases', ha='center', va='center', fontsize=9, color='#555')
ax_input.text(0.5, 0.22, 'Cell Detection + Classification\n(Oracle Ground Truth)', ha='center', va='center', fontsize=8, color='#888', style='italic')

# Column 2: Methods (center, wider)
ax_methods = fig.add_subplot(gs[1, 1])
ax_methods.set_xlim(0, 1); ax_methods.set_ylim(0, 1); ax_methods.axis('off')
rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.1",
                       facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
ax_methods.add_patch(rect)
ax_methods.text(0.5, 0.92, 'METHODS', ha='center', va='center', fontsize=13, fontweight='bold', color='#f39c12')

# Two sub-panels
ax_methods.text(0.25, 0.78, 'Level 1: Image-Level', ha='center', va='center', fontsize=10, fontweight='bold', color='#e67e22')
ax_methods.text(0.25, 0.68, 'Feature Extraction', ha='center', va='center', fontsize=8, color='#555')
for i, ft in enumerate(['Morphological (14-d)', 'Topological (26-d)', 'ResNet50 (2048-d)',
                          'Fine-tuned ResNet50', 'Patch ResNet18', 'Combined']):
    ax_methods.text(0.25, 0.60 - i*0.08, f'• {ft}', ha='center', va='center', fontsize=7, color='#333')

ax_methods.text(0.75, 0.78, 'Level 2: Node-Level', ha='center', va='center', fontsize=10, fontweight='bold', color='#e67e22')
ax_methods.text(0.75, 0.68, 'Spatial Graph + GNN', ha='center', va='center', fontsize=8, color='#555')
for i, item in enumerate(['Delaunay Triangulation', 'k-NN Refinement (k=8)',
                            'GATv2 Backbone', '3 Ablation Modes',
                            'Full Graph / MLP / Random']):
    ax_methods.text(0.75, 0.60 - i*0.08, f'• {item}', ha='center', va='center', fontsize=7, color='#333')

ax_methods.text(0.5, 0.20, 'Evaluation: GroupKFold (5-fold, case-level)\nBootstrap 95% CI + Wilcoxon + Bonferroni + Cohen\'s d',
                ha='center', va='center', fontsize=8, color='#888', style='italic')

# Column 3: Output
ax_output = fig.add_subplot(gs[1, 2])
ax_output.set_xlim(0, 1); ax_output.set_ylim(0, 1); ax_output.axis('off')
rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.1",
                       facecolor='#eafaf1', edgecolor='#27ae60', linewidth=2)
ax_output.add_patch(rect)
ax_output.text(0.5, 0.92, 'KEY FINDINGS', ha='center', va='center', fontsize=13, fontweight='bold', color='#27ae60')

findings = [
    ('Graph Structure', 'ΔAUC=+0.336, p=0.031'),
    ('Cohen\'s d', '12.0 (large effect)'),
    ('Mitosis Proximity', 'AUC=0.984 [0.979-0.989]'),
    ('Apoptosis Proximity', 'AUC=0.944 [0.932-0.955]'),
    ('Tubule Proximity', 'AUC=0.923 [0.914-0.930]'),
    ('Label Leakage', '92.1% non-self nodes'),
    ('185K params', '<5ms inference'),
]
for i, (label, value) in enumerate(findings):
    y = 0.78 - i*0.09
    ax_output.text(0.5, y, f'{label}: ', ha='right', va='center', fontsize=7, color='#555', fontweight='bold')
    ax_output.text(0.52, y, value, ha='left', va='center', fontsize=7, color='#27ae60', fontweight='bold')

# ---- Row 2: Delaunay example + verification badges ----
# Left: Real Delaunay triangulation
ax_delaunay = fig.add_subplot(gs[2, 0])
ax_delaunay.set_title('Delaunay Triangulation (Case_1-01)', fontsize=10, fontweight='bold')

# Draw edges
for u, v in edges:
    ax_delaunay.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
                     color='#bdc3c7', linewidth=0.3, alpha=0.4, zorder=1)

# Draw nodes
for cid in range(6):
    mask = class_ids == cid
    if mask.sum() > 0:
        ax_delaunay.scatter(coords[mask, 0], coords[mask, 1],
                           c=class_colors[cid], s=15, alpha=0.8,
                           edgecolors='white', linewidth=0.3,
                           label=class_names[cid], zorder=2)

# Highlight mitosis cells
mit_mask = class_ids == 0
if mit_mask.sum() > 0:
    ax_delaunay.scatter(coords[mit_mask, 0], coords[mit_mask, 1],
                       c='none', s=80, edgecolors='red', linewidth=1.5,
                       zorder=3, marker='o')

ax_delaunay.set_xlim(0, 1360); ax_delaunay.set_ylim(1024, 0)
ax_delaunay.set_aspect('equal')
ax_delaunay.legend(loc='upper right', fontsize=5, ncol=3, framealpha=0.7)
ax_delaunay.set_xlabel('X (pixels)', fontsize=8)
ax_delaunay.set_ylabel('Y (pixels)', fontsize=8)

# Center: Ablation concept
ax_ablation = fig.add_subplot(gs[2, 1])
ax_ablation.set_xlim(0, 1); ax_ablation.set_ylim(0, 1); ax_ablation.axis('off')
rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.1",
                       facecolor='#f5eef8', edgecolor='#8e44ad', linewidth=2)
ax_ablation.add_patch(rect)
ax_ablation.text(0.5, 0.90, 'ABLATION DESIGN', ha='center', va='center', fontsize=11, fontweight='bold', color='#8e44ad')

ablations = [
    ('Full Graph (Delaunay + k-NN)', 'GATv2 message passing over spatial edges', '#27ae60'),
    ('No Edges (MLP Baseline)', 'Same architecture, zero adjacency matrix', '#e74c3c'),
    ('Random Edges', 'Same density, shuffled connections', '#f39c12'),
]
for i, (name, desc, color) in enumerate(ablations):
    y = 0.72 - i*0.20
    ax_ablation.text(0.5, y+0.04, name, ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    ax_ablation.text(0.5, y-0.04, desc, ha='center', va='center', fontsize=7, color='#888', style='italic')

# Right: Verification badges
ax_verify = fig.add_subplot(gs[2, 2])
ax_verify.set_xlim(0, 1); ax_verify.set_ylim(0, 1); ax_verify.axis('off')
rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.1",
                       facecolor='#eaf2f8', edgecolor='#2980b9', linewidth=2)
ax_verify.add_patch(rect)
ax_verify.text(0.5, 0.90, 'VERIFICATION', ha='center', va='center', fontsize=11, fontweight='bold', color='#2980b9')

checks = [
    ('✓', 'Label Leakage Analysis', '#27ae60'),
    ('✓', 'Radius Sensitivity (100-500px)', '#27ae60'),
    ('✓', 'Bootstrap 95% CI', '#27ae60'),
    ('✓', 'Bonferroni Correction', '#27ae60'),
    ('✓', 'Cohen\'s d Effect Size', '#27ae60'),
    ('✓', 'GroupKFold (Case-Level)', '#27ae60'),
]
for i, (icon, text, color) in enumerate(checks):
    y = 0.72 - i*0.10
    ax_verify.text(0.25, y, icon, ha='center', va='center', fontsize=11, color=color, fontweight='bold')
    ax_verify.text(0.30, y, text, ha='left', va='center', fontsize=7.5, color='#333')

plt.suptitle('Figure 1: Study Overview and Framework', fontsize=16, fontweight='bold', y=1.01)
plt.savefig(FIG_DIR/'fig1_framework.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ fig1_framework.png saved to {FIG_DIR}")
