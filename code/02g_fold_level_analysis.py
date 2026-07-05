"""
P1-5: 基于已有CI估计fold-level变异，并在manuscript中补充讨论
使用bootstrap CI宽度作为fold间变异的代理指标
"""
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output/v4_groupkfold_results.csv')
print("GroupKFold Results (Traditional features, 5-fold case-level):")
print("="*70)
for _, row in df.iterrows():
    task = row['task']
    trad_auc = row['Trad_AUC']
    trad_ci_w = row['Trad_CI_high'] - row['Trad_CI_low']
    topo_auc = row['Topo_AUC']
    topo_ci_w = row['Topo_CI_high'] - row['Topo_CI_low']
    comb_auc = row['Comb_AUC']
    comb_ci_w = row['Comb_CI_high'] - row['Comb_CI_low']
    
    print(f"\n{task}:")
    print(f"  Traditional: AUC={trad_auc:.3f}, 95% CI=[{row['Trad_CI_low']:.3f}, {row['Trad_CI_high']:.3f}], width={trad_ci_w:.3f}")
    print(f"  Topological: AUC={topo_auc:.3f}, 95% CI=[{row['Topo_CI_low']:.3f}, {row['Topo_CI_high']:.3f}], width={topo_ci_w:.3f}")
    print(f"  Combined:    AUC={comb_auc:.3f}, 95% CI=[{row['Comb_CI_low']:.3f}, {row['Comb_CI_high']:.3f}], width={comb_ci_w:.3f}")

print("\n\nInterpretation:")
print("- Bootstrap CI width reflects both fold-to-fold variation and within-fold sampling uncertainty")
print("- Wider CIs for Apoptosis (Trad: 0.189) suggest greater fold-to-fold variability")
print("- Narrower CIs for Mitosis (Trad: 0.121) suggest more consistent performance across folds")
print("- This is consistent with 17 cases split into 5 folds (3-4 test cases per fold)")
