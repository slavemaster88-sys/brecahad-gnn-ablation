"""
BreCAHAD 数据预处理与探索性分析
=================================
功能:
  1. 解析YOLO标注格式，构建结构化细胞对象表
  2. 统计分析：类别分布、空间分布、形态特征
  3. 生成可视化：类别热力图、细胞密度图、共现矩阵
  4. 导出处理后的数据供下游模型使用

作者: BreCAHAD Analysis Pipeline
"""

import os
import json
import warnings
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import KDTree, Delaunay
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_ROOT = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/BreCAHAD')
IMAGES_DIR = DATA_ROOT / 'images'
LABELS_DIR = DATA_ROOT / 'labels'
ANNOTATIONS_DIR = DATA_ROOT / 'Annotations'
OUTPUT_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/analysis_code/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: 'Mitosis',
    1: 'Apoptosis',
    2: 'Tumor_nuclei',
    3: 'Non_tumor_nuclei',
    4: 'Tubule',
    5: 'Non_tubule'
}

CLASS_COLORS = {
    0: '#FF0000',  # Mitosis - Red
    1: '#FF8C00',  # Apoptosis - Dark Orange
    2: '#1E90FF',  # Tumor_nuclei - Dodger Blue
    3: '#32CD32',  # Non_tumor_nuclei - Lime Green
    4: '#9370DB',  # Tubule - Medium Purple
    5: '#FFD700'   # Non_tubule - Gold
}

IMAGE_SIZE = (1360, 1024)  # W, H


# ============================================================
# 数据结构
# ============================================================
@dataclass
class CellObject:
    """单个细胞/结构标注对象"""
    class_id: int
    class_name: str
    x_center: float      # 归一化 (0-1)
    y_center: float
    width: float          # 归一化
    height: float
    # 绝对坐标
    x_center_abs: float = 0.0
    y_center_abs: float = 0.0
    width_abs: float = 0.0
    height_abs: float = 0.0
    area_abs: float = 0.0

@dataclass
class ImageData:
    """单张图像的所有标注信息"""
    filename: str
    case_id: str
    image_id: str
    cells: List[CellObject] = field(default_factory=list)
    image_path: Optional[Path] = None

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def class_counts(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for c in self.cells:
            counts[c.class_name] += 1
        return dict(counts)


# ============================================================
# 数据加载
# ============================================================
def parse_yolo_label(label_path: Path, img_w: int = 1360, img_h: int = 1024) -> List[CellObject]:
    """解析YOLO格式标注文件"""
    cells = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])

            cell = CellObject(
                class_id=cls_id,
                class_name=CLASS_NAMES.get(cls_id, f'Unknown_{cls_id}'),
                x_center=x_c, y_center=y_c,
                width=w, height=h,
                x_center_abs=x_c * img_w,
                y_center_abs=y_c * img_h,
                width_abs=w * img_w,
                height_abs=h * img_h,
                area_abs=(w * img_w) * (h * img_h)
            )
            cells.append(cell)
    return cells


def load_all_data() -> List[ImageData]:
    """加载全部数据集"""
    all_data = []
    for label_file in sorted(LABELS_DIR.glob('*.txt')):
        # 解析文件名: Case_X-YY.txt
        stem = label_file.stem  # e.g., Case_1-01
        parts = stem.split('-')
        case_id = parts[0]      # Case_1
        image_id = parts[1]     # 01

        cells = parse_yolo_label(label_file)
        img_path = IMAGES_DIR / f"{stem}.jpg"

        img_data = ImageData(
            filename=stem,
            case_id=case_id,
            image_id=image_id,
            cells=cells,
            image_path=img_path if img_path.exists() else None
        )
        all_data.append(img_data)

    return all_data


# ============================================================
# 统计分析
# ============================================================
def compute_statistics(all_data: List[ImageData]) -> Dict:
    """计算全面的统计信息"""
    stats = {
        'total_images': len(all_data),
        'total_cases': len(set(d.case_id for d in all_data)),
        'total_cells': sum(d.n_cells for d in all_data),
        'images_per_case': defaultdict(int),
        'class_counts': defaultdict(int),
        'cells_per_image': [],
        'class_area_stats': defaultdict(list),
        'spatial_stats': {}
    }

    for data in all_data:
        stats['images_per_case'][data.case_id] += 1
        stats['cells_per_image'].append(data.n_cells)

        for cell in data.cells:
            stats['class_counts'][cell.class_name] += 1
            stats['class_area_stats'][cell.class_name].append(cell.area_abs)

    stats['cells_per_image_mean'] = np.mean(stats['cells_per_image'])
    stats['cells_per_image_std'] = np.std(stats['cells_per_image'])

    # 类别面积统计
    for cls_name, areas in stats['class_area_stats'].items():
        areas_arr = np.array(areas)
        stats['class_area_stats'][cls_name] = {
            'mean': np.mean(areas_arr),
            'std': np.std(areas_arr),
            'median': np.median(areas_arr),
            'min': np.min(areas_arr),
            'max': np.max(areas_arr),
            'count': len(areas_arr)
        }

    return stats


def compute_spatial_statistics(all_data: List[ImageData]) -> pd.DataFrame:
    """计算空间分布统计：每张图的细胞密度、聚类程度等"""
    records = []
    for data in all_data:
        if data.n_cells < 3:
            continue

        # 提取所有细胞中心坐标
        coords = np.array([[c.x_center_abs, c.y_center_abs] for c in data.cells])

        # 最近邻距离
        if len(coords) >= 2:
            nbrs = NearestNeighbors(n_neighbors=min(5, len(coords))).fit(coords)
            distances, _ = nbrs.kneighbors(coords)
            mean_nn_dist = np.mean(distances[:, 1]) if distances.shape[1] > 1 else 0
        else:
            mean_nn_dist = 0

        # 细胞密度
        img_area = IMAGE_SIZE[0] * IMAGE_SIZE[1]
        cell_density = data.n_cells / img_area * 1e6  # cells per mm² (approx)

        # 类别丰富度
        class_richness = len(set(c.class_id for c in data.cells))

        # 类别Shannon多样性指数
        class_counts = defaultdict(int)
        for c in data.cells:
            class_counts[c.class_id] += 1
        total = sum(class_counts.values())
        shannon = -sum((c/total) * np.log(c/total) for c in class_counts.values() if c > 0)

        records.append({
            'filename': data.filename,
            'case_id': data.case_id,
            'n_cells': data.n_cells,
            'cell_density': cell_density,
            'mean_nn_dist': mean_nn_dist,
            'class_richness': class_richness,
            'shannon_diversity': shannon
        })

    return pd.DataFrame(records)


# ============================================================
# 可视化
# ============================================================
def plot_class_distribution(stats: Dict):
    """类别分布柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 总计数
    classes = list(stats['class_counts'].keys())
    counts = list(stats['class_counts'].values())
    colors = [CLASS_COLORS[list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(c)]]
              for c in classes]

    axes[0].barh(classes, counts, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_xlabel('Count (log scale)')
    axes[0].set_xscale('log')
    axes[0].set_title('Class Distribution (Log Scale)')
    for i, (c, v) in enumerate(zip(classes, counts)):
        axes[0].text(v + 0.02, i, str(v), va='center', fontsize=9)

    # 右: 每图细胞数分布
    cells_per_img = stats['cells_per_image']
    axes[1].hist(cells_per_img, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(stats['cells_per_image_mean'], color='red', linestyle='--',
                    label=f'Mean: {stats["cells_per_image_mean"]:.1f}')
    axes[1].set_xlabel('Cells per Image')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Cells per Image Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: class_distribution.png")


def plot_cell_size_comparison(stats: Dict):
    """各类别细胞面积对比箱线图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    data_for_box = []
    labels = []
    for cls_name, area_stats in stats['class_area_stats'].items():
        # area_stats is now a dict, need original data
        pass

    # Recompute from raw data
    all_data = load_all_data()
    class_areas = defaultdict(list)
    for data in all_data:
        for cell in data.cells:
            class_areas[cell.class_name].append(cell.area_abs)

    classes_ordered = ['Mitosis', 'Apoptosis', 'Tumor_nuclei',
                       'Non_tumor_nuclei', 'Tubule', 'Non_tubule']
    box_data = [class_areas[c] for c in classes_ordered if c in class_areas]
    box_labels = [c for c in classes_ordered if c in class_areas]
    box_colors = [CLASS_COLORS[list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(c)]]
                  for c in box_labels]

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                     showfliers=False)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Cell Area (pixels²)')
    ax.set_title('Cell Size Distribution by Class')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cell_size_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: cell_size_comparison.png")


def plot_spatial_heatmap(all_data: List[ImageData], n_samples: int = 6):
    """绘制多张图像的空间热力图"""
    import random
    random.seed(42)
    samples = random.sample([d for d in all_data if d.n_cells > 30], min(n_samples, len(all_data)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, data in enumerate(samples[:6]):
        ax = axes[idx]

        # 提取坐标
        coords = np.array([[c.x_center_abs, c.y_center_abs] for c in data.cells])
        classes = np.array([c.class_id for c in data.cells])

        # 密度估计
        if len(coords) > 5:
            try:
                kde = gaussian_kde(coords.T, bw_method=0.05)
                xi, yi = np.mgrid[0:IMAGE_SIZE[0]:2, 0:IMAGE_SIZE[1]:2]
                zi = kde(np.vstack([xi.ravel(), yi.ravel()]))
                zi = zi.reshape(xi.shape)
                ax.contourf(xi, yi, zi, levels=15, cmap='YlOrRd', alpha=0.7)
            except Exception:
                pass

        # 散点
        for cls_id in np.unique(classes):
            mask = classes == cls_id
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=CLASS_COLORS[cls_id], s=3, alpha=0.6,
                      label=CLASS_NAMES[cls_id])

        ax.set_xlim(0, IMAGE_SIZE[0])
        ax.set_ylim(IMAGE_SIZE[1], 0)  # 翻转Y轴匹配图像坐标
        ax.set_title(f'{data.filename} (n={data.n_cells})', fontsize=10)
        ax.set_aspect('equal')

    # 图例
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(6)]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8)
    plt.suptitle('Spatial Distribution of Cells (with Density Heatmap)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spatial_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: spatial_heatmap.png")


def plot_cooccurrence_matrix(all_data: List[ImageData]):
    """细胞类型共现矩阵"""
    n_classes = len(CLASS_NAMES)
    cooc_matrix = np.zeros((n_classes, n_classes))

    for data in all_data:
        classes_in_img = set(c.class_id for c in data.cells)
        for c1 in classes_in_img:
            for c2 in classes_in_img:
                cooc_matrix[c1, c2] += 1

    # 归一化
    row_sums = cooc_matrix.sum(axis=1, keepdims=True)
    cooc_norm = cooc_matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    class_labels = [CLASS_NAMES[i] for i in range(n_classes)]

    sns.heatmap(cooc_norm, annot=cooc_matrix.astype(int), fmt='.0f',
                xticklabels=class_labels, yticklabels=class_labels,
                cmap='YlOrRd', square=True, ax=ax,
                cbar_kws={'label': 'Normalized Co-occurrence'})
    ax.set_title('Cell Type Co-occurrence Matrix', fontsize=14)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Cell Type')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cooccurrence_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: cooccurrence_matrix.png")


def plot_case_level_analysis(spatial_df: pd.DataFrame):
    """病例级别分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 每病例图像数
    case_counts = spatial_df.groupby('case_id')['filename'].count()
    axes[0, 0].bar(case_counts.index, case_counts.values, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Images per Case')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylabel('Count')

    # 细胞密度 by case
    case_density = spatial_df.groupby('case_id')['cell_density'].agg(['mean', 'std'])
    x = range(len(case_density))
    axes[0, 1].bar(x, case_density['mean'], yerr=case_density['std'],
                   color='coral', edgecolor='black', capsize=5)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(case_density.index, rotation=45)
    axes[0, 1].set_title('Cell Density by Case')
    axes[0, 1].set_ylabel('Cells / mm² (approx)')

    # Shannon多样性
    case_diversity = spatial_df.groupby('case_id')['shannon_diversity'].agg(['mean', 'std'])
    axes[1, 0].bar(x, case_diversity['mean'], yerr=case_diversity['std'],
                   color='mediumseagreen', edgecolor='black', capsize=5)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(case_diversity.index, rotation=45)
    axes[1, 0].set_title('Shannon Diversity by Case')
    axes[1, 0].set_ylabel('Shannon Index')

    # 细胞数 vs 多样性散点
    axes[1, 1].scatter(spatial_df['n_cells'], spatial_df['shannon_diversity'],
                       c=spatial_df['cell_density'], cmap='viridis', alpha=0.7, edgecolors='black')
    axes[1, 1].set_xlabel('Number of Cells')
    axes[1, 1].set_ylabel('Shannon Diversity')
    axes[1, 1].set_title('Cells vs Diversity (colored by density)')
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Cell Density')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'case_level_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: case_level_analysis.png")


# ============================================================
# 数据导出
# ============================================================
def export_processed_data(all_data: List[ImageData], stats: Dict):
    """导出处理后的数据供下游使用"""
    # 1. 结构化细胞表
    cell_records = []
    for data in all_data:
        for cell in data.cells:
            cell_records.append({
                'filename': data.filename,
                'case_id': data.case_id,
                'image_id': data.image_id,
                'class_id': cell.class_id,
                'class_name': cell.class_name,
                'x_center': cell.x_center,
                'y_center': cell.y_center,
                'width': cell.width,
                'height': cell.height,
                'x_center_abs': cell.x_center_abs,
                'y_center_abs': cell.y_center_abs,
                'width_abs': cell.width_abs,
                'height_abs': cell.height_abs,
                'area_abs': cell.area_abs
            })

    cell_df = pd.DataFrame(cell_records)
    cell_df.to_csv(OUTPUT_DIR / 'all_cells.csv', index=False)
    print(f"[✓] Exported: all_cells.csv ({len(cell_df)} cells)")

    # 2. 图像级统计
    img_stats = []
    for data in all_data:
        counts = data.class_counts()
        img_stats.append({
            'filename': data.filename,
            'case_id': data.case_id,
            'n_cells': data.n_cells,
            **{f'n_{cls}': counts.get(cls, 0) for cls in CLASS_NAMES.values()}
        })

    img_df = pd.DataFrame(img_stats)
    img_df.to_csv(OUTPUT_DIR / 'image_statistics.csv', index=False)
    print(f"[✓] Exported: image_statistics.csv ({len(img_df)} images)")

    # 3. 统计摘要JSON
    stats_export = {
        'total_images': stats['total_images'],
        'total_cases': stats['total_cases'],
        'total_cells': stats['total_cells'],
        'cells_per_image_mean': float(stats['cells_per_image_mean']),
        'cells_per_image_std': float(stats['cells_per_image_std']),
        'class_counts': dict(stats['class_counts']),
        'class_area_stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                  for kk, vv in v.items()}
                             for k, v in stats['class_area_stats'].items()}
    }
    with open(OUTPUT_DIR / 'dataset_summary.json', 'w') as f:
        json.dump(stats_export, f, indent=2)
    print(f"[✓] Exported: dataset_summary.json")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("BreCAHAD Data Preprocessing & EDA Pipeline")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] Loading data...")
    all_data = load_all_data()
    print(f"  Loaded {len(all_data)} images with {sum(d.n_cells for d in all_data)} total cells")

    # 2. 统计计算
    print("\n[2/6] Computing statistics...")
    stats = compute_statistics(all_data)
    spatial_df = compute_spatial_statistics(all_data)

    print(f"  Cases: {stats['total_cases']}")
    print(f"  Avg cells/image: {stats['cells_per_image_mean']:.1f} ± {stats['cells_per_image_std']:.1f}")
    print(f"  Class distribution:")
    for cls, count in sorted(stats['class_counts'].items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count}")

    # 3. 可视化
    print("\n[3/6] Generating visualizations...")
    plot_class_distribution(stats)
    plot_cell_size_comparison(stats)
    plot_spatial_heatmap(all_data)
    plot_cooccurrence_matrix(all_data)
    plot_case_level_analysis(spatial_df)

    # 4. 导出数据
    print("\n[4/6] Exporting processed data...")
    export_processed_data(all_data, stats)

    # 5. 数据划分建议
    print("\n[5/6] Data split recommendation:")
    cases = sorted(set(d.case_id for d in all_data))
    n_cases = len(cases)
    train_cases = cases[:int(0.7 * n_cases)]
    val_cases = cases[int(0.7 * n_cases):int(0.85 * n_cases)]
    test_cases = cases[int(0.85 * n_cases):]

    train_imgs = sum(1 for d in all_data if d.case_id in train_cases)
    val_imgs = sum(1 for d in all_data if d.case_id in val_cases)
    test_imgs = sum(1 for d in all_data if d.case_id in test_cases)

    print(f"  Train: {train_cases} ({train_imgs} images)")
    print(f"  Val:   {val_cases} ({val_imgs} images)")
    print(f"  Test:  {test_cases} ({test_imgs} images)")

    # 保存划分
    split_info = {
        'train_cases': train_cases,
        'val_cases': val_cases,
        'test_cases': test_cases,
        'train_images': train_imgs,
        'val_images': val_imgs,
        'test_images': test_imgs
    }
    with open(OUTPUT_DIR / 'data_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\n[6/6] Done! All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
