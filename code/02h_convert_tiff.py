"""
P1-3: 将所有主图和补充图从PNG转为300 DPI TIFF（期刊要求）
"""
from PIL import Image
from pathlib import Path

FIG_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/Pathology/figures')
SUPP_DIR = Path('/Users/rolex8866/aionclaw/project/BreCAHAD/Pathology/supplementary')

# Create TIFF output directories
tiff_main = FIG_DIR / 'tiff_300dpi'
tiff_supp = SUPP_DIR / 'tiff_300dpi'
tiff_main.mkdir(exist_ok=True)
tiff_supp.mkdir(exist_ok=True)

converted = 0
for src_dir, tiff_dir, label in [(FIG_DIR, tiff_main, 'Main'), (SUPP_DIR, tiff_supp, 'Supp')]:
    for png_file in sorted(src_dir.glob('*.png')):
        img = Image.open(png_file)
        # Get DPI info if available
        dpi = img.info.get('dpi', (200, 200))
        print(f'[{label}] {png_file.name}: {img.size}, DPI={dpi}')
        
        # Convert to TIFF at 300 DPI
        tiff_path = tiff_dir / (png_file.stem + '.tiff')
        # Calculate new size for 300 DPI (preserve physical size)
        # Current: size in pixels at current DPI → physical inches → 300 DPI pixels
        if dpi[0] > 0:
            physical_w = img.width / dpi[0]
            physical_h = img.height / dpi[1]
            new_w = int(physical_w * 300)
            new_h = int(physical_h * 300)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            img_resized = img
        
        img_resized.save(tiff_path, format='TIFF', dpi=(300, 300), compression='lzw')
        print(f'  → {tiff_path.name}: {img_resized.size}, 300 DPI')
        converted += 1

print(f'\n✓ {converted} images converted to 300 DPI TIFF')
print(f'  Main: {tiff_main}')
print(f'  Supp: {tiff_supp}')
