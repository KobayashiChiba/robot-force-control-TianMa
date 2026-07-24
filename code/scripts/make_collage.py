"""4张图拼成2×2布局"""
from PIL import Image
import os

BASE = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'
files = [
    'fig1_points_and_cylinders.png',
    'fig2_measured_vs_curve.png',
    'fig3_xz_projection.png',
    'fig4_xy_projection.png',
]

images = [Image.open(os.path.join(BASE, f)) for f in files]

# 统一尺寸（取最小宽高）
min_w = min(im.width for im in images)
min_h = min(im.height for im in images)
images = [im.resize((min_w, min_h), Image.LANCZOS) for im in images]

# 2×2 拼接
canvas_w = min_w * 2
canvas_h = min_h * 2
canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')

# 加上标题标签
from PIL import ImageDraw
draw = ImageDraw.Draw(canvas)

labels = [
    '(a) Points + Cylinders 3D',
    '(b) Points vs Intersection',
    '(c) Y-Projection (XZ)',
    '(d) Z-Projection (XY)',
]

positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
for i, (col, row) in enumerate(positions):
    x = col * min_w
    y = row * min_h
    canvas.paste(images[i], (x, y))
    # 白色半透明背景标签
    draw.rectangle([x+8, y+8, x+8+260, y+8+24], fill='white', outline='gray')
    draw.text((x+12, y+10), labels[i], fill='black')

out = os.path.join(BASE, 'fig_collage_2x2.png')
canvas.save(out, dpi=(150, 150))
print(f'Saved: {out}')
print(f'Size: {canvas.width}x{canvas.height}')
