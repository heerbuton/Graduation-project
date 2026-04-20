from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

BASE = Path(r'F:\AIcharacter\End\thesis_materials_2026-04-18\paper_ui_figures_CD_2026-04-18')
RAW = BASE / 'raw_screenshots' / 'paper_states'
FINAL = BASE / 'final_figures'
FINAL.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\simsun.ttc',
]

def get_font(size=28):
    for fp in FONT_CANDIDATES:
        p = Path(fp)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()

FONT_TITLE = get_font(42)
FONT_SUB = get_font(28)
FONT_CAP = get_font(24)
FONT_SMALL = get_font(20)


def load(name):
    return Image.open(RAW / name).convert('RGB')


def draw_label(draw, xy, text, fill=(20, 30, 52), text_fill=(240, 248, 255), pad=8):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=FONT_SMALL)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.rounded_rectangle((x - pad, y - pad, x + w + pad, y + h + pad), radius=8, fill=fill, outline=(120, 220, 255), width=2)
    draw.text((x, y), text, font=FONT_SMALL, fill=text_fill)


def arrow(draw, p1, p2, color=(255, 196, 64), width=5):
    draw.line([p1, p2], fill=color, width=width)
    # arrow head
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return
    import math
    ang = math.atan2(dy, dx)
    L = 16
    a1 = ang + math.pi * 0.82
    a2 = ang - math.pi * 0.82
    p3 = (x2 + int(L * math.cos(a1)), y2 + int(L * math.sin(a1)))
    p4 = (x2 + int(L * math.cos(a2)), y2 + int(L * math.sin(a2)))
    draw.polygon([p2, p3, p4], fill=color)


# ---------------- Figure 4-4 ----------------
img = load('S01_upload_preview_start.png').copy()
d = ImageDraw.Draw(img)

# highlight boxes
box_upload = (500, 230, 1415, 850)
box_preview = (860, 395, 1060, 720)
box_trigger = (850, 765, 1075, 845)

for box, color in [
    (box_upload, (88, 190, 255)),
    (box_preview, (255, 215, 90)),
    (box_trigger, (80, 240, 180)),
]:
    d.rounded_rectangle(box, radius=12, outline=color, width=5)

# title
banner_h = 86
canvas_44 = Image.new('RGB', (img.width, img.height + banner_h), (8, 18, 34))
canvas_44.paste(img, (0, banner_h))
dc = ImageDraw.Draw(canvas_44)
dc.text((36, 22), '图4-4 系统原始输入与任务触发界面示意图', font=FONT_TITLE, fill=(235, 246, 255))

# labels on image layer
di = ImageDraw.Draw(canvas_44)
draw_label(di, (530, 280 + banner_h), '原始谱页输入')
draw_label(di, (1080, 415 + banner_h), '任务触发入口')
draw_label(di, (1090, 770 + banner_h), '开始自动打谱')
arrow(di, (700, 315 + banner_h), (860, 430 + banner_h), color=(88, 190, 255), width=4)
arrow(di, (1210, 450 + banner_h), (1060, 470 + banner_h), color=(255, 215, 90), width=4)
arrow(di, (1240, 805 + banner_h), (1075, 805 + banner_h), color=(80, 240, 180), width=4)

fig44 = FINAL / 'fig4-4_系统原始输入与任务触发界面示意图.png'
canvas_44.save(fig44)


# ---------------- Figure 4-8 ----------------
left = load('S04_edit_panel_open.png').crop((480, 180, 1890, 980))
right = load('S05_apply_feedback.png').crop((480, 180, 1890, 980))
left = left.resize((1120, 640))
right = right.resize((1120, 640))

W, H = 2400, 980
canvas = Image.new('RGB', (W, H), (10, 20, 36))
d = ImageDraw.Draw(canvas)
d.text((40, 20), '图4-8 人工修正与局部迭代优化回环界面示意图', font=FONT_TITLE, fill=(236, 246, 255))
canvas.paste(left, (40, 130))
canvas.paste(right, (1240, 130))

# panel frames
for x in [40, 1240]:
    d.rounded_rectangle((x, 130, x+1120, 770), radius=14, outline=(100, 205, 255), width=4)

# captions
draw_label(d, (80, 790), '减字组字段编辑')
draw_label(d, (1280, 790), '局部迭代优化结果回显')

# arrow between panels
arrow(d, (1168, 450), (1232, 450), color=(255, 205, 88), width=6)
draw_label(d, (1040, 390), '提交修正结果', fill=(52, 40, 20), text_fill=(255, 236, 180))

fig48 = FINAL / 'fig4-8_人工修正与局部迭代优化回环界面示意图.png'
canvas.save(fig48)


# ---------------- Figure 5-1 ----------------
# 2x2 full-chain black-box verification
p1 = load('S01_upload_preview_start.png').resize((1080, 580))
p2 = load('S02_feature_extraction.png').resize((1080, 580))
p3 = load('S07_llm_inference.png').resize((1080, 580))
p4 = load('S08_final_render.png').resize((1080, 580))

W, H = 2280, 1360
canvas = Image.new('RGB', (W, H), (8, 18, 34))
d = ImageDraw.Draw(canvas)
d.text((40, 20), '图5-1 全链路自动打谱业务黑盒功能验证界面示意图', font=FONT_TITLE, fill=(236, 246, 255))

coords = [(40, 120), (1160, 120), (40, 740), (1160, 740)]
for im, (x, y), cap in [
    (p1, coords[0], '① 输入与触发'),
    (p2, coords[1], '② 特征提取与校对'),
    (p3, coords[2], '③ 乐理推理结果'),
    (p4, coords[3], '④ 最终打谱渲染'),
]:
    canvas.paste(im, (x, y))
    d.rounded_rectangle((x, y, x+1080, y+580), radius=12, outline=(90, 200, 255), width=4)
    draw_label(d, (x+18, y+16), cap)

# directional arrows
arrow(d, (1080, 410), (1160, 410), color=(255, 205, 88), width=6)
arrow(d, (1700, 700), (1700, 740), color=(255, 205, 88), width=6)
arrow(d, (1160, 1030), (1080, 1030), color=(255, 205, 88), width=6)

fig51 = FINAL / 'fig5-1_全链路自动打谱业务黑盒功能验证界面示意图.png'
canvas.save(fig51)


# ---------------- Figure 5-2 ----------------
# three-panel: hover -> edit -> topology
a = load('S03_hover_explain.png').crop((500, 180, 1880, 980)).resize((760, 500))
b = load('S04_edit_panel_open.png').crop((500, 180, 1880, 980)).resize((760, 500))
c = load('S06_topology_sequence.png').crop((80, 180, 1830, 980)).resize((760, 500))

W, H = 2400, 860
canvas = Image.new('RGB', (W, H), (8, 16, 30))
d = ImageDraw.Draw(canvas)
d.text((40, 20), '图5-2 减字组中间态呈现及交互修正链路示意图', font=FONT_TITLE, fill=(236, 246, 255))

positions = [40, 820, 1600]
for idx, (im, x, cap) in enumerate([
    (a, positions[0], '悬停字段提示框'),
    (b, positions[1], '编辑面板打开'),
    (c, positions[2], '拓扑序列卡片'),
]):
    canvas.paste(im, (x, 140))
    # main card border
    d.rounded_rectangle((x, 140, x+760, 640), radius=12, outline=(120, 210, 255), width=4)
    draw_label(d, (x+20, 660), f'{idx+1}. {cap}')

# core-area highlights (yellow/blue thin frames)
# panel A tooltip zone
d.rounded_rectangle((positions[0] + 330, 290, positions[0] + 635, 430), radius=8, outline=(255, 220, 80), width=3)
# panel B edit panel zone
d.rounded_rectangle((positions[1] + 520, 185, positions[1] + 748, 635), radius=8, outline=(80, 210, 255), width=3)
# panel C topology cards zone
d.rounded_rectangle((positions[2] + 20, 190, positions[2] + 740, 620), radius=8, outline=(255, 220, 80), width=3)

# flow arrows
arrow(d, (780, 390), (820, 390), color=(255, 205, 88), width=5)
arrow(d, (1560, 390), (1600, 390), color=(255, 205, 88), width=5)

fig52 = FINAL / 'fig5-2_减字组中间态呈现及交互修正链路示意图.png'
canvas.save(fig52)


# ---------------- Figure 5-3 ----------------
left = load('S04_edit_panel_open.png').crop((500, 180, 1880, 980)).resize((980, 560))
right = load('S08_final_render.png').crop((420, 170, 1880, 980)).resize((980, 560))

W, H = 2200, 980
canvas = Image.new('RGB', (W, H), (8, 18, 34))
d = ImageDraw.Draw(canvas)
d.text((40, 20), '图5-3 人机协同修正方案下的局部迭代优化回环示意图', font=FONT_TITLE, fill=(236, 246, 255))

canvas.paste(left, (60, 130))
canvas.paste(right, (1160, 130))
d.rounded_rectangle((60, 130, 1040, 690), radius=12, outline=(100, 205, 255), width=4)
d.rounded_rectangle((1160, 130, 2140, 690), radius=12, outline=(100, 205, 255), width=4)

draw_label(d, (92, 710), 'Before：减字组字段修正')
draw_label(d, (1192, 710), 'After：后续结果链路刷新')

arrow(d, (1060, 420), (1140, 420), color=(255, 205, 88), width=6)
draw_label(d, (880, 360), '仅刷新后续语义重构与渲染链路', fill=(58, 45, 20), text_fill=(255, 238, 180))

# bottom note box
note_box = (60, 780, 2140, 930)
d.rounded_rectangle(note_box, radius=14, fill=(15, 28, 48), outline=(120, 210, 255), width=3)
note = '说明：修正发生在减字组中间态，不需要重新上传图像，也不需要重新执行整页视觉识别；仅触发后续链路迭代。'
d.text((90, 835), note, font=FONT_CAP, fill=(220, 238, 255))

fig53 = FINAL / 'fig5-3_人机协同修正方案下的局部迭代优化回环示意图.png'
canvas.save(fig53)


# ---------------- source manifest ----------------
manifest_md = BASE / 'final_figures' / 'figure_source_manifest.md'
manifest_text = f'''# 论文界面图来源清单（C/D 档）

## 运行信息
- 主要截图运行日志：`{(BASE / 'logs' / 'paper_states_capture_log.json')}`
- Demo 时序日志：`{(BASE / 'logs' / 'demo_timeline_probe.json')}`
- 前端地址：`http://localhost:3001`
- 后端地址：`http://127.0.0.1:5000`
- 典型样例：`F:/AIcharacter/End/test/testpicture-1.jpg`

## 原始截图保留目录
- `{RAW}`

## 成图与原图映射
- 图4-4：`fig4-4_系统原始输入与任务触发界面示意图.png`
  - 原图：`S01_upload_preview_start.png`
- 图4-8：`fig4-8_人工修正与局部迭代优化回环界面示意图.png`
  - 原图：`S04_edit_panel_open.png` + `S05_apply_feedback.png`
- 图5-1：`fig5-1_全链路自动打谱业务黑盒功能验证界面示意图.png`
  - 原图：`S01_upload_preview_start.png` + `S02_feature_extraction.png` + `S07_llm_inference.png` + `S08_final_render.png`
- 图5-2：`fig5-2_减字组中间态呈现及交互修正链路示意图.png`
  - 原图：`S03_hover_explain.png` + `S04_edit_panel_open.png` + `S06_topology_sequence.png`
- 图5-3：`fig5-3_人机协同修正方案下的局部迭代优化回环示意图.png`
  - 原图：`S04_edit_panel_open.png` + `S08_final_render.png`
'''
manifest_md.write_text(manifest_text, encoding='utf-8')

print('Done')
print(fig44)
print(fig48)
print(fig51)
print(fig52)
print(fig53)
print(manifest_md)
