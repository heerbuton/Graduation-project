from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"F:\AIcharacter\End")
ASSET_ROOT = ROOT / "paper_assets" / "llm_notation_2026-04-03"
OUT_DIR = ASSET_ROOT / "09_paper_figures"
SNIPPET_DIR = OUT_DIR / "snippets"

MSYH = Path(r"C:\Windows\Fonts\msyh.ttc")
MSYH_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

WHITE = "#FFFFFF"
TEXT = "#1F2937"
MUTED = "#5B6472"
BORDER = "#C8D0DA"
LIGHT = "#F4F6F8"
BLUE = "#2E5CE6"
BLUE_FILL = "#EEF3FF"
GREEN = "#1E8A5A"
GREEN_FILL = "#ECF9F1"
RED = "#D14F4F"
RED_FILL = "#FFF0F0"
GOLD = "#9A6C00"
GOLD_FILL = "#FFF7E6"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_path = MSYH_BOLD if bold else MSYH
    return ImageFont.truetype(str(font_path), size=size)


def wrap_text(text: str, body_font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    if not text:
        return [""]

    lines: List[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue
        current = ""
        for ch in raw_line:
            probe = current + ch
            if body_font.getlength(probe) <= max_width or not current:
                current = probe
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines or [""]


def fit_box_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    body_font: ImageFont.FreeTypeFont,
    color: str,
    max_width: int,
    max_height: int | None = None,
    line_gap: int = 8,
) -> int:
    x, y = xy
    lines = wrap_text(text, body_font, max_width)
    bbox = draw.textbbox((0, 0), "测", font=body_font)
    line_height = bbox[3] - bbox[1]
    if max_height is not None:
        max_lines = max(1, (max_height + line_gap) // (line_height + line_gap))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines:
                truncated = lines[-1]
                while truncated and body_font.getlength(truncated + "...") > max_width:
                    truncated = truncated[:-1]
                lines[-1] = (truncated or "") + "..."
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * (line_height + line_gap)), line, fill=color, font=body_font)
    return y + len(lines) * (line_height + line_gap) - line_gap


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    fill: str = WHITE,
    outline: str = BORDER,
    width: int = 3,
    radius: int = 22,
):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    fill: str,
    width: int = 4,
    dash: int = 14,
    gap: int = 10,
):
    x1, y1 = start
    x2, y2 = end
    total = math.dist(start, end)
    if total == 0:
        return
    dx = (x2 - x1) / total
    dy = (y2 - y1) / total
    step = dash + gap
    drawn = 0.0
    while drawn < total:
        seg = min(dash, total - drawn)
        sx = x1 + dx * drawn
        sy = y1 + dy * drawn
        ex = x1 + dx * (drawn + seg)
        ey = y1 + dy * (drawn + seg)
        draw.line((sx, sy, ex, ey), fill=fill, width=width)
        drawn += step


def arrow_head(
    draw: ImageDraw.ImageDraw,
    end: Tuple[int, int],
    angle: float,
    color: str,
    size: int = 18,
):
    x, y = end
    a1 = angle + math.radians(155)
    a2 = angle - math.radians(155)
    p1 = (x + size * math.cos(a1), y + size * math.sin(a1))
    p2 = (x + size * math.cos(a2), y + size * math.sin(a2))
    draw.polygon([end, p1, p2], fill=color)


def straight_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: str,
    width: int = 5,
    dashed: bool = False,
):
    if dashed:
        dashed_line(draw, start, end, color, width=width)
    else:
        draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_head(draw, end, angle, color, size=20)


def polyline_arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[int, int]],
    color: str,
    width: int = 5,
    dashed: bool = False,
):
    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:]):
        if dashed:
            dashed_line(draw, start, end, color, width=width)
        else:
            draw.line((start, end), fill=color, width=width)
    end = points[-1]
    prev = points[-2]
    angle = math.atan2(end[1] - prev[1], end[0] - prev[0])
    arrow_head(draw, end, angle, color, size=20)


def badge(
    draw: ImageDraw.ImageDraw,
    pos: Tuple[int, int],
    text_value: str,
    fill: str,
    outline: str,
    text_color: str,
    body_font: ImageFont.FreeTypeFont,
    pad_x: int = 16,
    pad_y: int = 10,
    radius: int = 18,
) -> Tuple[int, int, int, int]:
    x, y = pos
    bbox = draw.textbbox((0, 0), text_value, font=body_font)
    w = bbox[2] - bbox[0] + pad_x * 2
    h = bbox[3] - bbox[1] + pad_y * 2
    box = (x, y, x + w, y + h)
    rounded_box(draw, box, fill=fill, outline=outline, width=2, radius=radius)
    draw.text((x + pad_x, y + pad_y - 2), text_value, fill=text_color, font=body_font)
    return box


def panel_title(
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    title: str,
    fill: str,
    title_font: ImageFont.FreeTypeFont,
):
    x1, y1, x2, _ = box
    title_box = (x1 + 18, y1 + 16, x2 - 18, y1 + 82)
    rounded_box(draw, title_box, fill=fill, outline=fill, width=0, radius=18)
    draw.text((x1 + 36, y1 + 27), title, fill=TEXT, font=title_font)


def code_panel(
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    title: str,
    code_text: str,
    title_fill: str = BLUE_FILL,
):
    rounded_box(draw, box, fill=WHITE, outline=BORDER, width=3, radius=24)
    panel_title(draw, box, title, title_fill, font(42, bold=True))
    x1, y1, x2, y2 = box
    inner = (x1 + 28, y1 + 98, x2 - 28, y2 - 28)
    rounded_box(draw, inner, fill=LIGHT, outline=LIGHT, width=0, radius=18)
    fit_box_text(
        draw,
        (inner[0] + 24, inner[1] + 18),
        code_text,
        font(24),
        TEXT,
        inner[2] - inner[0] - 48,
        max_height=inner[3] - inner[1] - 36,
        line_gap=6,
    )


def academic_panel(
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    title: str,
    fill: str = WHITE,
    outline: str = BORDER,
    accent: str | None = None,
):
    rounded_box(draw, box, fill=fill, outline=outline, width=3, radius=20)
    x1, y1, x2, _ = box
    if accent:
        draw.rounded_rectangle((x1 + 18, y1 + 16, x2 - 18, y1 + 28), radius=8, fill=accent, outline=accent, width=0)
    draw.text((x1 + 28, y1 + 42), title, fill=TEXT, font=font(48, bold=True))


def academic_code_panel(
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    title: str,
    code_text: str,
    outline: str = BORDER,
    accent: str | None = None,
):
    academic_panel(draw, box, title, fill=WHITE, outline=outline, accent=accent)
    x1, y1, x2, y2 = box
    inner = (x1 + 22, y1 + 98, x2 - 22, y2 - 22)
    rounded_box(draw, inner, fill=LIGHT, outline=LIGHT, width=0, radius=16)
    fit_box_text(
        draw,
        (inner[0] + 20, inner[1] + 18),
        code_text,
        font(24),
        TEXT,
        inner[2] - inner[0] - 40,
        max_height=inner[3] - inner[1] - 36,
        line_gap=6,
    )


def image_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: Sequence[int],
    title: str,
    source_path: Path,
):
    rounded_box(draw, box, fill=WHITE, outline=BORDER, width=3, radius=24)
    panel_title(draw, box, title, BLUE_FILL, font(36, bold=True))
    x1, y1, x2, y2 = box
    inner = (x1 + 24, y1 + 96, x2 - 24, y2 - 24)
    rounded_box(draw, inner, fill=LIGHT, outline=LIGHT, width=0, radius=18)
    image = Image.open(source_path).convert("RGB")
    image = ImageOps.contain(image, (inner[2] - inner[0] - 30, inner[3] - inner[1] - 30))
    paste_x = inner[0] + ((inner[2] - inner[0]) - image.width) // 2
    paste_y = inner[1] + ((inner[3] - inner[1]) - image.height) // 2
    canvas.paste(image, (paste_x, paste_y))


def save_snippet(path: Path, content: str):
    path.write_text(content.strip() + "\n", encoding="utf-8")


def pretty_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_fig_3_9(compact_groups, llm_notes):
    width, height = 4120, 1220
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    outer_y1, outer_y2 = 70, 1135
    gap = 38
    widths = [620, 620, 700, 760, 760]
    total_width = sum(widths) + gap * (len(widths) - 1)
    x = (width - total_width) // 2
    boxes = []
    headings = [
        ("减字组序列", BLUE_FILL),
        ("音位表增强", GOLD_FILL),
        ("提示词约束", BLUE_FILL),
        ("本地校验 / 容错修复", LIGHT),
        ("结构化输出", GREEN_FILL),
    ]
    for panel_w, (title, fill_color) in zip(widths, headings):
        box = (x, outer_y1, x + panel_w, outer_y2)
        boxes.append(box)
        rounded_box(draw, box, fill=WHITE, outline=BORDER, width=3, radius=28)
        panel_title(draw, box, title, fill_color, font(44, bold=True))
        x += panel_w + gap

    # Stage 1
    group_specs = [("action", 28), ("string", 72), ("position", 116), ("finger", 160)]
    for idx, group in enumerate(compact_groups[:3]):
        gx1, _, gx2, _ = boxes[0]
        card = (gx1 + 28, 155 + idx * 250, gx2 - 28, 385 + idx * 250)
        rounded_box(draw, card, fill=LIGHT, outline="#D7DEE7", width=2, radius=22)
        badge(draw, (card[0] + 18, card[1] + 18), group["group_id"], BLUE_FILL, BLUE, BLUE, font(26, bold=True))
        for label, dy in group_specs:
            draw.text((card[0] + 28, card[1] + dy + 50), f"{label}", fill=MUTED, font=font(27, bold=True))
            draw.text((card[0] + 180, card[1] + dy + 50), group.get(label, "") or "∅", fill=TEXT, font=font(30))

    # Stage 2
    kx1, _, kx2, _ = boxes[1]
    info_box = (kx1 + 28, 155, kx2 - 28, 1040)
    rounded_box(draw, info_box, fill=LIGHT, outline="#D7DEE7", width=2, radius=22)
    draw.text((info_box[0] + 24, info_box[1] + 28), "领域知识命中", fill=TEXT, font=font(34, bold=True))
    draw.text((info_box[0] + 24, info_box[1] + 82), "由弦位与徽位直接约束音高与八度", fill=MUTED, font=font(27))
    header_y = info_box[1] + 168
    col_x = [info_box[0] + 24, info_box[0] + 236, info_box[0] + 430]
    for hx, head in zip(col_x, ["命中项", "音高", "八度"]):
        draw.text((hx, header_y), head, fill=MUTED, font=font(28, bold=True))
    draw.line((info_box[0] + 22, header_y + 46, info_box[2] - 22, header_y + 46), fill=BORDER, width=2)
    rows = [
        ("6弦 空弦", "pitch=5", "octave=3"),
        ("1弦 九", "pitch=2", "octave=3"),
        ("4弦 七", "pitch=2", "octave=4"),
        ("2弦 十二三", "pitch=1", "octave=3"),
        ("7弦 一", "pitch=6", "octave=5"),
    ]
    for idx, row in enumerate(rows):
        yy = header_y + 78 + idx * 104
        draw.text((col_x[0], yy), row[0], fill=TEXT, font=font(31, bold=True))
        draw.text((col_x[1], yy), row[1], fill=TEXT, font=font(30))
        draw.text((col_x[2], yy), row[2], fill=TEXT, font=font(30))

    # Stage 3
    mx1, _, mx2, _ = boxes[2]
    llm_box = (mx1 + 45, 190, mx2 - 45, 1035)
    rounded_box(draw, llm_box, fill=BLUE_FILL, outline="#C9D8FF", width=3, radius=28)
    draw.text((llm_box[0] + 44, llm_box[1] + 62), "LLM 受约束生成", fill=TEXT, font=font(48, bold=True))
    draw.text((llm_box[0] + 44, llm_box[1] + 140), "qwen3.5-plus", fill=BLUE, font=font(36, bold=True))
    badge(draw, (llm_box[0] + 44, llm_box[1] + 236), "减字组序列", WHITE, BLUE, BLUE, font(27, bold=True))
    badge(draw, (llm_box[0] + 238, llm_box[1] + 236), "音位表增强", WHITE, BLUE, BLUE, font(27, bold=True))
    draw.text((llm_box[0] + 46, llm_box[1] + 332), "提示词约束", fill=MUTED, font=font(30, bold=True))
    draw.text((llm_box[0] + 46, llm_box[1] + 384), "完整上下文推断", fill=TEXT, font=font(32))
    draw.text((llm_box[0] + 46, llm_box[1] + 434), "音位表命中优先采用", fill=TEXT, font=font(32))
    draw.text((llm_box[0] + 46, llm_box[1] + 484), "JSON Schema 约束字段", fill=TEXT, font=font(32))
    inner_prompt = (llm_box[0] + 42, llm_box[1] + 470, llm_box[2] - 42, llm_box[3] - 34)
    rounded_box(draw, inner_prompt, fill=WHITE, outline="#D7E2FF", width=2, radius=20)
    fit_box_text(
        draw,
        (inner_prompt[0] + 24, inner_prompt[1] + 22),
        "notes 必须包含 group_id、pitch、octave、duration、action、string、position、finger、new_measure。",
        font(28),
        TEXT,
        inner_prompt[2] - inner_prompt[0] - 48,
        max_height=inner_prompt[3] - inner_prompt[1] - 44,
        line_gap=8,
    )

    # Stage 4
    vx1, _, vx2, _ = boxes[3]
    validator_box = (vx1 + 28, 155, vx2 - 28, 500)
    rounded_box(draw, validator_box, fill=LIGHT, outline="#D7DEE7", width=2, radius=22)
    draw.text((validator_box[0] + 26, validator_box[1] + 28), "本地校验", fill=TEXT, font=font(38, bold=True))
    check_lines = ["字段合法性校验", "数量一致性校验", "顺序一致性校验"]
    for idx, line in enumerate(check_lines):
        draw.text((validator_box[0] + 28, validator_box[1] + 110 + idx * 64), line, fill=TEXT, font=font(32, bold=True))
    success_y = 575
    polyline_arrow(
        draw,
        [(validator_box[0] + 60, success_y), (validator_box[2] - 50, success_y)],
        GREEN,
        width=6,
    )
    badge(draw, (validator_box[0] + 120, success_y - 44), "合法通过", GREEN_FILL, GREEN, GREEN, font(28, bold=True))

    repair_box = (vx1 + 28, 665, vx2 - 28, 1040)
    rounded_box(draw, repair_box, fill=RED_FILL, outline="#F2C3C3", width=2, radius=22)
    draw.text((repair_box[0] + 26, repair_box[1] + 28), "容错修复", fill=RED, font=font(38, bold=True))
    fit_box_text(
        draw,
        (repair_box[0] + 26, repair_box[1] + 108),
        "非法字段触发修复，仅修正非法字段，保持 group 顺序与上下文不变。",
        font(29),
        TEXT,
        repair_box[2] - repair_box[0] - 52,
        max_height=170,
        line_gap=8,
    )
    polyline_arrow(draw, [(validator_box[0] + 60, 640), (validator_box[2] - 50, 640)], RED, width=6, dashed=True)
    badge(draw, (validator_box[0] + 102, 592), "非法字段触发修复", RED_FILL, RED, RED, font(28, bold=True))

    # Stage 5
    output_notes = llm_notes[:6]
    rounded_box(draw, boxes[4], fill=WHITE, outline=BORDER, width=3, radius=28)
    panel_title(draw, boxes[4], "结构化输出", GREEN_FILL, font(44, bold=True))
    ox1, oy1, ox2, oy2 = boxes[4]
    inner_code = (ox1 + 70, oy1 + 120, ox2 - 70, oy2 - 48)
    rounded_box(draw, inner_code, fill=LIGHT, outline=LIGHT, width=0, radius=20)
    fit_box_text(
        draw,
        (inner_code[0] + 24, inner_code[1] + 20),
        pretty_json(output_notes),
        font(24),
        TEXT,
        inner_code[2] - inner_code[0] - 48,
        max_height=inner_code[3] - inner_code[1] - 40,
        line_gap=6,
    )

    # Connect stages with dedicated gutter arrows
    center_y = 540
    for idx in range(4):
        start = (boxes[idx][2] + 8, center_y)
        end = (boxes[idx + 1][0] - 8, center_y)
        straight_arrow(draw, start, end, BLUE, width=6)

    out_path = OUT_DIR / "figure_3_9_semantic_reconstruction.png"
    image.save(out_path, quality=95)
    return out_path


def build_fig_3_11(first_round_json: str, invalid_draft_json: str, repaired_json: str):
    width, height = 3600, 1520
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    def center_text(
        text: str,
        box: Sequence[int],
        body_font: ImageFont.FreeTypeFont,
        fill: str = TEXT,
        y_offset: int = 0,
    ):
        x1, y1, x2, y2 = box
        bbox = draw.textbbox((0, 0), text, font=body_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x1 + (x2 - x1 - text_w) // 2, y1 + (y2 - y1 - text_h) // 2 + y_offset),
            text,
            fill=fill,
            font=body_font,
        )

    def node(
        box: Sequence[int],
        title: str,
        lines: Sequence[str],
        accent: str = BLUE,
        fill: str = WHITE,
        outline: str = "#BFC8D4",
    ):
        rounded_box(draw, box, fill=fill, outline=outline, width=3, radius=22)
        x1, y1, x2, y2 = box
        draw.line((x1 + 26, y1 + 26, x1 + 26, y2 - 26), fill=accent, width=10)
        draw.text((x1 + 52, y1 + 34), title, fill=TEXT, font=font(42, bold=True))
        draw.line((x1 + 52, y1 + 98, x2 - 30, y1 + 98), fill="#D9DEE7", width=2)
        for idx, line in enumerate(lines):
            draw.text((x1 + 54, y1 + 126 + idx * 48), line, fill=MUTED, font=font(28))

    def code_chip(box: Sequence[int], title: str, code_text: str, accent: str = BLUE, outline: str = "#BFC8D4"):
        rounded_box(draw, box, fill=WHITE, outline=outline, width=3, radius=18)
        x1, y1, x2, y2 = box
        draw.text((x1 + 28, y1 + 26), title, fill=TEXT, font=font(36, bold=True))
        draw.line((x1 + 28, y1 + 82, x2 - 28, y1 + 82), fill=accent, width=6)
        inner = (x1 + 28, y1 + 108, x2 - 28, y2 - 28)
        rounded_box(draw, inner, fill=LIGHT, outline=LIGHT, width=0, radius=14)
        fit_box_text(
            draw,
            (inner[0] + 22, inner[1] + 18),
            code_text,
            font(24),
            TEXT,
            inner[2] - inner[0] - 44,
            max_height=inner[3] - inner[1] - 36,
            line_gap=5,
        )

    def diamond(box: Sequence[int], title: str, lines: Sequence[str]):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pts = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
        draw.polygon(pts, fill=WHITE)
        draw.line([pts[0], pts[1], pts[2], pts[3], pts[0]], fill="#8E99A8", width=4)
        center_text(title, (x1, y1 + 38, x2, y1 + 112), font(42, bold=True))
        for idx, line in enumerate(lines):
            center_text(line, (x1 + 70, y1 + 136 + idx * 44, x2 - 70, y1 + 178 + idx * 44), font(28), MUTED)

    def label(text_value: str, pos: Tuple[int, int], color: str):
        draw.text(pos, text_value, fill=color, font=font(26, bold=True))

    main_band = (120, 110, 3480, 650)
    repair_band = (510, 780, 3090, 1330)
    draw.rounded_rectangle(main_band, radius=34, fill="#FAFBFD", outline="#E5E9F0", width=2)
    draw.rounded_rectangle(repair_band, radius=34, fill="#FFF9F9", outline="#F4D2D2", width=2)
    draw.text((150, 136), "主生成路径", fill=BLUE, font=font(34, bold=True))
    draw.text((540, 806), "局部修复回路", fill=RED, font=font(34, bold=True))

    boxes = {
        "input": (180, 240, 610, 560),
        "main": (805, 220, 1295, 580),
        "first": (1485, 220, 1975, 580),
        "validator": (2185, 190, 2685, 610),
        "final": (2930, 240, 3420, 560),
        "repair": (720, 920, 1280, 1248),
        "repaired": (1570, 920, 2110, 1248),
    }

    input_excerpt = "\n".join(
        [
            "X = [g1, g2, ...]",
            "g1: action=历, string=六",
            "fields: hui / xian / finger",
        ]
    )
    first_excerpt = "\n".join(
        [
            "y0 = [",
            '  {"group_id":"group_1",',
            '   "pitch":"1", "octave":"4",',
            '   "duration":"4", ...}',
            "]",
        ]
    )
    repaired_excerpt = "\n".join(
        [
            "y1 = repair(y0, e)",
            'pitch: "9"  -> "5"',
            'octave: "7" -> "3"',
            "group order unchanged",
        ]
    )

    code_chip(boxes["input"], "减字组序列 X", input_excerpt, accent=BLUE, outline="#C9D8FF")
    node(
        boxes["main"],
        "主生成器 G",
        ["主提示词约束生成", "完整上下文 + 音位表", "强制 JSON Schema"],
        accent=BLUE,
        outline="#C9D8FF",
    )
    code_chip(boxes["first"], "首轮 JSON y0", first_excerpt, accent=BLUE, outline="#C9D8FF")
    diamond(boxes["validator"], "V(y)", ["字段合法性", "数量一致性", "顺序一致性"])
    node(
        boxes["final"],
        "最终结果 y*",
        ["结构化 JSON", "MusicXML 标准化", "进入谱面渲染"],
        accent=GREEN,
        outline="#B5E0CA",
    )
    node(
        boxes["repair"],
        "修复提示词 R",
        ["只修正非法字段", "不重写全部上下文", "保持 group 顺序"],
        accent=RED,
        fill=WHITE,
        outline="#F0B4B4",
    )
    code_chip(boxes["repaired"], "修复后 JSON y1", repaired_excerpt, accent=GREEN, outline="#B5E0CA")

    main_y = 400
    straight_arrow(draw, (boxes["input"][2] + 18, main_y), (boxes["main"][0] - 18, main_y), BLUE, width=6)
    straight_arrow(draw, (boxes["main"][2] + 18, main_y), (boxes["first"][0] - 18, main_y), BLUE, width=6)
    straight_arrow(draw, (boxes["first"][2] + 18, main_y), (boxes["validator"][0] - 18, main_y), BLUE, width=6)
    straight_arrow(draw, (boxes["validator"][2] + 18, main_y), (boxes["final"][0] - 18, main_y), GREEN, width=6)
    label("合法通过", (2725, 348), GREEN)

    validator_bottom = ((boxes["validator"][0] + boxes["validator"][2]) // 2, boxes["validator"][3] + 12)
    repair_top = ((boxes["repair"][0] + boxes["repair"][2]) // 2, boxes["repair"][1] - 16)
    polyline_arrow(
        draw,
        [validator_bottom, (2435, 720), (1000, 720), repair_top],
        RED,
        width=5,
        dashed=True,
    )
    label("校验失败 e", (2460, 676), RED)

    straight_arrow(
        draw,
        (boxes["repair"][2] + 18, 1085),
        (boxes["repaired"][0] - 18, 1085),
        RED,
        width=5,
        dashed=True,
    )
    label("局部约束修复", (1328, 1038), RED)

    polyline_arrow(
        draw,
        [
            ((boxes["repaired"][0] + boxes["repaired"][2]) // 2, boxes["repaired"][1] - 16),
            (1840, 760),
            (2435, 760),
            ((boxes["validator"][0] + boxes["validator"][2]) // 2, boxes["validator"][3] + 12),
        ],
        RED,
        width=5,
        dashed=True,
    )
    label("再次校验", (1910, 716), RED)

    note_box = (2280, 940, 2940, 1210)
    rounded_box(draw, note_box, fill=WHITE, outline="#E6D6B5", width=2, radius=18)
    draw.text((note_box[0] + 28, note_box[1] + 26), "校验器约束", fill=GOLD, font=font(34, bold=True))
    for idx, line in enumerate(["pitch=1-7", "octave=3/4/5", "duration=2/4/8/16", "new_measure 为布尔值"]):
        draw.text((note_box[0] + 34, note_box[1] + 88 + idx * 42), line, fill=MUTED, font=font(26))

    out_path = OUT_DIR / "figure_3_11_prompt_repair_loop.png"
    image.save(out_path, quality=95)
    return out_path


def build_fig_5_4(xml_snippet: str):
    width, height = 3000, 2200
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    margin_x, margin_y = 110, 130
    gap_x, gap_y = 90, 100
    panel_w = (width - margin_x * 2 - gap_x) // 2
    panel_h = (height - margin_y * 2 - gap_y) // 2

    tl = (margin_x, margin_y, margin_x + panel_w, margin_y + panel_h)
    tr = (margin_x + panel_w + gap_x, margin_y, width - margin_x, margin_y + panel_h)
    bl = (margin_x, margin_y + panel_h + gap_y, margin_x + panel_w, height - margin_y)
    br = (margin_x + panel_w + gap_x, margin_y + panel_h + gap_y, width - margin_x, height - margin_y)

    image_panel(image, draw, tl, "输入谱例", ASSET_ROOT / "07_extra_materials" / "testpicture-1.jpg")
    image_panel(image, draw, tr, "中间结果", ASSET_ROOT / "05_conversion_evidence" / "testpicture-1_group_boxes_only_v3.jpg")
    image_panel(image, draw, bl, "终态谱面", ASSET_ROOT / "05_conversion_evidence" / "scoremodel-after-fix.png")
    code_panel(draw, br, "MusicXML片段", xml_snippet, title_fill=GREEN_FILL)

    # Arrows between panels routed through gutters
    top_center_y = tl[1] + panel_h // 2
    straight_arrow(draw, (tl[2] + 14, top_center_y), (tr[0] - 14, top_center_y), BLUE, width=6)

    gutter_x = (tl[2] + tr[0]) // 2
    polyline_arrow(
        draw,
        [
            ((tr[0] + tr[2]) // 2, tr[3] + 14),
            ((tr[0] + tr[2]) // 2, tr[3] + 52),
            (gutter_x, tr[3] + 52),
            (gutter_x, bl[1] - 24),
            ((bl[0] + bl[2]) // 2, bl[1] - 24),
        ],
        BLUE,
        width=6,
    )
    polyline_arrow(
        draw,
        [(bl[2] + 14, bl[1] + panel_h // 2), (br[0] - 14, br[1] + panel_h // 2)],
        BLUE,
        width=6,
    )

    out_path = OUT_DIR / "figure_5_4_case_study_rendering.png"
    image.save(out_path, quality=95)
    return out_path


def build_fig_3_9_publication(compact_groups, llm_notes):
    """Five-stage method figure with strict column alignment."""
    width, height = 3200, 1160
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    def stage_box(box: Sequence[int], idx: str, heading: str, accent: str, fill: str = WHITE):
        rounded_box(draw, box, fill=fill, outline="#CBD3DF", width=3, radius=22)
        x1, y1, x2, y2 = box
        draw.rounded_rectangle((x1 + 22, y1 + 22, x1 + 84, y1 + 76), radius=14, fill=accent)
        draw.text((x1 + 37, y1 + 31), idx, fill=WHITE, font=font(25, bold=True))
        draw.text((x1 + 104, y1 + 28), heading, fill=TEXT, font=font(30, bold=True))
        draw.line((x1 + 26, y1 + 96, x2 - 26, y1 + 96), fill="#D9DEE7", width=2)

    def body_lines(box: Sequence[int], lines: Sequence[str], size: int = 24, start_y: int = 128, gap: int = 40, color: str = TEXT):
        x1, y1, _, _ = box
        for idx, line in enumerate(lines):
            draw.text((x1 + 34, y1 + start_y + idx * gap), line, fill=color, font=font(size))

    def small_card(box: Sequence[int], title: str, lines: Sequence[str], accent: str = BLUE, fill: str = LIGHT):
        rounded_box(draw, box, fill=fill, outline="#E1E6EE", width=2, radius=14)
        x1, y1, x2, _ = box
        draw.text((x1 + 18, y1 + 14), title, fill=accent, font=font(22, bold=True))
        draw.line((x1 + 18, y1 + 48, x2 - 18, y1 + 48), fill="#D9DEE7", width=2)
        for idx, line in enumerate(lines):
            draw.text((x1 + 20, y1 + 66 + idx * 32), line, fill=TEXT, font=font(20))

    def pill(pos: Tuple[int, int], label: str, color: str, fill: str):
        badge(draw, pos, label, fill=fill, outline=color, text_color=color, body_font=font(22, bold=True), pad_x=14, pad_y=8, radius=16)

    def centered_label(text_value: str, center: Tuple[int, int], color: str):
        label_font = font(24, bold=True)
        bbox = draw.textbbox((0, 0), text_value, font=label_font)
        draw.text((center[0] - (bbox[2] - bbox[0]) // 2, center[1]), text_value, fill=color, font=label_font)

    draw.text((120, 66), "基于中间表示与知识注入的语义重构", fill=TEXT, font=font(42, bold=True))
    draw.text(
        (122, 124),
        "五个阶段共用同一水平网格：结构化减字组先注入音位表知识，再由提示词约束生成并通过本地校验。",
        fill=MUTED,
        font=font(28),
    )

    top, bottom = 238, 1000
    xs = [120, 720, 1260, 1800, 2400]
    ws = [540, 480, 480, 540, 560]
    boxes = {name: (x, top, x + w, bottom) for name, x, w in zip(["x", "k", "g", "v", "y"], xs, ws)}

    sample = list(compact_groups[:3])
    while len(sample) < 3:
        sample.append({"group_id": f"group_{len(sample) + 1}", "action": "-", "string": "-", "hui": "-"})

    stage_box(boxes["x"], "01", "减字组序列", BLUE)
    x1, y1, x2, _ = boxes["x"]
    for idx, group in enumerate(sample[:3]):
        card = (x1 + 28, y1 + 128 + idx * 174, x2 - 28, y1 + 278 + idx * 174)
        small_card(
            card,
            group.get("group_id", f"group_{idx + 1}"),
            [
                f"action={group.get('action') or '-'}    string={group.get('string') or '-'}",
                f"hui={group.get('hui') or '-'}    finger={group.get('finger') or '-'}",
            ],
            BLUE,
        )

    stage_box(boxes["k"], "02", "音位表增强", GOLD, fill="#FFFCF4")
    k_lines = []
    for group in sample[:3]:
        ref = group.get("tone_table_ref") or "弦位/徽位未命中"
        pitch = group.get("tone_table_pitch") or "-"
        octave = group.get("tone_table_octave") or "-"
        k_lines.append(f"{ref} -> {pitch}/{octave}")
    body_lines(boxes["k"], ["由弦位与徽位约束音高", "命中项优先于经验推断", ""] + k_lines, size=22, gap=38, color=TEXT)
    pill((boxes["k"][0] + 34, boxes["k"][3] - 88), "音位表增强", GOLD, GOLD_FILL)

    stage_box(boxes["g"], "03", "LLM 受约束生成", BLUE, fill=BLUE_FILL)
    gx1, gy1, gx2, _ = boxes["g"]
    draw.text((gx1 + 48, gy1 + 160), "G(X, K, P)", fill=BLUE, font=font(42, bold=True))
    draw.line((gx1 + 40, gy1 + 228, gx2 - 40, gy1 + 228), fill="#C9D8FF", width=3)
    body_lines(
        boxes["g"],
        ["提示词约束", "动作语义补全", "音高 / 八度推断", "节拍与小节组织", "按 group 顺序输出"],
        size=25,
        start_y=274,
        gap=48,
    )

    stage_box(boxes["v"], "04", "本地校验与修复", "#8E99A8", fill="#FAFBFD")
    vx1, vy1, vx2, _ = boxes["v"]
    valid_box = (vx1 + 34, vy1 + 136, vx2 - 34, vy1 + 390)
    repair_box = (vx1 + 34, vy1 + 470, vx2 - 34, vy1 + 690)
    small_card(valid_box, "本地校验", ["字段合法性", "数量一致性", "顺序一致性"], "#6B7280", fill=WHITE)
    small_card(repair_box, "容错修复", ["非法字段触发", "仅局部修正", "不重写上下文"], RED, fill=RED_FILL)
    straight_arrow(draw, ((valid_box[0] + valid_box[2]) // 2, valid_box[3] + 14), ((repair_box[0] + repair_box[2]) // 2, repair_box[1] - 14), RED, width=4, dashed=True)
    centered_label("非法字段", ((valid_box[0] + valid_box[2]) // 2 + 90, valid_box[3] + 18), RED)

    stage_box(boxes["y"], "05", "结构化输出", GREEN, fill=GREEN_FILL)
    yx1, yy1, yx2, yy2 = boxes["y"]
    code_inner = (yx1 + 34, yy1 + 140, yx2 - 34, yy2 - 54)
    rounded_box(draw, code_inner, fill=WHITE, outline="#D8EBDD", width=2, radius=14)
    output_text = '{\n  "notes": [\n    {"pitch":1,\n     "octave":4,\n     "duration":4,\n     "action":"历",\n     "measure":true}\n  ]\n}'
    fit_box_text(draw, (code_inner[0] + 24, code_inner[1] + 22), output_text, font(20), TEXT, code_inner[2] - code_inner[0] - 48, max_height=code_inner[3] - code_inner[1] - 44, line_gap=5)

    # One ruler-straight main chain. Secondary repair relation stays inside stage 04.
    main_y = 610
    chain = ["x", "k", "g", "v", "y"]
    for left_name, right_name in zip(chain[:-1], chain[1:]):
        left = boxes[left_name]
        right = boxes[right_name]
        color = GREEN if left_name == "v" else BLUE
        straight_arrow(draw, (left[2] + 18, main_y), (right[0] - 18, main_y), color, width=6)
    centered_label("合法通过", ((boxes["v"][2] + boxes["y"][0]) // 2, main_y - 52), GREEN)

    out_path = OUT_DIR / "figure_3_9_semantic_reconstruction.png"
    image.save(out_path, quality=95)
    return out_path


def build_fig_3_11_publication(first_round_json: str, invalid_draft_json: str, repaired_json: str):
    """Two-lane prompt generation and local repair loop with fixed grid routing."""
    width, height = 3200, 1160
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    def process_box(box: Sequence[int], title_text: str, lines: Sequence[str], accent: str, outline: str = "#CAD2DE", fill: str = WHITE):
        rounded_box(draw, box, fill=fill, outline=outline, width=3, radius=20)
        x1, y1, x2, y2 = box
        draw.line((x1 + 24, y1 + 24, x2 - 24, y1 + 24), fill=accent, width=8)
        draw.text((x1 + 30, y1 + 48), title_text, fill=TEXT, font=font(31, bold=True))
        draw.line((x1 + 30, y1 + 98, x2 - 30, y1 + 98), fill="#D9DEE7", width=2)
        for idx, line in enumerate(lines):
            draw.text((x1 + 34, y1 + 130 + idx * 38), line, fill=MUTED, font=font(23))

    def code_box(box: Sequence[int], title_text: str, code_text: str, accent: str):
        rounded_box(draw, box, fill=WHITE, outline="#CAD2DE", width=3, radius=20)
        x1, y1, x2, y2 = box
        draw.line((x1 + 24, y1 + 24, x2 - 24, y1 + 24), fill=accent, width=8)
        draw.text((x1 + 30, y1 + 48), title_text, fill=TEXT, font=font(29, bold=True))
        draw.line((x1 + 30, y1 + 96, x2 - 30, y1 + 96), fill="#D9DEE7", width=2)
        inner = (x1 + 28, y1 + 118, x2 - 28, y2 - 24)
        rounded_box(draw, inner, fill=LIGHT, outline=LIGHT, width=0, radius=13)
        fit_box_text(
            draw,
            (inner[0] + 18, inner[1] + 14),
            code_text,
            font(19),
            TEXT,
            inner[2] - inner[0] - 36,
            max_height=inner[3] - inner[1] - 28,
            line_gap=4,
        )

    def lane_label(text_value: str, pos: Tuple[int, int], color: str):
        draw.text(pos, text_value, fill=color, font=font(24, bold=True))

    draw.text((120, 66), "大语言模型主生成与修复提示词协同", fill=TEXT, font=font(42, bold=True))
    draw.text((122, 124), "上方为主生成路径；下方为局部修复路径。所有反馈线只在固定泳道内转折，避免重写全部上下文。", fill=MUTED, font=font(28))

    col_x = [120, 650, 1180, 1710, 2440]
    box_w = 430
    top_y, box_h = 248, 264
    bottom_y = 764
    lane_y = 644
    boxes = {
        "input": (col_x[0], top_y, col_x[0] + box_w, top_y + box_h),
        "main": (col_x[1], top_y, col_x[1] + box_w, top_y + box_h),
        "y0": (col_x[2], top_y, col_x[2] + box_w, top_y + box_h),
        "validator": (col_x[3], top_y, col_x[3] + box_w, top_y + box_h),
        "final": (col_x[4], top_y, col_x[4] + 540, top_y + box_h),
        "error": (col_x[1], bottom_y, col_x[1] + box_w, bottom_y + box_h),
        "repair": (col_x[2], bottom_y, col_x[2] + box_w, bottom_y + box_h),
        "y1": (col_x[3], bottom_y, col_x[3] + box_w, bottom_y + box_h),
    }

    draw.text((120, 202), "主生成路径", fill=BLUE, font=font(24, bold=True))
    draw.line((250, 216, 2980, 216), fill="#E5EAF2", width=2)
    draw.text((650, 716), "局部修复路径", fill=RED, font=font(24, bold=True))
    draw.line((818, 730, 2140, 730), fill="#F1D0D0", width=2)

    code_box(boxes["input"], "减字组序列", "X=[g1,g2,...]\ng1: action=历\n    string=六\n    hui=-", BLUE)
    process_box(boxes["main"], "主生成提示词", ["完整上下文", "音位表增强", "JSON Schema", "按 group 顺序"], BLUE, "#C9D8FF", BLUE_FILL)
    code_box(boxes["y0"], "首轮 JSON y0", "notes=[\n {pitch:1, octave:4,\n  duration:4, ...}\n]", BLUE)
    process_box(
        boxes["validator"],
        "本地校验器 V(y)",
        ["字段合法性校验", "数量与顺序一致", "pitch/octave/duration 值域"],
        "#6B7280",
        "#BFC7D2",
        "#FAFBFD",
    )
    process_box(boxes["final"], "最终结果 y*", ["结构化 JSON", "MusicXML 标准化输出", "终态谱面渲染"], GREEN, "#B5E0CA", GREEN_FILL)
    process_box(boxes["error"], "错误摘要 e", ["非法字段集合", "错误位置 group_id", "失败原因最小化"], RED, "#F0B4B4", RED_FILL)
    process_box(boxes["repair"], "修复提示词 R", ["只修正非法字段", "保持 group 数量", "不重写全部上下文"], RED, "#F0B4B4", RED_FILL)
    code_box(boxes["y1"], "修复后 JSON y1", 'repair(y0,e)\npitch "9" -> "5"\noctave "7" -> "3"\norder unchanged', GREEN)

    main_y = top_y + box_h // 2
    for left_name, right_name, color in [
        ("input", "main", BLUE),
        ("main", "y0", BLUE),
        ("y0", "validator", BLUE),
        ("validator", "final", GREEN),
    ]:
        left = boxes[left_name]
        right = boxes[right_name]
        straight_arrow(draw, (left[2] + 18, main_y), (right[0] - 18, main_y), color, width=6)
    lane_label("合法通过", (boxes["validator"][2] + 62, main_y - 48), GREEN)

    validator_cx = (boxes["validator"][0] + boxes["validator"][2]) // 2
    error_cx = (boxes["error"][0] + boxes["error"][2]) // 2
    y1_cx = (boxes["y1"][0] + boxes["y1"][2]) // 2
    bottom_mid_y = bottom_y + box_h // 2

    polyline_arrow(
        draw,
        [
            (validator_cx, boxes["validator"][3] + 14),
            (validator_cx, lane_y),
            (error_cx, lane_y),
            (error_cx, boxes["error"][1] - 16),
        ],
        RED,
        width=5,
        dashed=True,
    )
    lane_label("校验失败 e", (validator_cx + 34, lane_y - 42), RED)
    straight_arrow(draw, (boxes["error"][2] + 18, bottom_mid_y), (boxes["repair"][0] - 18, bottom_mid_y), RED, width=5, dashed=True)
    lane_label("生成修复提示词", (boxes["error"][2] + 42, bottom_mid_y - 45), RED)
    straight_arrow(draw, (boxes["repair"][2] + 18, bottom_mid_y), (boxes["y1"][0] - 18, bottom_mid_y), RED, width=5, dashed=True)
    lane_label("局部约束修复", (boxes["repair"][2] + 44, bottom_mid_y - 45), RED)
    polyline_arrow(
        draw,
        [
            (y1_cx, boxes["y1"][1] - 16),
            (y1_cx, lane_y + 42),
            (validator_cx, lane_y + 42),
            (validator_cx, boxes["validator"][3] + 16),
        ],
        RED,
        width=5,
        dashed=True,
    )
    lane_label("再次校验", (y1_cx + 28, lane_y + 2), RED)

    out_path = OUT_DIR / "figure_3_11_prompt_repair_loop.png"
    image.save(out_path, quality=95)
    return out_path


def build_fig_5_4_publication(xml_snippet: str):
    """Publication-oriented qualitative grid: real input, aggregation, render, standardized output."""
    width, height = 3200, 2180
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    def trim_content(img: Image.Image, pad: int = 24, threshold: int = 245) -> Image.Image:
        gray = img.convert("L")
        mask = gray.point(lambda p: 255 if p < threshold else 0)
        bbox = mask.getbbox()
        if not bbox:
            return img
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.width, x2 + pad)
        y2 = min(img.height, y2 + pad)
        return img.crop((x1, y1, x2, y2))

    def panel(box: Sequence[int], label_text: str, title_text: str):
        x1, y1, x2, y2 = box
        rounded_box(draw, box, fill=WHITE, outline="#CAD2DE", width=2, radius=18)
        draw.text((x1 + 24, y1 + 20), label_text, fill=BLUE, font=font(30, bold=True))
        draw.text((x1 + 88, y1 + 20), title_text, fill=TEXT, font=font(30, bold=True))
        draw.line((x1 + 24, y1 + 70, x2 - 24, y1 + 70), fill="#D9DEE7", width=2)
        return (x1 + 24, y1 + 92, x2 - 24, y2 - 24)

    def paste_fit(src_path: Path, inner: Sequence[int], crop: Tuple[int, int, int, int] | None = None, trim: bool = False):
        src = Image.open(src_path).convert("RGB")
        if crop is not None:
            src = src.crop(crop)
        if trim:
            src = trim_content(src)
        target_w = inner[2] - inner[0]
        target_h = inner[3] - inner[1]
        src = ImageOps.contain(src, (target_w, target_h))
        px = inner[0] + (target_w - src.width) // 2
        py = inner[1] + (target_h - src.height) // 2
        image.paste(src, (px, py))

    def code_panel_compact(inner: Sequence[int], text_value: str):
        rounded_box(draw, inner, fill=LIGHT, outline=LIGHT, width=0, radius=14)
        fit_box_text(
            draw,
            (inner[0] + 26, inner[1] + 22),
            text_value,
            font(24),
            TEXT,
            inner[2] - inner[0] - 52,
            max_height=inner[3] - inner[1] - 44,
            line_gap=5,
        )

    margin_x, margin_y = 90, 90
    gap_x, gap_y = 70, 86
    panel_w = (width - margin_x * 2 - gap_x) // 2
    panel_h = (height - margin_y * 2 - gap_y) // 2
    tl = (margin_x, margin_y, margin_x + panel_w, margin_y + panel_h)
    tr = (margin_x + panel_w + gap_x, margin_y, width - margin_x, margin_y + panel_h)
    bl = (margin_x, margin_y + panel_h + gap_y, margin_x + panel_w, height - margin_y)
    br = (margin_x + panel_w + gap_x, margin_y + panel_h + gap_y, width - margin_x, height - margin_y)

    inner_tl = panel(tl, "(a)", "输入谱例")
    inner_tr = panel(tr, "(b)", "减字组聚合结果")
    inner_bl = panel(bl, "(c)", "终态谱面渲染")
    inner_br = panel(br, "(d)", "标准化 MusicXML 片段")

    paste_fit(ASSET_ROOT / "07_extra_materials" / "testpicture-1.jpg", inner_tl, trim=True)
    paste_fit(ASSET_ROOT / "05_conversion_evidence" / "testpicture-1_group_boxes_only_v3.jpg", inner_tr, trim=True)
    # Crop only the rendered score region from the existing system screenshot.
    paste_fit(ASSET_ROOT / "05_conversion_evidence" / "scoremodel-after-measure-wrap-fix.png", inner_bl, crop=(805, 185, 1505, 1305), trim=True)
    code_panel_compact(inner_br, xml_snippet)

    straight_arrow(draw, (tl[2] + 12, tl[1] + panel_h // 2), (tr[0] - 12, tr[1] + panel_h // 2), BLUE, width=5)
    center_gutter_x = (tl[2] + tr[0]) // 2
    route_y = tr[3] + gap_y // 2
    polyline_arrow(
        draw,
        [
            (tr[0] + panel_w // 2, tr[3] + 16),
            (tr[0] + panel_w // 2, route_y),
            (center_gutter_x, route_y),
            (center_gutter_x, bl[1] - 18),
            (bl[0] + panel_w // 2, bl[1] - 18),
        ],
        BLUE,
        width=5,
    )
    straight_arrow(draw, (bl[2] + 12, bl[1] + panel_h // 2), (br[0] - 12, br[1] + panel_h // 2), BLUE, width=5)

    out_path = OUT_DIR / "figure_5_4_case_study_rendering.png"
    image.save(out_path, quality=95)
    return out_path


def extract_musicxml_snippet(xml_path: Path) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    part = root.find("part")
    if part is None:
        return xml_path.read_text(encoding="utf-8")[:1200]
    measures = part.findall("measure")
    if len(measures) < 2:
        return xml_path.read_text(encoding="utf-8")[:1200]

    snippet_root = ET.Element("score-partwise", version="3.1")
    snippet_part = ET.SubElement(snippet_root, "part", id="P1")
    snippet_part.append(measures[1])
    xml_text = ET.tostring(snippet_root, encoding="unicode")
    return xml_text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SNIPPET_DIR.mkdir(parents=True, exist_ok=True)

    compact_groups = load_json(ASSET_ROOT / "01_input_data_spec" / "input_sample_05_compact_with_tone_table_hits.json")
    notes_full = load_json(ASSET_ROOT / "05_conversion_evidence" / "testpicture-1_notation_result_full.json")
    xml_snippet = extract_musicxml_snippet(ASSET_ROOT / "05_conversion_evidence" / "musicxml_full_from_testpicture.xml")

    first_round_json = pretty_json(notes_full[:3])

    invalid_draft = [
        {
            "group_id": "group_1",
            "pitch": "9",
            "octave": "7",
            "duration": "32",
            "action": "历",
            "string": "六",
            "position": "",
            "finger": "",
            "new_measure": False,
        }
    ]
    repaired_output = [
        {
            "group_id": "group_1",
            "pitch": "5",
            "octave": "3",
            "duration": "4",
            "action": "历",
            "string": "六",
            "position": "",
            "finger": "",
            "new_measure": False,
        },
        {
            "group_id": "group_2",
            "pitch": "2",
            "octave": "3",
            "duration": "4",
            "action": "勾",
            "string": "一",
            "position": "九",
            "finger": "大",
            "new_measure": False,
        },
        {
            "group_id": "group_3",
            "pitch": "2",
            "octave": "4",
            "duration": "4",
            "action": "抹",
            "string": "四",
            "position": "七",
            "finger": "中",
            "new_measure": False,
        },
    ]

    fig_3_9 = build_fig_3_9_publication(compact_groups, notes_full)
    fig_3_11 = build_fig_3_11_publication(pretty_json(notes_full[:3]), pretty_json(invalid_draft), pretty_json(repaired_output))
    fig_5_4 = build_fig_5_4_publication(xml_snippet[:1200] + "\n...")

    save_snippet(SNIPPET_DIR / "figure_3_9_input_groups.json", pretty_json(compact_groups))
    save_snippet(SNIPPET_DIR / "figure_3_9_output_notes.json", pretty_json(notes_full[:8]))
    save_snippet(SNIPPET_DIR / "figure_3_11_first_round_output.json", pretty_json(notes_full[:3]))
    save_snippet(SNIPPET_DIR / "figure_3_11_invalid_draft.json", pretty_json(invalid_draft))
    save_snippet(SNIPPET_DIR / "figure_3_11_repaired_output.json", pretty_json(repaired_output))
    save_snippet(
        SNIPPET_DIR / "figure_3_11_validator_rules.txt",
        "\n".join(
            [
                "字段合法性校验：pitch in 1..7, octave in {3,4,5}, duration in {2,4,8,16}, new_measure 为布尔。",
                "数量一致性校验：输出 note 数量必须与输入 group 数量一致。",
                "顺序一致性验证：group_id 必须与输入顺序完全一致。",
                "修复策略：仅修正非法字段，不重写全部上下文。",
            ]
        ),
    )
    save_snippet(SNIPPET_DIR / "figure_5_4_musicxml_snippet.xml", xml_snippet[:1800])
    save_snippet(SNIPPET_DIR / "figure_5_4_structured_notes.json", pretty_json(notes_full[:10]))

    readme = "\n".join(
        [
            "# 论文插图导出",
            "",
            "## PNG 文件",
            f"- {fig_3_9.name}",
            f"- {fig_3_11.name}",
            f"- {fig_5_4.name}",
            "",
            "## 关键片段",
            "- snippets/figure_3_9_input_groups.json",
            "- snippets/figure_3_9_output_notes.json",
            "- snippets/figure_3_11_first_round_output.json",
            "- snippets/figure_3_11_invalid_draft.json",
            "- snippets/figure_3_11_repaired_output.json",
            "- snippets/figure_3_11_validator_rules.txt",
            "- snippets/figure_5_4_musicxml_snippet.xml",
            "- snippets/figure_5_4_structured_notes.json",
            "",
              "## 生成说明",
              "- 图3-9 使用真实减字组样例与音位表命中结果重绘方法图，强调 X/K/P 三类约束如何共同完成语义重构。",
              "- 图3-11 使用本地修复提示样例中的非法字段案例展示主生成、校验门与局部修复闭环。",
              "- 图5-4 使用 testpicture-1 的原始输入、聚合结果、裁剪后的终态谱面渲染与 MusicXML 片段。",
              "- 本次未重新运行 YOLO/上传链路，沿用线程中已保存的成功样例资产。",
              "- 模型配置沿用当前后端：qwen3.5-plus，temperature=0.0。",
              "",
              "## Caption / Alt Text 草案",
              "- 图3-9 caption：中间表示、音位表知识和提示词约束共同限制 LLM 的减字谱语义重构过程，校验与修复模块进一步保证结构化输出合法。",
              "- 图3-9 alt：图中从减字组序列 X、音位表 K 和提示词约束 P 三路输入进入 LLM 语义重构器，经本地校验和局部修复后输出标准化字段。",
              "- 图3-11 caption：LLM 首轮生成结果先进入本地校验门，合法结果直接输出；非法字段被送入修复提示词进行局部修正，并再次校验。",
              "- 图3-11 alt：闭环流程图展示减字组序列经主生成提示词得到 JSON，校验失败时生成错误反馈并进入修复提示词，修复后返回校验器。",
              "- 图5-4 caption：代表性测试样例从原始减字谱输入、减字组检测聚合，到终态谱面渲染与 MusicXML 标准化片段的实测链路。",
              "- 图5-4 alt：四面板图展示同一测试图片的原始谱例、红框聚合检测结果、裁剪后的现代谱面渲染和对应 MusicXML 代码片段。",
          ]
      )
    save_snippet(OUT_DIR / "README.md", readme)


if __name__ == "__main__":
    main()
