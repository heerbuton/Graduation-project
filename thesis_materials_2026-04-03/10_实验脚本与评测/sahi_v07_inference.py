import cv2
import numpy as np
import os
import torch
import torchvision
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


# ==========================================
# 工程红线底层 I/O 模块
# ==========================================
def read_image_safe(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到文件: {image_path}")
    img_array = np.fromfile(image_path, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def save_image_safe(image_mat, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    is_success, im_buf_arr = cv2.imencode(".jpg", image_mat)
    if is_success:
        im_buf_arr.tofile(save_path)
        print(f"\n✅ 成功保存整页拼接渲染结果至: {save_path}")
    else:
        print("❌ 图片编码保存失败！")


def draw_chinese_boxes(cv2_img, boxes, classes, scores, class_names):
    """
    使用 PIL 绘制中文标签，避开 cv2.putText 的中文乱码缺陷
    """
    # cv2(BGR) 转 PIL(RGB)
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试加载 Windows 默认中文字体，若失败则使用默认字体
    try:
        font = ImageFont.truetype("msyh.ttc", 20)  # 微软雅黑
    except:
        try:
            font = ImageFont.truetype("simsun.ttc", 20)  # 宋体
        except:
            font = ImageFont.load_default()

    for box, cls_id, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        label = f"{class_names[int(cls_id)]} {score:.2f}"

        # 画框 (红色)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        # 画文字底色背景与文字
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=(255, 0, 0))
        draw.text((x1 + 1, y1 - 1), label, font=font, fill=(255, 255, 255))

    # PIL(RGB) 转回 cv2(BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ==========================================
# 核心业务逻辑：SAHI 融合推理
# ==========================================
def sahi_sliding_window_inference(model, img, window_size=256, stride=128, conf_thresh=0.08):
    """
    1. 网格切割 2. 等比放大推理 3. 坐标映射还原 4. Batched NMS 去重
    """
    img_h, img_w = img.shape[:2]

    # V07 的核心物理放大系数：256 放大到 640 = 2.5倍
    scale_factor = 640.0 / window_size

    all_boxes = []
    all_scores = []
    all_classes = []

    print(f"\n--- 🚀 启动 SAHI 滑动切片推理 ---")
    print(f"原图尺寸: {img_w}x{img_h} | 切片窗口: {window_size}x{window_size} | 步长: {stride}")

    patch_count = 0
    # 1. 扫描仪启动：按步长滑动窗口
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            # 防止窗口越界
            y1 = min(y, max(0, img_h - window_size))
            x1 = min(x, max(0, img_w - window_size))
            y2 = min(y1 + window_size, img_h)
            x2 = min(x1 + window_size, img_w)

            # 提取物理切片
            patch = img[y1:y2, x1:x2]

            # 智能 Padding 防形变 (针对原图极度狭窄的边缘情况)
            if patch.shape[0] < window_size or patch.shape[1] < window_size:
                pad_h = window_size - patch.shape[0]
                pad_w = window_size - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

            # 2. 模拟 V07 物理放大 (强制 Resize 到 640)
            resized_patch = cv2.resize(patch, (640, 640), interpolation=cv2.INTER_CUBIC)

            # 3. 推理 (开启极低置信度施压，放过所有可能目标，交由 NMS 处理)
            # verbose=False 防止控制台被几百个窗口的日志淹没
            results = model.predict(source=resized_patch, imgsz=640, conf=conf_thresh, verbose=False)
            boxes = results[0].boxes

            patch_count += 1

            # 4. 坐标映射还原
            for box in boxes:
                # 获取 640 尺度下的坐标
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].item()
                cls_id = int(box.cls[0].item())

                # 核心解码：除以 2.5 缩小回真实物理尺度，并加上局部窗口在原图的绝对坐标
                orig_x1 = (bx1 / scale_factor) + x1
                orig_y1 = (by1 / scale_factor) + y1
                orig_x2 = (bx2 / scale_factor) + x1
                orig_y2 = (by2 / scale_factor) + y1

                all_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                all_scores.append(score)
                all_classes.append(cls_id)

    print(f"扫描完毕！共切割 {patch_count} 个微距窗口，累计捕获 {len(all_boxes)} 个原始预测框。")

    if len(all_boxes) == 0:
        print("未检测到任何目标。")
        return img, [], [], []

    # ==========================================
    # 终极拦截：Batched NMS (按类别独立结算)
    # 彻底防止“大指”的框吞噬内部的“一”的框
    # ==========================================
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    classes_tensor = torch.tensor(all_classes, dtype=torch.float32)

    # NMS IoU 阈值：古籍排版紧密，适当调低防止同类误杀
    nms_iou_threshold = 0.45

    print("正在执行 Batched NMS (按类别的非极大值抑制) 以剔除重叠幻觉框...")
    keep_indices = torchvision.ops.batched_nms(boxes_tensor, scores_tensor, classes_tensor, nms_iou_threshold)

    final_boxes = boxes_tensor[keep_indices].numpy()
    final_scores = scores_tensor[keep_indices].numpy()
    final_classes = classes_tensor[keep_indices].numpy()

    print(f"去重完成！最终保留 {len(final_boxes)} 个结构化部件。")

    # 5. 渲染中文边界框
    class_names = model.names
    rendered_img = draw_chinese_boxes(img.copy(), final_boxes, final_classes, final_scores, class_names)

    return rendered_img, final_boxes, final_classes, final_scores


if __name__ == "__main__":
    # --- 1. 配置路径 ---
    WEIGHTS_PATH = r"F:\AIcharacter\SmallProject\runs_yolo11\y11_v07_hybrid_s_finetuned\weights\best.pt"
    TEST_IMAGE_PATH = r"F:\AIcharacter\SmallProject\RawScans\神奇秘谱\神奇秘谱图片\神奇秘谱3-逐页转图片\神奇秘谱3-逐页转图片-00076.jpg"
    OUTPUT_SAHI_PATH = r"F:\AIcharacter\SmallProject\test_out\v07_SAHI_全页重建结果2-4.jpg"

    # --- 2. 加载模型 ---
    print(f"正在加载 V07 权重: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    # --- 3. 读取大图 ---
    try:
        img = read_image_safe(TEST_IMAGE_PATH)
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        exit(1)

    # --- 4. 启动端到端 SAHI 推理流 ---
    # conf_thresh=0.08 用于捕获你刚才提到的低置信度正确部件，交由后续模块过滤
    rendered_img, boxes, classes, scores = sahi_sliding_window_inference(
        model, img,
        window_size=256,
        stride=128,
        conf_thresh=0.2
    )

    # --- 5. 落盘验证 ---
    save_image_safe(rendered_img, OUTPUT_SAHI_PATH)

    # 模拟模块 B 的前置数据准备：打印出所有部件坐标，这正是你生成 recon.json 所需的数据！
    print("\n--- 预备进入 模块 B (空间拓扑层) 的数据结构展示 ---")
    for i in range(min(5, len(boxes))):  # 仅打印前5个做示例
        cls_name = model.names[int(classes[i])]
        print(f"部件: {cls_name:<4} | 置信度: {scores[i]:.2f} | 物理坐标 (x1, y1, x2, y2): {boxes[i].astype(int)}")
    print("... (其余部件数据已省略)")