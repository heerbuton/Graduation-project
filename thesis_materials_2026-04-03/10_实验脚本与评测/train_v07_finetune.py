import os
from ultralytics import YOLO

# ==========================================
# ⚙️ 核心路径配置区
# ==========================================
# 1. 刚刚创建好的 V07 数据集 YAML 配置文件路径
DATA_YAML_PATH = r"F:\AIcharacter\SmallProject\DatasetBuild\V07_Hybrid\data.yaml"

# 2. 之前 V05 跑出来的成熟基座权重绝对路径
# (这里帮你按照你的输出目录习惯推导了路径，请确保该路径下有 best.pt)
PRETRAINED_WEIGHTS = r"F:\AIcharacter\SmallProject\runs_yolo11\y11_v05_hybrid_s_tuned\weights\best.pt"


def train_v07_hybrid():
    print("🔥 启动基于 Ultralytics API 的终极对抗微调引擎...")

    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ 找不到配置文件: {DATA_YAML_PATH}")
        return
    if not os.path.exists(PRETRAINED_WEIGHTS):
        print(f"❌ 找不到预训练权重: {PRETRAINED_WEIGHTS}")
        return

    # 1. 实例化模型并加载 V05 成熟权重
    print(f"📦 正在加载成熟基座权重: {PRETRAINED_WEIGHTS}")
    model = YOLO(PRETRAINED_WEIGHTS)

    # 2. 注入核心超参并启动训练
    print("⚙️ 正在注入核心超参，严守防崩红线与特征保护策略...")

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=50,  # 终极对抗微调 50 轮
        batch=8,  # 防 OOM 限制，控制在 4-16
        imgsz=640,
        workers=0,  # Windows 防崩红线
        device=0,

        # --- 核心特征保护与对抗策略 ---
        patience=0,  # 禁用早停
        mosaic=0.0,  # 强制关闭马赛克拼接
        scale=0.3,  # 大幅放开特征锁
        translate=0.1,  # 彻底摧毁尺度幻觉
        lr0=0.001,  # 下调初始学习率进行微调

        # --- 🎯 工程与输出规范 (已与你的历史习惯对齐) ---
        project=r"F:\AIcharacter\SmallProject\runs_yolo11",  # 统一的主输出目录
        name="y11_v07_hybrid_s_finetuned",  # 保持一致性的子文件夹命名
        exist_ok=True,
        resume=False  # 如果显存碎片化崩溃，改为 True 重启
    )

    print("🎉 V07 终极微调训练任务已顺利结束！")


if __name__ == '__main__':
    train_v07_hybrid()