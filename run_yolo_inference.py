import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ================= 1. 加载参数矩阵 =================
# 载入你刚刚训练出的具有 8.1 GFLOPs 算力的最佳权重
WEIGHTS_PATH = os.path.join(BASE_DIR, "SAR_Dataset", "runs", "detect", "train5", "weights", "best.pt")
model = YOLO(WEIGHTS_PATH)

# ================= 2. 执行前向传播 =================
# 【前辈注意】：请把这里替换为你想要测试的一张 SAR 幅度图路径 (可以先拿验证集里的图试试)
TEST_IMAGE = os.path.join(BASE_DIR, "SAR_Dataset", "images", "val", "rigui_257.png")

print(f"正在对 {TEST_IMAGE} 执行空间异常检测...")

# 执行推断，强行要求系统输出归一化坐标 TXT 文件
results = model.predict(
    source=TEST_IMAGE,
    conf=0.25,        # 置信度阈值
    save_txt=True,    # 核心指令：输出 txt 坐标
    save_conf=False,  # 不需要保存置信度数值到 txt 里
    project=BASE_DIR,  # 根目录
    name='yolo_outputs',          # 输出文件夹名称
    exist_ok=True                 # 允许覆盖写入
)

print("\n--- [视觉先验提取完成] ---")
print(f"目标的归一化空间坐标已成功导出至: {os.path.join(BASE_DIR, 'yolo_outputs', 'labels')} 目录！")