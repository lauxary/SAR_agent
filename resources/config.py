import os

# ================= 0. 全局动态路径与架构配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 静态拓扑路径定义
YOLO_DATA_DIR = os.path.join(BASE_DIR, "yolo_data")

PNG_BACKUP_DIR = os.path.join(BASE_DIR, "assets", "png") 

MAT_SOURCE_DIR = os.path.join(BASE_DIR, "assets", "mat")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "SAR_Dataset", "runs", "detect", "train5", "weights", "best.pt")
OUTPUT_CSV = os.path.join(BASE_DIR, "out", "csv", "final_experiment_results.csv")

# ================= 1. 系统环境变量安全注入 =================
API_KEY = os.environ.get("MY_API_KEY")
if not API_KEY:
    raise ValueError("[Security Error] 系统环境变量 MY_API_KEY 未设置，进程终止。")

BASE_URL = os.environ.get("MY_BASE_URL", "https://api.openai.com/v1")  