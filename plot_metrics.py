import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ================= 0. 环境与路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results", "final_experiment_results.csv")
OUTPUT_CM_PATH = os.path.join(BASE_DIR, "results", "confusion_matrix_IEEE.png")

# 强制使用矢量化字体，满足 IEEE 学术期刊出版规范
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 数据摄取与特征空间映射 =================
print("正在读取物理推断实验数据...")
df = pd.read_csv(CSV_PATH)

# 生成物理真值 (Ground Truth): 根据文件名规则，rigui 为 Dielectric，否则为 Metal
df['Ground_Truth'] = df['Target_ID'].apply(lambda x: 'Dielectric' if 'rigui' in str(x).lower() else 'Metal')

# 大语言模型语义结果向离散二元空间的映射函数
def map_llm_classification(cls_str):
    cls_str = str(cls_str).lower()
    if '金属' in cls_str or 'metal' in cls_str:
        return 'Metal'
    return 'Dielectric'

df['Predicted_Class'] = df['LLM_Classification'].apply(map_llm_classification)
y_true = df['Ground_Truth']
y_pred = df['Predicted_Class']

# ================= 2. 统计学度量计算 =================
print("\n=== [系统分类效能报告 (Classification Report)] ===")
report = classification_report(y_true, y_pred, target_names=['Dielectric', 'Metal'], zero_division=0)
print(report)

# ================= 3. 混淆矩阵高分辨率渲染 =================
print("正在渲染高分辨率混淆矩阵...")
cm = confusion_matrix(y_true, y_pred, labels=['Dielectric', 'Metal'])

fig, ax = plt.subplots(figsize=(6, 5), dpi=300) # 300 DPI 满足学术出版硬性要求
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Dielectric', 'Metal'], 
            yticklabels=['Dielectric', 'Metal'],
            annot_kws={"size": 14, "weight": "bold"})

ax.set_xlabel('Predicted Physical Class', fontsize=12, fontweight='bold')
ax.set_ylabel('True Physical Class', fontsize=12, fontweight='bold')
ax.set_title('SAR-RAG Agent Confusion Matrix', fontsize=14, pad=15, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_CM_PATH, format='png', bbox_inches='tight')
print(f"--- [渲染完成] ---")
print(f"IEEE 格式混淆矩阵已成功导出至: {OUTPUT_CM_PATH}")