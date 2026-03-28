import os
import glob
import json
import numpy as np
import h5py
import pandas as pd
from ultralytics import YOLO
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re

# ================= 0. 全局动态路径与架构配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 【前辈只需在这里配置一次物理路径即可】
PNG_VAL_DIR = os.path.join(BASE_DIR, "SAR_Dataset", "images", "val")  # 存放 YOLO 预测用的 64 张 PNG 图
MAT_SOURCE_DIR = os.path.join(BASE_DIR, "mat")      # 前辈存放所有原始 .mat 文件的物理目录
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "SAR_Dataset", "runs", "detect", "train5", "weights", "best.pt")

API_KEY = "***REMOVED_API_KEY***"  # 请替换为真实 Key
BASE_URL = "https://yunwu.ai/v1"

# ================= 1. 物理计算模块 (保持绝对严谨) =================
def load_sar_complex(file_path):
    with h5py.File(file_path, 'r') as f:
        data_struct = f['SAR_Data_Complex']
        real_part = data_struct['complex_matrix']['real'][:]
        imag_part = data_struct['complex_matrix']['imag'][:]
        return (real_part + 1j * imag_part).T

def adaptive_peak_search_and_extract(complex_mat, init_y, init_x, search_window=128, feature_window=64):
    h_mat, w_mat = complex_mat.shape
    y_s, y_e = max(0, init_y - search_window//2), min(h_mat, init_y + search_window//2)
    x_s, x_e = max(0, init_x - search_window//2), min(w_mat, init_x + search_window//2)
    
    search_roi = complex_mat[y_s:y_e, x_s:x_e]
    local_y, local_x = np.unravel_index(np.argmax(np.abs(search_roi)), search_roi.shape)
    peak_y, peak_x = y_s + local_y, x_s + local_x
    
    fy_s, fy_e = max(0, peak_y - feature_window//2), min(h_mat, peak_y + feature_window//2)
    fx_s, fx_e = max(0, peak_x - feature_window//2), min(w_mat, peak_x + feature_window//2)
    
    feature_roi = complex_mat[fy_s:fy_e, fx_s:fx_e]
    amplitude, phase = np.abs(feature_roi), np.angle(feature_roi)
    
    fh, fw = amplitude.shape
    center_mask = np.zeros((fh, fw), dtype=bool)
    center_mask[fh//4 : 3*fh//4, fw//4 : 3*fw//4] = True
    
    P_t = np.mean(amplitude[center_mask]**2) + 1e-9 
    P_c = np.mean(amplitude[~center_mask]**2) + 1e-9 
    return peak_y, peak_x, 10 * np.log10(P_t / P_c), np.var(phase[center_mask])

# ================= 2. 认知推理模块 =================
def retrieve_physical_knowledge(query_text):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query_text)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def call_rag_agent(observation_data, rag_context):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    system_prompt = f"""
    基于输入的电磁特征观测值，结合以下检索到的学术文献，进行多材质目标推断。
    【RAG Context】:\n{rag_context}\n
    必须且只能输出严格的 JSON 格式：{{"classification": "...", "confidence": "...", "reasoning": "..."}}
    """
    user_prompt = f"当前 ROI 物理特征观测值：\n{json.dumps(observation_data)}\n执行判定逻辑。"
    
    response = client.chat.completions.create(
        model="deepseek-chat", 
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.0
    )
    raw_content = response.choices[0].message.content
    cleaned_content = re.sub(r'^```json\s*', '', raw_content)
    cleaned_content = re.sub(r'^```\s*', '', cleaned_content).strip()
    return json.loads(cleaned_content)

# ================= 3. 自动化批处理主控引擎 =================
if __name__ == "__main__":
    print("--- [Automated Batch Processing Engine Initiated] ---")
    
    # 1. 挂载视觉模型并在 GPU 上待命
    yolo_model = YOLO(YOLO_WEIGHTS)
    
    # 2. 提前加载文献上下文 (避免在循环中重复检索消耗算力)
    print("正在初始化 RAG 物理知识库...")
    global_rag_context = retrieve_physical_knowledge("雷达目标识别，非金属、石质目标的信杂比特性与漫散射相位方差特性。")
    
    # 获取所有待测 PNG 列表
    png_files = glob.glob(os.path.join(PNG_VAL_DIR, "*.png"))
    results_log = []

    for img_path in png_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mat_file_path = os.path.join(MAT_SOURCE_DIR, f"{base_name}.mat")
        
        print(f"\n[Processing] {base_name} ...")
        
        # 安全性校验：确保对应的 .mat 文件存在
        if not os.path.exists(mat_file_path):
            print(f"  -> [跳过] 找不到对应的复数矩阵文件: {mat_file_path}")
            continue

        # 阶段 A：视觉空间显著性抓取 (内存张量直传)
        yolo_results = yolo_model.predict(source=img_path, conf=0.25, verbose=False)
        boxes = yolo_results[0].boxes
        
        if len(boxes) == 0:
            print("  -> [结果] YOLO 未在当前区域检测到显著目标。")
            continue
            
        # 提取置信度最高的目标的归一化坐标张量 [x_center, y_center, width, height]
        xywhn = boxes.xywhn[0].cpu().numpy() 
        x_c, y_c, w_n, h_n = xywhn
        
        init_x, init_y = int(x_c * 2048), int(y_c * 2048)
        w_pixel, h_pixel = int(w_n * 2048), int(h_n * 2048)
        
        # 阶段 B：底层的复数域特征提取
        mat_data = load_sar_complex(mat_file_path)
        dyn_search, dyn_feature = max(w_pixel, h_pixel) + 100, max(w_pixel, h_pixel) + 50
        real_y, real_x, scr, phase_var = adaptive_peak_search_and_extract(mat_data, init_y, init_x, dyn_search, dyn_feature)
        
        report = {
            "SCR_dB": round(scr, 2),
            "PhaseVar_rad2": round(phase_var, 4)
        }
        print(f"  -> [物理提取] SCR={report['SCR_dB']}dB, PhaseVar={report['PhaseVar_rad2']}rad^2")
        
        # 阶段 C：语义推理
        try:
            llm_result = call_rag_agent(report, global_rag_context)
            print(f"  -> [LLM 判定] {llm_result.get('classification')} (置信度: {llm_result.get('confidence')})")
            
            # 将多维数据压入日志序列
            results_log.append({
                "Target_ID": base_name,
                "SCR(dB)": report['SCR_dB'],
                "Phase_Variance": report['PhaseVar_rad2'],
                "LLM_Classification": llm_result.get('classification'),
                "Confidence": llm_result.get('confidence'),
                "Reasoning": llm_result.get('reasoning')
            })
        except Exception as e:
            print(f"  -> [异常] 推理失败: {e}")

    # ================= 4. 持久化输出实验结果表 =================
    if results_log:
        output_csv = os.path.join(BASE_DIR, "final_experiment_results.csv")
        df = pd.DataFrame(results_log)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n=== [批量评估完成] ===")
        print(f"实验数据汇总表已成功导出至: {output_csv}")