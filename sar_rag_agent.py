import h5py
import numpy as np
import json
import re
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# ================= 0. 配置区 =================
YOLO_TXT_PATH = os.path.join(os.path.dirname(__file__), "yolo_outputs", "labels","rigui_257.txt")
MAT_FILE_PATH = os.path.join(os.path.dirname(__file__), "mat", "rigui_001.mat")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
API_KEY = "***REMOVED_API_KEY***" # 请前辈务必替换为你的 yunwu.ai API Key
BASE_URL = "https://yunwu.ai/v1"

# ================= 1. 物理层：数据解析与特征提取 =================
# （这里直接复用前辈已经完美跑通的物理计算代码）
def parse_yolo_txt(txt_path, img_width=2048, img_height=2048):
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    x_c, y_c, w, h = map(float, parts[1:5])
    return int(x_c * img_width), int(y_c * img_height), int(w * img_width), int(h * img_height)

def load_sar_complex(file_path):
    with h5py.File(file_path, 'r') as f:
        data_struct = f['SAR_Data_Complex']
        real_part = data_struct['complex_matrix']['real'][:]
        imag_part = data_struct['complex_matrix']['imag'][:]
        return (real_part + 1j * imag_part).T

def adaptive_peak_search_and_extract(complex_mat, init_y, init_x, search_window=128, feature_window=64):
    h_mat, w_mat = complex_mat.shape
    half_search = search_window // 2
    
    y_s, y_e = max(0, init_y - half_search), min(h_mat, init_y + half_search)
    x_s, x_e = max(0, init_x - half_search), min(w_mat, init_x + half_search)
    
    search_roi = complex_mat[y_s:y_e, x_s:x_e]
    local_y, local_x = np.unravel_index(np.argmax(np.abs(search_roi)), search_roi.shape)
    peak_y, peak_x = y_s + local_y, x_s + local_x
    
    half_feat = feature_window // 2
    fy_s, fy_e = max(0, peak_y - half_feat), min(h_mat, peak_y + half_feat)
    fx_s, fx_e = max(0, peak_x - half_feat), min(w_mat, peak_x + half_feat)
    
    feature_roi = complex_mat[fy_s:fy_e, fx_s:fx_e]
    amplitude, phase = np.abs(feature_roi), np.angle(feature_roi)
    
    fh, fw = amplitude.shape
    center_mask = np.zeros((fh, fw), dtype=bool)
    center_mask[fh//4 : 3*fh//4, fw//4 : 3*fw//4] = True
    
    P_t = np.mean(amplitude[center_mask]**2) + 1e-9 
    P_c = np.mean(amplitude[~center_mask]**2) + 1e-9 
    
    return peak_y, peak_x, 10 * np.log10(P_t / P_c), np.var(phase[center_mask])

# ================= 2. 认知层：RAG 知识检索与 LLM 推断 =================
def retrieve_physical_knowledge(query_text):
    """从本地 Chroma 数据库中检索相关文献"""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    # k=3 表示提取与 query 最相关的 3 个物理文献段落
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query_text)
    
    # 将检索到的高维语义块拼接为纯文本上下文
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context

def call_rag_agent(observation_data, rag_context):
    """基于 RAG 注入的大语言模型推理引擎"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    system_prompt = f"""
    你是一位顶级的太赫兹 SAR 信号处理专家。你的任务是：基于输入的电磁特征观测值，结合以下检索到的学术文献，进行多材质目标推断。
    
    【检索到的学术文献先验知识 (RAG Context)】：
    {rag_context}
    
    【你的推理守则】：
    1. 你必须将观测值与文献中的理论值进行严格的比对。
    2. 如果文献中提到了特定材质（如非金属、粗糙表面、金属）对应的信杂比或相位方差阈值，请以此作为分类的绝对判据。
    
    必须且只能输出以下严格的 JSON 格式：
    {{
      "classification": "具体的目标材质分类",
      "confidence": "数值%",
      "reasoning": "引用文献内容并结合观测数据的严谨数学推理过程"
    }}
    """

    user_prompt = f"当前 ROI 物理特征观测值：\n{json.dumps(observation_data, indent=2)}\n执行判定逻辑。"

    response = client.chat.completions.create(
        model="deepseek-chat", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    raw_content = response.choices[0].message.content
    cleaned_content = re.sub(r'^```json\s*', '', raw_content)
    cleaned_content = re.sub(r'^```\s*', '', cleaned_content)
    cleaned_content = re.sub(r'\s*```$', '', cleaned_content).strip()

    return json.loads(cleaned_content)

# ================= 主控制流 =================
if __name__ == "__main__":
    print("--- [Autonomous SAR-RAG Pipeline Initiated] ---")
    
    # 1. 物理计算层
    init_x, init_y, w_pixel, h_pixel = parse_yolo_txt(YOLO_TXT_PATH)
    mat_data = load_sar_complex(MAT_FILE_PATH)
    
    dyn_search, dyn_feature = max(w_pixel, h_pixel) + 100, max(w_pixel, h_pixel) + 50
    real_y, real_x, scr, phase_var = adaptive_peak_search_and_extract(mat_data, init_y, init_x, dyn_search, dyn_feature)
    
    report = {
        "signal_to_clutter_ratio_dB": round(scr, 2),
        "phase_variance_rad2": round(phase_var, 4)
    }
    print(f"[Stage 1] 底层物理特征提取完毕: SCR={report['signal_to_clutter_ratio_dB']} dB, PhaseVar={report['phase_variance_rad2']} rad^2")
    
    # 2. RAG 知识检索层
    print("[Stage 2] 正在向本地向量库发起物理特性检索...")
    query = "雷达目标识别，非金属、石质或混凝土目标的信杂比特性与漫散射相位方差特性。"
    retrieved_knowledge = retrieve_physical_knowledge(query)
    print("         -> 知识检索完成，已提取相关文献段落。")
    
    # 3. 语义推理层
    print("[Stage 3] 注入文献先验知识，大模型正在进行联合物理推理...")
    try:
        final_result = call_rag_agent(report, retrieved_knowledge)
        print("\n=== [SAR-RAG Agent 推断报告] ===")
        print(f"🎯 类别判定: {final_result.get('classification', 'N/A')}")
        print(f"📈 置信度:   {final_result.get('confidence', 'N/A')}")
        print(f"🧠 推断逻辑: {final_result.get('reasoning', 'N/A')}")
        print("================================")
    except Exception as e:
        print(f"推理发生异常: {e}")