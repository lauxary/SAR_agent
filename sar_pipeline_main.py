import h5py
import numpy as np
import json
from openai import OpenAI
import re
import os

# ================= 1. 物理层：数据解析与预处理 =================
def parse_yolo_txt(txt_path, img_width=2048, img_height=2048):
    """
    解析 YOLO 归一化坐标并映射为离散矩阵索引及像素尺寸，
    TODO:注意这里2048是根据实际图像尺寸设定的，后续需要动态适配不同尺寸的输入图像
    """
    with open(txt_path, 'r') as f:
        line = f.readline().strip()    # TODO: 只读取第一行，后续仍需要支持多行处理
    
    parts = line.split()
    class_id = int(parts[0])
    x_c, y_c, w, h = map(float, parts[1:5])
    
    # 线性映射公式推导矩阵全局索引与像素跨度
    global_x = int(x_c * img_width)
    global_y = int(y_c * img_height)
    w_pixel = int(w * img_width)
    h_pixel = int(h * img_height)
    
    return global_x, global_y, w_pixel, h_pixel

def load_sar_complex(file_path):
    """
    读取单视复数 (SLC) 矩阵 Z(m,n)
    SAR_Data_Complex = struct();
    SAR_Data_Complex.complex_matrix = Im1;      % 核心：复数矩阵
    SAR_Data_Complex.fc = fc;                   % 中心频率 (216GHz)
    SAR_Data_Complex.fs = Fs;                   % 采样率
    SAR_Data_Complex.v_mean = v_mean;           % 飞机速度
    SAR_Data_Complex.lambda = lambda;           % 波长
    SAR_Data_Complex.pixel_spacing_r = DeltaR;  % 距离向分辨率
    SAR_Data_Complex.n_azi_index = n1;          % 帧序号
    MATLAB采用列优先，这里需要转置以对齐坐标系
    """
    with h5py.File(file_path, 'r') as f:
        data_struct = f['SAR_Data_Complex']
        real_part = data_struct['complex_matrix']['real'][:]
        imag_part = data_struct['complex_matrix']['imag'][:]
        complex_mat = real_part + 1j * imag_part
        return complex_mat.T

# ================= 2. 算法层：自适应寻峰与特征提取 =================
def adaptive_peak_search_and_extract(complex_mat, init_y, init_x, search_window=128, feature_window=64):
    """
    自适应寻峰算法：在视觉几何中心附近搜索电磁强散射点，并提取精确特征
    """
    h_mat, w_mat = complex_mat.shape
    half_search = search_window // 2
    
    # 1. 划定初始搜索域 \Omega
    y_s = max(0, init_y - half_search)
    y_e = min(h_mat, init_y + half_search)
    x_s = max(0, init_x - half_search)
    x_e = min(w_mat, init_x + half_search)
    
    search_roi = complex_mat[y_s:y_e, x_s:x_e]
    amplitude_search = np.abs(search_roi)
    
    # 2. 求解二维寻峰极值点索引
    local_y, local_x = np.unravel_index(np.argmax(amplitude_search), amplitude_search.shape)
    
    # 3. 映射回全局矩阵坐标系得到真实的电磁相干中心
    peak_y = y_s + local_y
    peak_x = x_s + local_x
    
    # 4. 以真实极值点为中心，截取特征提取窗口
    half_feat = feature_window // 2
    fy_s = max(0, peak_y - half_feat)
    fy_e = min(h_mat, peak_y + half_feat)
    fx_s = max(0, peak_x - half_feat)
    fx_e = min(w_mat, peak_x + half_feat)
    
    feature_roi = complex_mat[fy_s:fy_e, fx_s:fx_e]
    
    # 5. 计算物理量：局部信杂比 (SCR) 与相位方差 (\sigma_{\phi}^2)
    amplitude = np.abs(feature_roi)
    phase = np.angle(feature_roi)
    
    fh, fw = amplitude.shape
    center_mask = np.zeros((fh, fw), dtype=bool)
    center_mask[fh//4 : 3*fh//4, fw//4 : 3*fw//4] = True
    # 在太赫兹 SAR 成像中，这种“中心 vs 邻域”的策略是检测算法（如 CFAR 检测）的核心思想。
    
    P_t = np.mean(amplitude[center_mask]**2) + 1e-9 
    P_c = np.mean(amplitude[~center_mask]**2) + 1e-9 
    scr_db = 10 * np.log10(P_t / P_c)
    
    phase_variance = np.var(phase[center_mask])
    
    return peak_y, peak_x, scr_db, phase_variance

# ================= 3. 认知层：大语言模型推断 =================
def call_sar_agent(observation_data):
    """基于先验物理规则构建的推理中枢"""
    client = OpenAI(
        api_key="***REMOVED_API_KEY***", 
        base_url="https://yunwu.ai/v1" 
    )

    system_prompt = """
    你是一位太赫兹 SAR 信号处理系统。任务：基于输入的特征，进行多材质目标推断。
    物理准则：
    1. 刚性金属体：SCR > 5.0 dB，相位方差 < 2.0 rad^2（强相干散射）。
    2. 大型石质/混凝土介质目标（如日晷、建筑）：表面漫散射导致相位高度随机，相位方差通常在 3.0 ~ 3.3 rad^2 之间，但在大尺度观测下，其整体后向散射系数显著高于周边地物，SCR 通常在 2.0 dB ~ 15.0 dB 之间。
    3. 纯地物杂波：SCR < 1.0 dB，相位方差 \approx 3.28 rad^2。
    
    必须且只能输出以下严格的 JSON 格式，不要包含任何 Markdown 符号或代码块标记：
    {
      "classification": "杂波" 或 "石质/介质目标" 或 "金属强目标",
      "confidence": "数值%",
      "reasoning": "结合 SCR 与方差数值的严谨数学机理分析"
    }
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

    # 强大的正则表达式清洗逻辑：去除可能存在的 ```json 和 ``` 标记
    cleaned_content = re.sub(r'^```json\s*', '', raw_content)
    cleaned_content = re.sub(r'^```\s*', '', cleaned_content)
    cleaned_content = re.sub(r'\s*```$', '', cleaned_content)
    cleaned_content = cleaned_content.strip()

    return json.loads(cleaned_content)

# ================= 主控制流 =================
if __name__ == "__main__":
    YOLO_TXT_PATH = os.path.join(os.path.dirname(__file__),"yolo_prediction", "yolo_prediction.txt")
    MAT_FILE_PATH = os.path.join(os.path.dirname(__file__), "mat", "rigui_001.mat")
    
    print("--- [SAR Target Recognition Pipeline Initialized] ---")
    
    # Step 1: 解析视觉几何坐标与像素尺寸
    init_x, init_y, w_pixel, h_pixel = parse_yolo_txt(YOLO_TXT_PATH)
    print(f"[Stage 1] 映射完成: 坐标 (X:{init_x}, Y:{init_y}), 尺寸 {w_pixel}x{h_pixel}")
    
    # Step 2: 加载复数数据
    mat_data = load_sar_complex(MAT_FILE_PATH)
    print(f"[Stage 2] 原始复数矩阵载入完成，维度: {mat_data.shape}")
    
    # 动态计算搜索域与特征提取窗口的标量边界
    dynamic_search_window = max(w_pixel, h_pixel) + 100
    dynamic_feature_window = max(w_pixel, h_pixel) + 50
    
    # Step 3: 自适应寻峰与精确特征提取
    real_y, real_x, scr, phase_var = adaptive_peak_search_and_extract(
        mat_data, init_y, init_x, 
        search_window=dynamic_search_window, 
        feature_window=dynamic_feature_window
    )
    print(f"[Stage 3] 二维空间寻峰完成。")
    print(f"          -> 坐标位移修正: (X:{init_x}, Y:{init_y}) -> (X:{real_x}, Y:{real_y})")
    
    report = {
        "calibrated_coordinate": [int(real_y), int(real_x)],
        "signal_to_clutter_ratio_dB": round(scr, 2),
        "phase_variance_rad2": round(phase_var, 4)
    }
    
    print(f"[Stage 4] 局部物理特征计算完成: SCR={report['signal_to_clutter_ratio_dB']} dB, PhaseVar={report['phase_variance_rad2']} rad^2")
    print("[Stage 5] 请求大语言模型进行电磁机理推断...")
    
    # Step 4: 执行判定流
    try:
        final_result = call_sar_agent(report)
        print("\n=== [LLM 推断结论] ===")
        # 使用 .get() 方法，即使键名不完全匹配也不会让程序崩溃
        print(f"类别判定: {final_result.get('classification', '解析失败，请查看原始输出')}")
        print(f"置信度:   {final_result.get('confidence', '未知')}")
        print(f"推断逻辑: {final_result.get('reasoning', final_result)}")
        print("======================")
    except Exception as e:
        print(f"语言模型接口调用发生异常或 JSON 解析失败: {e}")