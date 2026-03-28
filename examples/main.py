import os
import sys
import glob
import logging
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from resources.config import YOLO_DATA_DIR, MAT_SOURCE_DIR, OUTPUT_CSV, BASE_DIR
from src.vision_detector import SARVisionDetector
from src.physics_engine import load_sar_complex, adaptive_peak_search_and_extract
from src.cognitive_agent import SARRagAgent

# ================= 0. 工业级日志系统初始化 =================
def setup_logger():
    # 创建全局记录器
    logger = logging.getLogger("SAR_Agent")
    logger.setLevel(logging.INFO)
    
    # 防止日志重复打印
    if not logger.handlers:
        # 处理器 1：磁盘日志文件 (记录所有细节，追加模式)
        log_file_path = os.path.join(BASE_DIR, "agent_run.log")
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s.[%(msecs)03d] | %(levelname)-8s | [%(module)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # 处理器 2：终端控制台输出 (简化版时间戳，保持清爽)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s | %(levelname)-6s | %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        
        # 挂载双路处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()

# ================= 1. 自动化流水线主循环 =================
def main():
    logger.info("="*50)
    logger.info("🚀 [SAR-RAG Pipeline] 模块化推理引擎启动")
    
    # 动态创建工作区目录（如果前辈还没建的话，系统会自动建好）
    os.makedirs(YOLO_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # 初始化解耦模块
    try:
        vision_module = SARVisionDetector()
        agent_module = SARRagAgent()
        logger.info("✔️ 视觉感知与大模型认知模块加载完毕")
    except Exception as e:
        logger.error(f"❌ 核心模块初始化失败: {e}")
        return
    
    logger.info("⏳ 正在预加载物理文献知识库...")
    global_context = agent_module.retrieve_context("雷达目标识别，非金属、石质目标的信杂比特性与漫散射相位方差特性。")
    logger.info("✔️ RAG 向量检索就绪")
    
    # 【核心修改 2】：扫描新的 yolo_data 文件夹
    png_files = glob.glob(os.path.join(YOLO_DATA_DIR, "*.png"))
    
    if not png_files:
        logger.warning(f"⚠️ 工作区 {YOLO_DATA_DIR} 为空。请将照片从 png/ 文件夹复制过来！")
        return
        
    logger.info(f"🎯 在工作区检测到 {len(png_files)} 个待处理目标。开始物理推断流水线...")
    
    results_log = []

    for img_path in png_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mat_file_path = os.path.join(MAT_SOURCE_DIR, f"{base_name}.mat")
        
        logger.info(f"▶️ [Processing] 目标 ID: {base_name}")
        
        if not os.path.exists(mat_file_path):
            logger.warning(f"  -> [Skip] 找不到底层复数矩阵: {mat_file_path}")
            continue

        # 1. 视觉抓取
        xywhn = vision_module.get_saliency_box(img_path)
        if xywhn is None:
            logger.warning(f"  -> [Vision] YOLO 未检测到显著空间异常。")
            continue
            
        x_c, y_c, w_n, h_n = xywhn
        init_x, init_y = int(x_c * 2048), int(y_c * 2048)
        w_p, h_p = int(w_n * 2048), int(h_n * 2048)
        
        # 2. 物理提取
        try:
            mat_data = load_sar_complex(mat_file_path)
            d_search, d_feat = max(w_p, h_p) + 100, max(w_p, h_p) + 50
            _, _, scr, phase_var = adaptive_peak_search_and_extract(mat_data, init_y, init_x, d_search, d_feat)
            
            report = {"SCR_dB": round(scr, 2), "PhaseVar_rad2": round(phase_var, 4)}
            logger.info(f"  -> [Physics] SCR={report['SCR_dB']}dB, Phase_Var={report['PhaseVar_rad2']}rad²")
        except Exception as e:
            logger.error(f"  -> [Physics Error] 复数矩阵解析崩溃: {e}")
            continue
        
        # 3. 认知推断
        try:
            llm_res = agent_module.predict_material(report, global_context)
            logger.info(f"  -> [LLM] 分类: {llm_res.get('classification')} | 置信度: {llm_res.get('confidence')}")
            
            results_log.append({
                "Target_ID": base_name,
                "SCR(dB)": report['SCR_dB'],
                "Phase_Variance": report['PhaseVar_rad2'],
                "LLM_Classification": llm_res.get('classification'),
                "Confidence": llm_res.get('confidence'),
                "Reasoning": llm_res.get('reasoning')
            })
        except Exception as e:
            logger.error(f"  -> [LLM Error] API 推理异常: {e}")

    # 4. 数据持久化
    if results_log:
        df = pd.DataFrame(results_log)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        logger.info(f"✅ [Success] 数据已导出至 CSV: {OUTPUT_CSV}")
    
    logger.info("🏁 物理推断流水线执行完毕。")
    logger.info("="*50)

if __name__ == "__main__":
    main()