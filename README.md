# SAR_agent

面向 SAR 目标识别的多模块实验项目，融合三类信息：

1. 视觉先验：YOLO 检测得到目标显著区域。
2. 物理特征：从复数 SAR 矩阵中提取 SCR、相位方差。
3. 认知推理：RAG 检索 + LLM 输出材质判断。

本 README 基于当前仓库真实结构编写，目标是让你快速跑通示例流程并理解关键模块。

## 1. 当前项目结构

核心目录：

- assets/mat：原始 MAT 数据。
- assets/png：MAT 转换后的 PNG 备份目录。
- yolo_data：主流程待处理 PNG 输入目录。
- knowledge：RAG 文献目录（当前包含 AD0859917.pdf）。
- chroma_db：向量数据库持久化目录。
- out/csv：批量推理结果 CSV 输出目录。
- src：核心模块代码。
- examples：主流程入口脚本。
- resources：项目配置与环境变量读取。

核心文件：

- examples/main.py：主入口（推荐）。
- src/vision_detector.py：YOLO 检测封装。
- src/physics_engine.py：复数矩阵解析与物理特征提取。
- src/cognitive_agent.py：RAG 检索 + LLM 推理。
- resources/config.py：统一路径、权重、API 配置。
- build_rag_db.py：构建 Chroma 向量库。
- SAR_Dataset/train_yolo.py：YOLO 训练脚本。
- plot_metrics.py：读取结果 CSV 并绘制混淆矩阵。

## 2. 环境准备

推荐 Python 版本：3.10 或 3.11。

安装依赖：

```bash
pip install -r requirements.txt
```

可选：若你想提前消除 Chroma 的 LangChain 弃用警告，可额外安装：

```bash
pip install -U langchain-chroma
```

## 3. 必要配置

主流程从 resources/config.py 读取环境变量。

必须设置：

- MY_API_KEY：你的模型服务密钥。

可选设置：

- MY_BASE_URL：默认是 https://api.openai.com/v1。

Linux/WSL 示例：

```bash
export MY_API_KEY="your_key"
export MY_BASE_URL="https://api.openai.com/v1"
```

PowerShell 示例：

```powershell
$env:MY_API_KEY="your_key"
$env:MY_BASE_URL="https://api.openai.com/v1"
```

## 4. 先决条件检查

在运行主流程前，请确认：

1. YOLO 权重文件存在于 resources/config.py 指定路径。
2. yolo_data 目录中有待处理 PNG 文件。
3. assets/mat 中有与 PNG 同名的 MAT 文件。
4. chroma_db 已构建（见下一节）。

## 5. 构建 RAG 向量库

执行：

```bash
python build_rag_db.py
```

注意：当前 build_rag_db.py 中 PDF 默认路径写为 knowlodge/AD0859917.pdf，而项目目录是 knowledge。

你可以选择其一：

1. 修改 build_rag_db.py 中的 PDF 路径为 knowledge/AD0859917.pdf。
2. 新建一个 knowlodge 目录并放入 PDF。

构建成功后，向量索引会写入 chroma_db。

## 6. 运行主流程

推荐在项目根目录执行：

```bash
python examples/main.py
```

流程说明：

1. 从 yolo_data 读取 PNG。
2. 用 YOLO 预测目标框。
3. 用同名 MAT 提取 SCR 与相位方差。
4. 调用 RAG + LLM 输出分类、置信度与解释。
5. 将结果写入 out/csv/final_experiment_results.csv。

日志输出：

- 控制台实时日志。
- 根目录 agent_run.log 持久化日志。

## 7. 训练 YOLO（可选）

执行：

```bash
python SAR_Dataset/train_yolo.py
```

当前脚本默认使用 yolov8n.pt，并读取 SAR_Dataset/sar_prior.yaml。

## 8. 指标绘图（可选）

执行：

```bash
python plot_metrics.py
```

默认读取 results/final_experiment_results.csv，并输出 results/confusion_matrix_IEEE.png。

如果你的主流程结果写在 out/csv，请先同步到 results 或修改脚本内 CSV_PATH。

## 9. 数据命名规则

主流程使用文件名同名匹配：

- yolo_data/rigui_001.png
- assets/mat/rigui_001.mat

若 MAT 缺失，对应样本会被跳过。

## 10. 常见问题

1. 报错 module 'resource' has no attribute getrlimit
原因通常是自定义目录名与 Python 标准库 resource 冲突。当前项目已使用 resources 目录规避。

2. 报错系统环境变量 MY_API_KEY 未设置
先按第 3 节设置环境变量后再运行。

3. Chroma 弃用警告
当前代码仍使用 langchain_community.vectorstores.Chroma，可运行但会提示弃用；后续可迁移到 langchain_chroma。

4. 运行时找不到权重
检查 resources/config.py 中 YOLO_WEIGHTS 对应路径是否存在。

## 11. 推荐运行顺序

```bash
pip install -r requirements.txt
python build_rag_db.py
python examples/main.py
```

如需完整实验链路，再执行训练与评估脚本。
