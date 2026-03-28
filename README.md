# SAR_agent

一个面向太赫兹 SAR 场景的实验项目，融合了：

- 复数域物理特征提取（SCR、相位方差）
- YOLO 视觉先验定位
- RAG 文献检索增强
- LLM 语义推理判别

本 README 目标是让新同学可以从 0 到 1 跑通完整流程。

## 1. 项目功能概览

本项目包含三条主要链路：

1. 数据准备链路：`mat_to_png.py`
2. 单样本推理链路：`run_yolo_inference.py` + `sar_rag_agent.py`（或 `sar_pipeline_main.py`）
3. 批量自动化链路：`main_batch_pipeline.py`

其中 RAG 向量库由 `build_rag_db.py` 预先构建。

## 2. 目录说明

建议重点关注以下目录/文件：

- `mat/`：原始 `.mat` 复数数据
- `png/`：`mat_to_png.py` 生成的 PNG 幅度图
- `SAR_Dataset/`：YOLO 训练与验证数据集
- `knowlodge/`：RAG 使用的论文 PDF
- `chroma_db/`：RAG 向量数据库落盘目录
- `yolo_outputs/`：YOLO 推理后输出的标签与可视化结果
- `requirements.txt`：Python 依赖

核心脚本：

- `mat_to_png.py`：`.mat -> .png`
- `SAR_Dataset/train_yolo.py`：训练 YOLO
- `run_yolo_inference.py`：YOLO 单图推理并导出标签
- `build_rag_db.py`：构建/刷新 Chroma 向量库
- `sar_rag_agent.py`：YOLO + 物理特征 + RAG + LLM 单样本推理
- `main_batch_pipeline.py`：批量样本自动评估并导出 CSV

## 3. 环境准备

### 3.1 Python 版本

推荐：Python 3.10 或 3.11。

### 3.2 创建并激活虚拟环境（可选但推荐）

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

WSL / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.3 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 必要配置

### 4.1 API Key 配置

当前脚本中使用 `OpenAI(api_key=..., base_url="https://yunwu.ai/v1")`。

你需要在以下脚本中把 API Key 改为自己的可用 Key：

- `sar_rag_agent.py`
- `sar_pipeline_main.py`
- `main_batch_pipeline.py`

### 4.2 模型可用性说明

如果报错类似“模型无可用渠道（distributor）”，说明该模型在当前平台暂不可用。

建议先使用：

- `deepseek-chat`

## 5. 快速开始（推荐顺序）

### Step 1：将 MAT 转成 PNG

```bash
python mat_to_png.py
```

默认读取：`mat/rigui_001.mat`

默认输出：`png/rigui_001.png`

### Step 2（可选）：训练 YOLO

进入数据集目录并训练：

```bash
cd SAR_Dataset
python train_yolo.py
```

训练配置由 `sar_prior.yaml` 提供（当前为相对路径配置）。

### Step 3：YOLO 单图推理，导出标签

回到项目根目录执行：

```bash
python run_yolo_inference.py
```

输出标签位于：`yolo_outputs/labels/`

### Step 4：构建 RAG 向量库

```bash
python build_rag_db.py
```

默认读取论文：`knowlodge/AD0859917.pdf`

默认向量库目录：`chroma_db/`

### Step 5：运行单样本 SAR-RAG 推理

```bash
python sar_rag_agent.py
```

你将看到：

- SCR 与相位方差
- 文献检索状态
- LLM 分类、置信度与推理解释

## 6. 批量推理（自动化）

直接运行：

```bash
python main_batch_pipeline.py
```

脚本会：

1. 加载 YOLO 权重
2. 遍历 `SAR_Dataset/images/val/*.png`
3. 匹配 `mat/` 下同名 `.mat`
4. 计算物理特征并调用 RAG + LLM
5. 导出 `final_experiment_results.csv`

## 7. 数据命名与匹配规则

批量流程按“同名匹配”关联 PNG 与 MAT：

- 例如 `SAR_Dataset/images/val/rigui_257.png`
- 对应 `mat/rigui_257.mat`

若同名 MAT 不存在，样本会被跳过。

## 8. 常见问题

### Q1：`Authentication failed` 无法 push 到 GitHub

- 原因：HTTPS 密码鉴权已禁用
- 方案：改用 SSH 或 PAT

### Q2：模型报 `503` / `无可用渠道`

- 原因：当前平台不支持该模型
- 方案：切回 `deepseek-chat` 或改用平台支持的模型 ID

### Q3：找不到文件（路径问题）

- 先确认在项目根目录执行脚本
- 本项目大部分脚本已改为相对路径 + `os.path.dirname(__file__)`

### Q4：RAG 检索报错（Chroma / Embedding）

- 先执行 `python build_rag_db.py`
- 再执行推理脚本

## 9. 建议的最小可复现实验

从空环境开始，按以下命令顺序：

```bash
pip install -r requirements.txt
python mat_to_png.py
python build_rag_db.py
python run_yolo_inference.py
python sar_rag_agent.py
```

## 10. 安全提示

- 不要把真实 API Key 提交到 Git 仓库
- 建议改用环境变量管理密钥
- 若密钥泄露，请立即在平台侧吊销并重建

---

如需团队协作版 README（包含实验记录模板、结果对比表、参数追踪规范），可以在此基础上继续扩展。
