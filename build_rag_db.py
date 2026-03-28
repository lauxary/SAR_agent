import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ================= 1. 数据摄取层 (Data Ingestion) =================
# 前辈请把这里的路径替换为你下载好的那篇（或多篇）顶级论文 PDF 的物理路径
PDF_FILE_PATH = os.path.join(os.path.dirname(__file__), "knowlodge", "AD0859917.pdf")
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db") # 本地向量数据库的持久化存储目录

print("[Stage 1] 启动 PDF 解析流，提取多模态文本特征...")
loader = PyMuPDFLoader(PDF_FILE_PATH)
docs = loader.load()
print(f"成功加载文献，共解析物理页面数量: {len(docs)} 页")

# ================= 2. 语义分块层 (Semantic Chunking) =================
# 数学原理：由于大语言模型存在严格的 Context Window (上下文窗口) 令牌限制，
# 且 Embedding 模型对超长文本的高维映射存在梯度衰减现象，必须进行文本切分。
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 每个特征块包含 500 个字符
    chunk_overlap=50,   # 设置 50 个字符的重叠度，防止跨段落的物理定理被强行截断，破坏语义连续性
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
splits = text_splitter.split_documents(docs)
print(f"[Stage 2] 文本滑窗切分完成，共生成高维语义块数量: {len(splits)} 个")

# ================= 3. 高维向量化与持久化层 (Embedding & Vectorization) =================
print("[Stage 3] 正在加载本地 Embedding 权重 (首次运行将自动从 HuggingFace 下载小型权重)...")
# 工程策略：这里不使用 API，而是调用轻量级的开源本地嵌入模型 BAAI/bge-small-zh-v1.5。
# 它的参数量极小，运行极快，且在双语检索任务上具有极高的余弦相似度区分度，完全不会修改其内部权重矩阵 \Theta。
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

print("[Stage 4] 正在计算文本向量投影并构建 Chroma 本地索引集...")
# 将离散文本映射为高维稠密向量，并写入本地磁盘 sqlite3 数据库
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory=PERSIST_DIRECTORY
)

print(f"--- [RAG 向量数据库构建完毕] ---")
print(f"数据库已物理持久化至: {PERSIST_DIRECTORY}")