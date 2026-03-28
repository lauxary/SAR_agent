import json
import re
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from resources.config import CHROMA_DB_PATH, API_KEY, BASE_URL

class SARRagAgent:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        self.vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def retrieve_context(self, query_text):
        docs = self.retriever.invoke(query_text)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    def predict_material(self, observation_data, rag_context):
        system_prompt = f"""
        基于输入的电磁特征观测值，结合以下检索到的学术文献，进行多材质目标推断。
        【RAG Context】:\n{rag_context}\n
        必须且只能输出严格的 JSON 格式：{{"classification": "...", "confidence": "...", "reasoning": "..."}}
        """
        user_prompt = f"当前 ROI 物理特征观测值：\n{json.dumps(observation_data)}\n执行判定逻辑。"
        
        response = self.client.chat.completions.create(
            model="deepseek-chat", 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0
        )
        raw_content = response.choices[0].message.content
        cleaned_content = re.sub(r'^```json\s*', '', raw_content)
        cleaned_content = re.sub(r'^```\s*', '', cleaned_content).strip()
        return json.loads(cleaned_content)