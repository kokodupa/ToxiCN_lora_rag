import json
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import jieba
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForSeq2Seq,
    pipeline,
    AutoModelForSequenceClassification
)
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(os.path.dirname(current_dir))     # 项目根目录
sys.path.insert(0, project_root)

from config.config import RAG_DATA_PATH, INSTRUCTION, SYSTEM_MESSAGE, TRAIN_DATA_PATH, TEST_DATA_PATH
from src.RAG.get_rag_prompt import rag_selftrain_prompt
_documents = None          # 缓存的词条列表
_bm25_index = None
_bm25_corpus = None
_embedding_model = None  


shibing_path = os.path.join(project_root, "model","shibing624/text2vec-base-chinese")
train_data_path = os.path.join(project_root, TRAIN_DATA_PATH)
test_data_path = os.path.join(project_root, TEST_DATA_PATH)

from modelscope import snapshot_download
model_dir = snapshot_download('Jerry0/text2vec-base-chinese', cache_dir='/mnt/workspace/Graduate/ToxiCN_lora_rag/model/shibing624')
print(model_dir)

from huggingface_hub import snapshot_download
import os

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
local_dir = os.path.join(project_root, "model", 'bge-reranker-base')
rerank_model_dir = snapshot_download(
    repo_id="BAAI/bge-reranker-base",
    local_dir=local_dir,      # 本地保存目录，可根据需要修改
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com"
)
print(rerank_model_dir)


import json
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def get_embedding_model():
    """懒加载 embedding 模型（避免重复加载）"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_dir)
    return _embedding_model

def load_documents():
    global _documents
    if _documents is not None:
        return _documents
    data = load_json(train_data_path)
    texts = [item['content'] for item in data]
    _documents = texts
    
    return _documents


def build_vector_database(train_data_path: str, batch_size: int = 32):
    # 加载训练数据
    data = load_json(train_data_path)  # 使用你已有的 load_json 函数
    persist_dir = os.path.join(project_root, "model/chroma_db")
    os.makedirs(persist_dir, exist_ok=True)

    embedding_model = get_embedding_model()

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    # 定义集合名称
    collection_name = "train_content"

    # 如果集合已存在则删除（可选，根据是否需要重建决定）
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # 余弦相似度
    )
    ids = []
    documents = []
    metadatas = []
    
    for item in data:
        ids.append(str(item["id"]))
        documents.append(item["content"])
        # 存储额外信息，例如 output，以便检索时连带返回
        metadatas.append({
            "content": item["content"],
            "output": item["output"]})
    
    # 分批添加，避免内存或请求限制
    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_metadatas = metadatas[start:end]
        
        # 生成向量（embeddings）
        batch_embeddings = embedding_model.encode(batch_docs, convert_to_numpy=True).tolist()
        
        # 添加到 collection
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metadatas
        )
        print(f"已添加 {end}/{total} 条数据")
    
    print("向量数据库构建完成！")
    return collection

def retriever(query, collection, top_k=2):
    # 获取嵌入模型（假设已有）
    embedding_model = get_embedding_model()
    
    # 编码查询为向量
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    
    # 执行检索（余弦距离自动生效）
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # 提取文档内容
    retrieved_texts = results['documents'][0] 
    retrived_meta = results['metadatas'][0]
    # 如果需要余弦相似度（而非距离）
    cosine_similarities = [1 - d for d in results['distances'][0]]
    
    return retrived_meta


def rerank(query, candidates, reranker, top_k=3):
    pass


collection = build_vector_database(train_data_path)
train_data = load_json(train_data_path)
# train_data = train_data[:10]
rag_self_train_data = []
for item in train_data:
    query = item["content"]
    candidates = retriever(query, collection, top_k=2)[1]
    # print(candidates)
    rag_self_train_data.append({
        'id': item["id"],
        "content": query,
        "candidates": candidates,
        "output": item["output"],
        "prompt": rag_selftrain_prompt(query, candidates)
    })

self_train_path = os.path.join(project_root, "data", "rag_self_train.json")
with open(self_train_path, "w", encoding="utf-8") as f:
    json.dump(rag_self_train_data, f, ensure_ascii=False, indent=4)

rag_self_test_data = []
test_data = load_json(test_data_path)
for item in test_data:
    query = item["content"]
    candidates = retriever(query, collection, top_k=2)[1]
    rag_self_test_data.append({
        'id': item["id"],
        "content": query,
        "candidates": candidates,
        "output": item["output"],
        "prompt": rag_selftrain_prompt(query, candidates)
    })

test_path = os.path.join(project_root, "data", "rag_self_test.json")
with open(test_path, "w", encoding="utf-8") as f:
    json.dump(rag_self_test_data, f, ensure_ascii=False, indent=4)
