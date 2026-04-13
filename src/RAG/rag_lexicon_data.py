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
from src.RAG.get_rag_prompt import rag_lexicon_prompt

_documents = None          # 缓存的词条列表
_bm25_index = None
_bm25_corpus = None
_embedding_model = None  

rag_data_path = os.path.join(project_root, RAG_DATA_PATH)
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



def load_documents():
    global _documents
    if _documents is not None:
        return _documents
    with open(rag_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载 {len(data['terms'])} 条词条")
    docs = []
    for term_info in data['terms']:
        term = term_info['term']
        category = term_info['category']
        definition = term_info['definition']
        # 仅用 term 作为检索文本（也可保留 term+category，但纯 term 最符合目标）
        retrieval_text = term   # 或 f"{term} {category}"
        docs.append({
            'term': term,
            'category': category,
            'definition': definition,
            'retrieval_text': retrieval_text,      # 用于 embedding 和 BM25
            'full_text': f"术语{term}, 类别是{category}, 其定义为{definition}"  # 保留给重排用
        })
    _documents = docs
    return docs


def get_embedding_model():
    """懒加载 embedding 模型（避免重复加载）"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_dir)
    return _embedding_model


class VectorDatabase:
    def __init__(self, documents, persist_dir: str = "./model/chroma"):
        self.documents = documents
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None  

    def build_vector_db(self):
        try:
            self.client.delete_collection("knowledge_base")
        except:
            pass
        self.collection = self.client.create_collection(name="knowledge_base")
        embedding_model = get_embedding_model()
        for idx, doc in enumerate(self.documents):
            # 关键改动：对 retrieval_text 编码，而不是 full_text
            embedding = embedding_model.encode(doc['retrieval_text']).tolist()
            self.collection.add(
                ids=str(idx),
                documents=doc['term'],          # 查询返回的 document 字段设为 term
                metadatas={
                    'term': doc['term'],
                    'category': doc['category'],
                    'definition': doc['definition'],
                    'retrieval_text': doc['retrieval_text'],
                    'full_text': doc['full_text']
                },
                embeddings=embedding
            )
        print(f"向量数据库构建完成，共 {len(self.documents)} 条数据")
        return self.collection
    


"""以下是检索相关的函数，包括稠密检索、稀疏检索和融合函数"""
def ensure_bm25_index():
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None:
        return
    _bm25_corpus = load_documents()
    # 使用 retrieval_text 进行分词建索引
    tokenized_corpus = [jieba.lcut(doc['retrieval_text']) for doc in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized_corpus)

def dense_search_internal(query, collection, top_k=20):
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )
    ids = results.get('ids', [[]])[0]
    docs = results.get('documents', [[]])[0]          # term
    distances = results.get('distances', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]

    hits = []
    for doc_id, term, dist, meta in zip(ids, docs, distances, metadatas):
        score = 1.0 / (1.0 + dist) if dist is not None else 1.0
        hits.append({
            'id': doc_id,
            'term': term,
            'definition': meta['definition'],
            'full_text': meta['full_text'],
            'score': score
        })
    hits.sort(key=lambda x: x['score'], reverse=True)
    return hits

def sparse_search_internal(query, top_k=20):
    ensure_bm25_index()
    tokens = jieba.lcut(query)
    scores = _bm25_index.get_scores(tokens)
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    hits = []
    for idx in ranked_idx:
        doc = _bm25_corpus[int(idx)]
        hits.append({
            'id': str(idx),
            'term': doc['term'],                 # 新增
            'definition': doc['definition'],    # 新增
            'full_text': doc['full_text'],       # 供重排使用
            'score': scores[idx]
        })
    return hits

def fuse_dense_sparse(dense_hits, sparse_hits, alpha=0.6):
    all_scores = []
    for h in dense_hits + sparse_hits:
        all_scores.append(h['score'])
    if not all_scores:
        return []
    min_score = min(all_scores)
    max_score = max(all_scores)
    range_score = max_score - min_score if max_score != min_score else 1.0

    fused = {}
    for weight, hits in ((alpha, dense_hits), (1 - alpha, sparse_hits)):
        for hit in hits:
            norm_score = (hit['score'] - min_score) / range_score
            item = fused.setdefault(
                hit['id'],
                {
                    'score': 0.0,
                    'term': hit['term'],
                    'definition': hit['definition'],
                    'full_text': hit['full_text']
                }
            )
            item['score'] += weight * norm_score

    fused_list = [
        {
            'id': doc_id,
            'term': v['term'],
            'definition': v['definition'],
            'full_text': v['full_text'],
            'score': v['score']
        }
        for doc_id, v in fused.items()
    ]
    fused_list.sort(key=lambda x: x['score'], reverse=True)
    return fused_list

def rag_retrieval(query, collection, top_k=10, alpha=0.1):  # 降低 alpha
    dense_hits = dense_search_internal(query, collection, top_k=top_k*5)
    sparse_hits = sparse_search_internal(query, top_k=top_k*5)
    fused = fuse_dense_sparse(dense_hits, sparse_hits, alpha=alpha)
    return fused[:top_k]


"""重排相关函数，使用 BGE-Reranker-Base 模型对检索结果进行重排序，提升相关性"""
class CrossEncoderReranker:
    def __init__(self, model_dir = model_dir,device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5):
        """
        candidates: 列表，每个元素是包含 'document' 键的字典
        返回重排序后的列表，每个元素增加 'rerank_score'
        """
        if not candidates:
            return []
        pairs = [(query, cand["document"]) for cand in candidates]
        # 分批处理，避免内存爆炸（可根据情况调整）
        batch_size = 32
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch = self.tokenizer(
                [p[0] for p in batch_pairs],
                [p[1] for p in batch_pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**batch).logits.squeeze(-1)
            all_scores.extend(logits.detach().cpu().numpy())
        # 添加分数
        for cand, score in zip(candidates, all_scores):
            cand["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


import json
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


persist_dir = os.path.join(project_root, "model", "chroma")
print(f"使用的向量数据库目录: {persist_dir}")
print("构建向量数据库...")
load_documents()
vdb = VectorDatabase(_documents, persist_dir=persist_dir)
collection = vdb.build_vector_db()
print("向量数据库构建完成")

reranker = CrossEncoderReranker(model_dir= rerank_model_dir)
train_data = load_json(train_data_path)
lexicon_rag_train_data = []
for item in train_data:
    query = item["content"]
    candidates = rag_retrieval(query, collection, top_k=20)
    # print(candidates)
    if not candidates:
        contexts = '无相关词条'
    else:
        reranked = reranker.rerank(query, candidates, top_k=3)
        contexts = "\n".join([item["document"] for item in reranked])
    lexicon_rag_train_data.append({
        'id': item["id"],
        "content": query,
        "contexts": contexts,
        "output": item["output"],
        "prompt": rag_lexicon_prompt(query, contexts)
    })

lexicon_train_path = os.path.join(project_root, "data", "lexicon_rag_train_data.json")
with open(lexicon_train_path, "w", encoding="utf-8") as f:
    json.dump(lexicon_rag_train_data, f, ensure_ascii=False, indent=4)


lexicon_rag_test_data = []
test_data = load_json(test_data_path)
for item in test_data:
    query = item["content"]
    candidates = rag_retrieval(query, collection, top_k=20)
    # print(candidates)
    if not candidates:
        contexts = '无相关词条'
    else:
        reranked = reranker.rerank(query, candidates, top_k=3)
        contexts = "\n".join([item["document"] for item in reranked])
    lexicon_rag_test_data.append({
        'id': item["id"],
        "content": query,
        "contexts": contexts,
        "output": item["output"],
        "prompt": rag_lexicon_prompt(query, contexts)
    })
lexicon_test_path = os.path.join(project_root, "data", "lexicon_rag_test_data.json")
with open(lexicon_test_path, "w", encoding="utf-8") as f:
    json.dump(lexicon_rag_test_data, f, ensure_ascii=False, indent=4)