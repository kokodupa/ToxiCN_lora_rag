import json
import numpy as np
from typing import List, Dict, Any, Tuple
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import jieba
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from config.config import RAG_DATA_PATH, TRAIN_MODEL_PATH, RAG_INSTRUCTION, RAG_SYSTEM_MESSAGE

# ---------- 全局变量（缓存数据，避免重复加载） ----------
_documents = None          # 缓存的词条列表
_bm25_index = None
_bm25_corpus = None
_embedding_model = None    # 后面会懒加载

def get_embedding_model():
    """懒加载 embedding 模型（避免重复加载）"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    return _embedding_model

def load_documents():
    """加载俚语数据，并缓存结果"""
    global _documents
    if _documents is not None:
        return _documents
    with open(RAG_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载 {len(data['terms'])} 条词条")
    docs = []
    for term_info in data['terms']:
        term = term_info['term']
        category = term_info['category']
        definition = term_info['definition']
        doc_text = f"术语{term}, 类别是{category}, 其定义为{definition}"
        docs.append({
            'term': term,
            'category': category,
            'definition': definition,
            'full_text': doc_text
        })
    _documents = docs
    return docs

def ensure_bm25_index():
    """构建 BM25 索引（懒加载）"""
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None:
        return
    _bm25_corpus = load_documents()
    tokenized_corpus = [jieba.lcut(doc['full_text']) for doc in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized_corpus)


class VectorDatabase:
    def __init__(self, documents, persist_dir: str = "./model/chroma"):
        self.documents = documents
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None  # 将在 build 时赋值

    def build_vector_db(self):
        """构建向量数据库，返回 chromadb 集合对象"""
        # 如果已存在 collection，先删除重建（避免重复数据）
        try:
            self.client.delete_collection("knowledge_base")
        except:
            pass
        self.collection = self.client.create_collection(name="knowledge_base")
        embedding_model = get_embedding_model()
        for idx, doc in enumerate(self.documents):
            embedding = embedding_model.encode(doc['full_text']).tolist()
            self.collection.add(
                ids=str(idx),
                documents=doc['full_text'],
                metadatas={
                    'term': doc['term'],
                    'category': doc['category'],
                    'definition': doc['definition']
                },
                embeddings=embedding
            )
        print(f"向量数据库构建完成，共 {len(self.documents)} 条数据")
        return self.collection


def dense_search_internal(query, collection, top_k=20):
    """稠密检索（向量相似度）"""
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    ids = results.get('ids', [[]])[0]
    docs = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]  # 距离越小越相似

    hits = []
    for doc_id, doc_text, dist in zip(ids, docs, distances):
        # 距离转分数（距离 0 表示完全匹配，越远分数越低）
        score = 1.0 / (1.0 + dist) if dist is not None else 1.0
        hits.append({
            'id': doc_id,
            'document': doc_text,
            'score': score
        })
    # 按分数降序
    hits.sort(key=lambda x: x['score'], reverse=True)
    return hits

def sparse_search_internal(query, top_k=20):
    """BM25 稀疏检索"""
    ensure_bm25_index()
    tokens = jieba.lcut(query)
    scores = _bm25_index.get_scores(tokens)
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    hits = []
    for rank, idx in enumerate(ranked_idx):
        doc = _bm25_corpus[int(idx)]
        hits.append({
            'id': str(idx),
            'document': doc['full_text'],
            'score': scores[idx]   # 直接用 BM25 分数
        })
    return hits

def fuse_dense_sparse(dense_hits, sparse_hits, alpha=0.6):
    """加权融合，使用分数归一化（min-max）"""
    # 收集所有分数，用于归一化
    all_scores = []
    for h in dense_hits:
        all_scores.append(h['score'])
    for h in sparse_hits:
        all_scores.append(h['score'])
    if not all_scores:
        return []
    min_score = min(all_scores)
    max_score = max(all_scores)
    range_score = max_score - min_score if max_score != min_score else 1.0

    # 归一化 + 融合
    fused = {}
    for weight, hits in ((alpha, dense_hits), (1 - alpha, sparse_hits)):
        for hit in hits:
            norm_score = (hit['score'] - min_score) / range_score
            item = fused.setdefault(
                hit['id'],
                {'score': 0.0, 'document': hit['document']}
            )
            item['score'] += weight * norm_score
    fused_list = [
        {'id': doc_id, 'document': v['document'], 'score': v['score']}
        for doc_id, v in fused.items()
    ]
    fused_list.sort(key=lambda x: x['score'], reverse=True)
    return fused_list

def rag_retrieval(query, collection, top_k=10, alpha=0.6):
    """混合检索入口，返回 top_k 条文档（字典格式，含分数）"""
    dense_hits = dense_search_internal(query, collection, top_k=top_k*2)   # 多召回一些再融合
    sparse_hits = sparse_search_internal(query, top_k=top_k*2)
    fused = fuse_dense_sparse(dense_hits, sparse_hits, alpha=alpha)
    return fused[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
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


class AnswerGenerator:
    def __init__(self, model_path=TRAIN_MODEL_PATH, max_input_length=2048,
                 max_new_tokens=256, temperature=0.2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(self, query, contexts):
        """
        contexts: 检索到的相关文档拼接成的字符串
        """
        # 构建消息列表
        messages = [
            {"role": "system", "content": RAG_SYSTEM_MESSAGE},
            {"role": "user", "content": RAG_INSTRUCTION + query + "\n相关知识：" + contexts}
        ]
        # 使用 tokenizer 的聊天模板转换为模型需要的文本格式
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # 解码时去掉输入的 prompt 部分
        input_len = inputs.input_ids.shape[1]
        answer = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return answer

# ---------- RAG 主流程 ----------
class RAGPipeline:
    def __init__(self, data_path=RAG_DATA_PATH, persist_dir="./model/chroma"):
        # 加载数据
        self.documents = load_documents()
        # 构建向量数据库
        self.vdb = VectorDatabase(self.documents, persist_dir=persist_dir)
        self.collection = self.vdb.build_vector_db()
        # 初始化重排序器和生成器
        self.reranker = CrossEncoderReranker()
        self.generator = AnswerGenerator()

    def answer(self, query, top_k=5, rerank_top_k=3):
        """
        query: 用户问题
        top_k: 混合检索返回的文档数（供重排序用）
        rerank_top_k: 重排序后最终使用的文档数（作为上下文）
        """
        # 1. 混合检索，得到候选文档（字典列表，含分数）
        candidates = rag_retrieval(query, self.collection, top_k=top_k)
        if not candidates:
            return answer = self.generator.generate(query, contexts="无")
        # 2. 重排序
        reranked = self.reranker.rerank(query, candidates, top_k=rerank_top_k)
        # 3. 拼接上下文
        contexts = "\n".join([item["document"] for item in reranked])
        # 4. 生成答案
        answer = self.generator.generate(query, contexts)
        return answer

if __name__ == "__main__":
    rag = RAGPipeline()
    