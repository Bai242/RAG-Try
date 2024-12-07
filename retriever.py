from abc import ABC
from transformers import AutoTokenizer, AutoModel
import torch
import jieba
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS


class TextEmbedding(Embeddings, ABC):
    def __init__(self, embedding_model_name_and_path, batch_size=64, max_len=512, device='cuda', **kwargs):

        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(embedding_model_name_and_path, trust_remote_code=True).half().to(device)
        # trust_remote_code=True允许从Hugging Face服务器加载模型。.half().将模型权重转换为半精度（以便在GPU上更快地处理）
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name_and_path, trust_remote_code=True)
        if 'bge' in embedding_model_name_and_path:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文档"
        else:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
        self.embedding_model_name_and_path = embedding_model_name_and_path
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        print("Successfully load embedding model")

    # 实现“白化”（Whitening）的线性变换，常用于提高嵌入向量的质量，特别是降低向量间的相关性，使其更适合后续的相似度计算等任务
    def compute_kernel_bias(self, vecs, n_components=384):
        """
            bertWhitening: https://spaces.ac.cn/archives/8069
            计算kernel和bias
            vecs.shape = [num_samples, embedding_size]，
            最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)  # true保持维度为(1, embedding_size)而不是（embedding_size）
        cov = np.cov(vecs.T)  # 计算协方差矩阵，协方差矩阵描述了不同维度之间线性相关的程度
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))  # 白化的核心步骤，通过奇异值分解对数据进行去相关和缩放

        # 只取前 n_components 列是为了降维, -mu 是偏差项，在应用变换时需要加到原始向量上
        return W[:, :n_components], -mu

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
              Compute corpus embeddings using a HuggingFace transformer model.
          Args:
              texts: The list of texts to embed.
          Returns:
              List of embeddings, one for each text.
          """
        num_text = len(texts)
        texts = [t.replace("\n", " ") for t in texts]
        sentence_embeddings = []

        for start in range(0, num_text, self.batch_size):
            end = min(start + self.batch_size, num_text)
            batch_texts = texts[start:end]
            encoded_input = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True,
                                           return_tensor='pt').to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)  # **的作用是将字典解包，其键值作为关键字参数传递给self.model
                # Perform pooling. In this case, cls pooling.
                if 'gte' in self.embedding_model_name_and_path:
                    batch_embeddings = model_output.last_hidden_state[:, 0]  # (batch_size, sequence_length, hidden_size),选出的是每一批次的[cls]
                else:
                    batch_embeddings = model_output[0][:, 0]  # 也是选择[cls],只不过模型输出是个list，[0]包含所有token的hidden_state

                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                sentence_embeddings.extend(batch_embeddings.tolist())

                # 后处理：白化
                # sentence_embeddings = np.array(sentence_embeddings)
                # self.W, self.mu = self.compute_kernel_bias(sentence_embeddings)
                # sentence_embeddings = (sentence_embeddings+self.mu) @ self.W
                # self.W, self.mu = torch.from_numpy(self.W).cuda(), torch.from_numpy(self.mu).cuda()
                return sentence_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
            Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        if 'bge' in self.embedding_model_name_and_path:
            encoded_input = self.tokenizer([self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], padding=True,
                                           truncation=True, return_tensor='pt').to(self.device)
        else:
            encoded_input = self.tokenizer([text], padding=True,
                                           truncation=True, return_tensor='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
        return sentence_embeddings[0].tolist()


class Retriever:
    def __init__(self, embedding_model_name_and_path=None, corpus=None, device='cuda',language='Chinese'):
        self.device = device
        self.langchain_corpus = [Document(page_content=t) for t in corpus]
        self.corpus = corpus
        self.language = language
        if language == 'Chinese':
            tokenized_documents = [jieba.lcut(doc) for doc in corpus]
        else:
            tokenized_documents = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_documents)

        self.embedding_model = TextEmbedding(embedding_model_name_and_path=embedding_model_name_and_path)
        self.db = FAISS.from_documents(self.langchain_corpus, self.embedding_model)

    def bm25_retrieval(self, query, n=10):
        # 此处中文使用jieba分词
        query = jieba.lcut(query)
        res = self.bm25.get_top_n(query, self.corpus, n=n)
        return res

    def emb_retriever(self, query, k=10):

        search_docs = self.db.similarity_search(query, k=k)
        res = [doc.page_content for doc in search_docs]
        return res

    def retrieval(self, query, methods=None):
        if methods is None:
            methods = ['bm25', 'emb']
        search_res = list()
        for method in methods:
            if method == 'bm25':
                bm25_res = self.bm25_retrieval(query)
                for item in bm25_res:
                    if item not in search_res:
                        search_res.append(item)
            elif method == 'emb':
                emd_res = self.emb_retriever(query)
                for item in emd_res:
                    if item not in search_res:
                        search_res.append(item)

        return search_res











