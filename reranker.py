import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Reranker:
    def __init__(self, reranker_model_name_and_path, device='cuda'):
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name_and_path)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name_and_path)\
            .half().to(device).eval()
        self.device = device
        print("Successfully load rerank model")

    def rerank(self, docs, query, k=5):
        docs_ = []
        for item in docs:
            if isinstance(item, str):
                docs_.append(item)
            else:
                docs_.append(item.page_content)
        docs = list(docs_)
        pairs = []
        for d in docs:
            pairs.append([query, d])
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, max_length=512, truncation=True, return_tensor='pt')\
                .to(self.device)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()
        docs = [(docs[i], scores[i]) for i in range(len(docs))]
        docs = sorted(docs, key=lambda x: x[1], reverse=True)  # 根据元组的scores进行降序排序
        docs_ = []
        for item in docs:
            docs_.append(item[0])

        return docs_[:k]
