import json
from argparse import ArgumentParser
from llm_infer import LLMPredictor

from retriever import Retriever
from reranker import Reranker
from read_corpus import Reader


def arg_parse():

    parser = ArgumentParser()
    parser.add_argument('--llm_model_name_and_path', default="Qwen/Qwen1.5-7B-Chat", help='the path of llm to use')
    parser.add_argument('--embedding_model_name_and_path', default='BAAI/bge-large-zh-v1.5', help='the path of embedding to use')
    parser.add_argument('--rerank_model_name_or_path', default='BAAI/bge-reranker-large',
                        help='the path of rerank model to use')
    parser.add_argument('--retrieval_methods', default=None,
                        help='the method to retrieval, you can choose from [bm25, emb]')
    parser.add_argument('--corpus_path', default="dataset/tianchi/初赛训练数据集.pdf", help='corpus path')
    parser.add_argument('--test_query_path', default='dataset/tianchi/测试问题.json', help='test query path')
    parser.add_argument('--num_input_docs', default=4, help='num of context docs')

    return parser.parse_args()


def main():
    args = arg_parse()

    # 语料库读取
    reader = Reader(args.corpus_path)
    corpus = reader.corpus

    retriever = Retriever(embedding_model_name_and_path=args.embedding_model_name_and_path, corpus=corpus)
    reranker = Reranker(reranker_model_name_and_path=args.rerank_model_name_and_path)
    llm = LLMPredictor(llm_model_name_and_path=args.llm_model_name_and_path)

    result_list = []
    with open(args.text_query_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    for i, line in enumerate(result):
        query = line['question']
        retriever_res = retriever.retrieval(query)
        rerank_res = reranker.rerank(retriever_res, query, args.num_input_docs)
        res = llm.predict('\n'.join(rerank_res), query).strip()
        line['answer'] = res
        del line['answer_1'], line['answer_2'], line['answer_3']
        result_list.append(line)
        print('question {}'.format(i))
        print(line, '\n')

    res_file_path = 'res.json'
    with open(res_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()




