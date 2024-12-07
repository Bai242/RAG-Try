import re
import tqdm
import PyPDF2
import pandas as pd

class Reader:
    def __init__(self, corpus_path: str):
        self.corpus = None
        if corpus_path.endswith('.pdf'):
            self.corpus = self.extract_pdf_page_text(corpus_path)
        elif 'Multi-CPR' in corpus_path:
            self.corpus = self.extract_multiCPR_text(corpus_path)
        else:
            self.corpus = self.extract_my_file(corpus_path)

    def extract_pdf_page_text(self, filepath, max_len=256, overlap_len=100):
        page_content = []
        # 打开pdf文件，读取每页内容
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in tqdm.tqdm(pdf_reader.pages, desc='解析PDF文件ing...'):
                page_text = page.extract_text().strip()
                raw_text = [text.strip() for text in page_text.split('\n')]
                new_text = '\n'.join(raw_text)  # 连接成长字符串
                new_text = re.sub(r'\n\d{2, 3}\s?', '\n')
                # 匹配以换行符 \n 开头，后面跟着 2 到 3 位数字（\d{2,3}），并可能有一个空格（\s?）的部分，替换词\n
                if len(new_text) > 10 and '..............' not in new_text:
                    page_content.append(new_text)

        cleaned_chunks = []
        i = 0
        # 暴力将整个pdf当做一个字符串，然后按照固定大小的滑动窗口切割
        all_str = ''.join(page_content)
        all_str = all_str.replace('\n', '')
        while i < len(all_str):
            chunk = all_str[i:i+max_len]
            if len(chunk) > 10:
                cleaned_chunks.append(chunk)
            i += (max_len - overlap_len)

        return cleaned_chunks

    def extract_multiCPR_text(self, filepath):

        corpus = pd.read_csv(filepath, sep='\t', header=None)
        corpus.columns = ['pid', 'passage']

        return  corpus.passage.values.tolist()

    def extract_my_file(self):

        return NotImplementedError
