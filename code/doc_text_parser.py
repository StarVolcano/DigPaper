import os
import json
import requests

class AcademicPaperParser:
    """
    一个用于解析学术论文PDF的类，使用LlamaParse提取文本、表格和图像，
    并保留其元数据以供下游使用。
    """
    def __init__(self, parser_name, parser_api_url):

        self.parser_name = parser_name

        if parser_name == 'nougat':
            self.nougat_api_url = parser_api_url
        elif parser_name == 'llama_parse':
            raise ValueError(f"{parser_name} has not been deployed yet.")
        else:
            raise ValueError("Invalid parser name. Supported values are 'nougat' and 'llama_parse'.")
        
    def parse_pdf_nougat(self, pdf_path, out_mmd_path, params={}):

        # 读取并上传PDF
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path, f, "application/pdf")}
            response = requests.post(self.nougat_api_url, files=files, params=params)

        # 输出解析结果（Markdown格式）
        if response.status_code == 200:

            markdown_text = json.loads(response.text)
            with open(out_mmd_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

        else:
            raise ValueError(f"Error: {response.status_code}, response content: {response.text}")
        
        return response.text
        

    def parse_paper(self, file_path, out_path, params=None):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.parser_name == "nougat":

            if params is not None:
                nougat_params = params
            else:
                nougat_params = {}

            return self.parse_pdf_nougat(file_path, out_path, nougat_params)
            
        elif self.parser_name == 'llama_parse':
            raise ValueError(f"{self.parser_name} has not been deployed yet.")
        else:
            raise ValueError("Invalid parser name. Supported values are 'nougat' and 'llama_parse'.")
