import os
import pdb
import json
import torch

from doc_text_parser import AcademicPaperParser
from doc_multimodal_parser import convert_pdf_to_images, analyze_page_images, batch_extract_tables_figures, generate_rich_descriptions
from llm_utils import define_ollama_llm, define_openailike_llm
from llama_index.llms.openai_like import OpenAILike
from extract_abstruct_info import AbstructExtractor

from multimodal_citation_query_engine import EmbeddingModel, MultiModalPaperIndex, CustomMMRetriever
from llama_index.core.query_engine import CitationQueryEngine
from extract_key_info_agent import PaperReader


class PaperReaderProcessor:
    def __init__(self, config_path, api_keys_path):
        print("🔧 正在初始化 PaperReader...")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        with open(api_keys_path, "r", encoding="utf-8") as f:
            api_keys = json.load(f)

        self.nougat_api_url = config["nougat_api_url"]
        self.layoutlm_api_url = config["layoutlm_api_url"]

        # 1.define parser llm
        if config["parser_llm"]["type"] == "ollama":
            self.parser_llm = define_ollama_llm(config["parser_llm"]["model_name"])
        elif config["parser_llm"]["type"] == "openai_like":
            api_key = api_keys[config["parser_llm"]["company"]]
            self.parser_llm = define_openailike_llm(
                model_name=config["parser_llm"]["model_name"],
                base_url=config["parser_llm"]["model_url"],
                api_key=api_key,
            )
        else:
            raise ValueError(f"Invalid parser_llm type {config['parser_llm']['type']}.")
        
        # 2.define parser mllm
        if config["parser_mllm"]["type"] == "ollama":
            self.parser_mllm = define_ollama_llm(config["parser_mllm"]["model_name"])
        elif config["parser_mllm"]["type"] == "openai_like":
            api_key = api_keys[config["parser_mllm"]["company"]]
            self.parser_mllm = define_openailike_llm(
                model_name=config["parser_mllm"]["model_name"],
                base_url=config["parser_mllm"]["model_url"],
                api_key=api_key,
            )
        else:
            raise ValueError(f"Invalid parser_mllm type {config['parser_mllm']['type']}.")
        
        # 3.define agent llm
        if config["agent_llm"]["type"] == "ollama":
            self.agent_llmm = define_ollama_llm(config["agent_llm"]["model_name"])
        elif config["agent_llm"]["type"] == "openai_like":
            api_key = api_keys[config["agent_llm"]["company"]]
            self.agent_llm = define_openailike_llm(
                model_name=config["agent_llm"]["model_name"],
                base_url=config["agent_llm"]["model_url"],
                api_key=api_key,
            )
        else:
            raise ValueError(f"Invalid agent_llm type {config['agent_llm']['type']}.")
        
        # 4.define long context llm
        if config["long_context_llm"]["type"] == "ollama":
            self.long_context_llm = define_ollama_llm(config["long_context_llm"]["model_name"])
        elif config["long_context_llm"]["type"] == "openai_like":
            api_key = api_keys[config["long_context_llm"]["company"]]
            self.long_context_llm = define_openailike_llm(
                model_name=config["long_context_llm"]["model_name"],
                base_url=config["long_context_llm"]["model_url"],
                api_key=api_key,
            )
        else:
            raise ValueError(f"Invalid long_context_llm type {config['long_context_llm']['type']}.")
        
        self.embed_model = EmbeddingModel(local_model_path=config["local_embedding_model_path"], 
                                    device="cuda:1" if torch.cuda.is_available() else "cpu", 
                                    batch_size=16).embed_model


    def process_pdf(self, file_path, out_dir):

        filename = os.path.splitext(os.path.basename(file_path))[0]
        out_fold = f"{out_dir}/{filename}"
        self.out_fold = out_fold

        out_parser_dir = f"{out_fold}/parser"
        page_images_dir = f"{out_parser_dir}/page_images"

        out_mmd_path = f"{out_parser_dir}/{filename}.mmd"
        os.makedirs(os.path.dirname(out_mmd_path), exist_ok=True)

        print("开始解析pdf...")

        # 第一步：使用nougat解析pdf为mmd格式的文本
        text_parser = AcademicPaperParser(parser_name="nougat", parser_api_url=self.nougat_api_url)
        text_parser.parse_paper(file_path, out_path=out_mmd_path)

        # 第二步：分析文档图片
        page_image_paths = convert_pdf_to_images(file_path, page_images_dir)
        layout_results = analyze_page_images(page_image_paths, self.layoutlm_api_url, print_results=False)

        # 第三步：提取图表的图片，生成图例
        merged_results = batch_extract_tables_figures(layout_results, out_parser_dir)

        # 第四步：使用多模态大模型生成图表的详细描述
        generate_rich_descriptions(self.parser_llm, self.parser_mllm, merged_results, out_parser_dir)

        # 第五步：使用多模态大模型生成摘要
        extractor = AbstructExtractor(self.parser_llm)
        output_obj = extractor.extract_abstruct(out_mmd_path)

        # 第六步：构建多模态索引
        out_index_path = f"{out_fold}/paper_index"
        os.makedirs(out_index_path, exist_ok=True)
        paper_index = MultiModalPaperIndex(self.embed_model, out_mmd_path, 
                                           f"{out_parser_dir}/descrption_for_figures_tables.json",
                                           out_index_path)
        index = paper_index.index
        paper_full_text = paper_index.get_full_text()

        # 第七步：构建带引用的检索引擎
        custom_retriever = CustomMMRetriever(
            index=index,
            candidate_k=20,
            text_k=5,
            image_k=1
        )

        query_engine = CitationQueryEngine.from_args(
            index=index,
            llm=self.agent_llm,
            retriever=custom_retriever,
            citation_chunk_size=512,
        )

        # 第八步：构建Agent
        self.paper_reader = PaperReader(query_engine, paper_full_text, self.agent_llm, self.long_context_llm)
        
        # 整理分析结果
        final_results = self.paper_reader.summarize_paper()

        abstruct_info = {
            "title": output_obj.title,
            "university": output_obj.university,
            "abstruct": output_obj.abstruct,
        }

        merged_dict = {**abstruct_info, **final_results}

        with open(f"{out_fold}/paper_interpretation.json", "w", encoding="utf-8") as f:
            json.dump(merged_dict, f, ensure_ascii=False, indent=2)

        return {
            'full_analysis':   f'{out_fold}/paper_interpretation.json',  # 全文分析JSON文件
            'figure_analysis': f'{out_fold}/parser/descrption_for_figures_tables.json'  # 图表分析JSON文件
        }

    def process_query(self, query):

        if not hasattr(self, "paper_reader"):
            raise ValueError("请先调用 process_pdf 方法")
        
        response = self.paper_reader.answer_question(query)

        reference = []
        if getattr(response, "sources", None):
            raw_output = response.sources[0]
            source_nodes = getattr(raw_output, "source_nodes", None)
            if source_nodes:
                reference = [source_node.node.get_text() for source_node in source_nodes]

        chat = {
            "response": str(response),
            "reference": reference
        }

        record = {
            "query": query,
            "response": str(response),
            "reference": reference
        }
        with open(f"{self.out_fold}/chat_data.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return chat


# 使用示例
if __name__ == "__main__":

    processor = PaperReaderProcessor('../config/config.json', '../config/api_keys.json')
