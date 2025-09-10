import os
import re
import json
import torch

from llama_index.core.schema import ImageNode, TextNode, Document
from llama_index.core.retrievers import BaseRetriever

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding

from llama_index.core import VectorStoreIndex, StorageContext, QueryBundle, Document, Settings, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceWindowNodeParser


#   将Markdown拆分为 文本段落 + 表格 的独立块，目的是保证表格的完整性，不要让其被拆散
def split_markdown_protect_latex_table(md_text):
    begin_pattern = r'\\begin\{table\}'
    end_pattern = r'\\end\{table\}'
    tag_pattern = re.compile(f'({begin_pattern}|{end_pattern})')

    pos = 0
    result = []
    matches = list(tag_pattern.finditer(md_text))
    i = 0

    while i < len(matches):
        match = matches[i]
        tag = match.group(1)

        if tag == r'\begin{table}':
            begin_pos = match.start()

            # 向后查找对应的 end
            j = i + 1
            matched = False
            while j < len(matches):
                next_tag = matches[j].group(1)
                if next_tag == r'\begin{table}':
                    # 中间又遇到了 begin，前面的 begin 不闭合，作为普通文本
                    result.append(md_text[pos:matches[j].start()])
                    pos = matches[j].start()
                    i = j - 1  # 从新 begin 开始匹配
                    matched = True
                    break
                elif next_tag == r'\end{table}':
                    # 找到闭合
                    end_pos = matches[j].end()
                    if begin_pos > pos:
                        result.append(md_text[pos:begin_pos])
                    result.append(md_text[begin_pos:end_pos])
                    pos = end_pos
                    i = j  # 跳到 end 之后
                    matched = True
                    break
                j += 1

            if not matched:
                # 没有匹配到 end，也作为普通文本
                result.append(md_text[pos:])
                return [r.strip() for r in result if r.strip()]
        else:
            # 是 end 但前面没匹配 begin，直接跳过并作为文本
            result.append(md_text[pos:match.end()])
            pos = match.end()
        i += 1

    if pos < len(md_text):
        result.append(md_text[pos:])

    return [r.strip() for r in result if r.strip()]


class EmbeddingModel:
    def __init__(self, embed_type='local_baai', local_model_path=None, device=None, batch_size=None, api_model_name=None, api_key=None):

        if embed_type == 'local_baai':
            self.local_model_path = local_model_path
            self.embed_model = HuggingFaceEmbedding(model_name=local_model_path, device=device, embed_batch_size=batch_size)
        elif embed_type == 'api_qwen':
            self.embed_model = DashScopeEmbedding(model_name=api_model_name,
                                                  text_type="document",
                                                  api_key=api_key)
        else:
            raise ValueError(f"Invalid embed type {embed_type}.")


class MultiModalPaperIndex:
    def __init__(self, embed_model, mmd_path, media_json_path, local_index_dir):

        self.mmd_path = mmd_path
        self.media_json_path = media_json_path

        if len(os.listdir(local_index_dir)) == 0:

            print("未找到本地索引，开始构建新的索引...")
            self.embed_model = embed_model

            self.image_nodes, self.image_descriptions = self.load_image_nodes(self.media_json_path)
            self.text_nodes = self.load_text_nodes(self.mmd_path, self.image_descriptions)
            
            self.all_nodes = self.text_nodes + self.image_nodes
            self.index = VectorStoreIndex(nodes=self.all_nodes, embed_model=self.embed_model)

            self.index.storage_context.persist(persist_dir=local_index_dir)
            print(f"索引构建完成，并保存到{local_index_dir}")

        else:
            print(f"发现本地索引，正在从 '{local_index_dir}' 加载...")
            Settings.embed_model = embed_model
            storage_context = StorageContext.from_defaults(persist_dir=local_index_dir)
            self.index = load_index_from_storage(storage_context)
            print("索引加载成功！")


    def load_text_nodes(self, mmd_path, image_description):

        with open(mmd_path, "r", encoding="utf-8") as f:
            mmd_content = f.read()
        
        text_document = Document(text=mmd_content)

        all_text_nodes = []

        # --- 策略1: SentenceWindowNodeParser ---
        window_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,  # 每个节点包含当前句子，以及前后各1个句子
            include_metadata=True,
            sentence_splitter=lambda text: re.split('(?<=[.?!。？！])\s+', text), # 简单的句子分割
        )
        window_nodes = window_parser.get_nodes_from_documents([text_document, image_description])
        all_text_nodes.extend(window_nodes)
        print(f"使用 SentenceWindowNodeParser 生成了 {len(window_nodes)} 个文本节点。")

        # --- 策略2: SemanticSplitterNodeParser ---
        semantic_parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model,
        )
        semantic_nodes = semantic_parser.get_nodes_from_documents([text_document, image_description])
        all_text_nodes.extend(semantic_nodes)
        print(f"使用 SemanticSplitterNodeParser 生成了 {len(semantic_nodes)} 个文本节点。")

        print(f"总共生成了 {len(all_text_nodes)} 个文本节点。")

        return all_text_nodes

    def load_image_nodes(self, json_path):

        image_nodes = []
        image_descriptions = ""
        with open(json_path, "r", encoding="utf-8") as f:
            media_data = json.load(f)

        for item in media_data:
            image_node = ImageNode(
                image=item["image_path"],        # 图片路径
                text=item["caption_text"],   # 图片的详细描述，这将用于被Embedding和检索
                metadata={                       # 存储额外信息
                    "path": item["image_path"],
                    "type": item["class"]
                }
            )
            image_nodes.append(image_node)
            image_descriptions += item["rich_description"] + '\n'

        print(f"成功加载了 {len(image_nodes)} 个媒体节点 (图片/表格)。")

        return image_nodes, Document(text=image_descriptions)
    
    def get_full_text(self):

        with open(self.mmd_path, "r", encoding="utf-8") as f:
            mmd_content = f.read()

        image_descriptions = ""
        with open(self.media_json_path, "r", encoding="utf-8") as f:
            media_data = json.load(f)

        for item in media_data:
            image_descriptions += item["rich_description"] + '\n'

        return mmd_content + '\n' + image_descriptions



class CustomMMRetriever(BaseRetriever):

    def __init__(self, index, candidate_k=30, text_k=5, image_k=3):
        base_vector_retriever = index.as_retriever(similarity_top_k=candidate_k)
        self._vector_retriever = base_vector_retriever

        self._text_k = text_k
        self._image_k = image_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        # 为了确保能找到足够的目标节点，我们先检索一个较大的数量作为候选池
        candidate_nodes = self._vector_retriever.retrieve(query_bundle)

        # 从候选池中筛选出所需数量的文本和图片节点
        retrieved_texts = []
        retrieved_images = []

        for node_with_score in candidate_nodes:
            node = node_with_score.node
            # 检查节点类型，并确保尚未达到数量上限
            if isinstance(node, TextNode) and len(retrieved_texts) < self._text_k:
                retrieved_texts.append(node_with_score)
            elif isinstance(node, ImageNode) and len(retrieved_images) < self._image_k:
                retrieved_images.append(node_with_score)
            
            # 如果两种类型的节点都已找齐，则提前结束
            if len(retrieved_texts) == self._text_k and len(retrieved_images) == self._image_k:
                break

        return retrieved_texts + retrieved_images
