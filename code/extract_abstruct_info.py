import nest_asyncio

nest_asyncio.apply()

from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llm_utils import define_ollama_llm


class AbstructInfo(BaseModel):
    
    title: str = Field(description="论文的标题")
    university: str = Field(description="作者团队所属的大学或机构（主要的两到三个）")
    abstruct: str = Field(description="用第三人称对论文摘要的总结")


class AbstructExtractor:

    def __init__(self, llm):
        self.sllm = llm.as_structured_llm(output_cls=AbstructInfo)

    def extract_abstruct(self, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            front_text = f.read()[:3000]
        input_msg = ChatMessage.from_str("请使用中文给出回答。\n" + front_text)
        output_obj = self.sllm.chat([input_msg]).raw
        return output_obj
