from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
import re

from pydantic import BaseModel, Field
from typing import List


# 长上下文全文分析工具
class StructuredListOutput(BaseModel):
    """一个用于承载结构化列表输出的Pydantic模型。"""
    items: List[str] = Field(..., description="从文本中提取出的项目列表，例如方法步骤或结论。")

class PaperReader:

    def __init__(self, query_engine, paper_full_text, llm, long_context_llm):

        self.long_context_llm = long_context_llm
        self.paper_full_text = paper_full_text
        self.query_engine = query_engine
        # 将查询引擎包装成工具
        # Agent 需要通过这个工具来调用 RAG 的能力
        self.query_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name="paper_rag_search",
                description=(
                    "一个基于RAG的论文查询工具。当你需要查找关于某个【特定】主题的【具体细节、原文证据或详细描述】时使用它。"
                    "例如：'详细描述数据预处理步骤' 或 '为结论A寻找数据支持'。"
                ),
            ),
        )
        
        # 长上下文大模型工具
        self.holistic_list_tool = FunctionTool.from_defaults(
            fn=self.get_holistic_list,
            name="holistic_list_extractor",
            description=(
                "一个基于长上下文的全文分析工具。当你需要【通读全文】来生成一个【宏观的、全面的列表】时，应该【优先使用】这个工具。"
                "例如：'列出所有的关键方法' 或 '总结所有的关键结论'。"
            )
        )

        self.agent = ReActAgent.from_tools([self.query_tool, self.holistic_list_tool], 
                                           llm=llm, verbose=True)
    
    def get_holistic_list(self, task_description):

        print(f"\n>> 正在调用长上下文LLM进行全文分析，任务: {task_description}...")
        
        prompt = (
            f"你是一位顶尖的科研助理。下面是一篇科研论文的完整文本。\n"
            f"请仔细通读全文，然后根据我的要求，提取出【{task_description}】。\n"
            f"--- 论文全文开始 ---\n"
            f"{self.paper_full_text}\n"
            f"--- 论文全文结束 ---\n\n"
            f"现在，请提取【{task_description}】："
        )
        prompt_template = PromptTemplate(template=prompt)
        structured_response = self.long_context_llm.structured_predict(
            StructuredListOutput, 
            prompt=prompt_template
        )
        return structured_response

    def answer_question(self, prompt):
        response = self.agent.chat(prompt)
        return response

    def summarize_paper(self):
        final_report = {
            "关键问题": "",
            "关键做法": [],
            "关键结论": []
        }

        print("\n\n===== 开始精读论文... =====\n")

        # --- 任务一：提取论文的关键问题 ---
        print("\n--- 步骤 1/3: 正在分析论文的关键问题 ---")
        try:
            key_question_prompt = "请总结这篇论文试图解决的核心科学问题或研究问题是什么？请用简洁的语言给出背景、问题和意义，请使用中文给出回答。"
            response = self.agent.chat(key_question_prompt)

            final_report["关键问题"] = str(response)
            print(f"【分析结果】\n{final_report['关键问题']}")
        except Exception as e:
            final_report["关键问题"] = f"分析失败: {e}"
            print(f"分析失败: {e}")

        # 2.1 Agent首先提取做法的宏观步骤
        print("\n--- 步骤 2.1: 提取方法论的宏观步骤 ---")
        method_steps_prompt = ("请从这篇论文的“方法”部分，完整地、按顺序地提取出作者所采取的所有关键步骤，请使用中文给出回答。")   
        method_steps_list = []
        try:
            response = self.agent.chat(method_steps_prompt)
            method_steps_list = re.findall(r'^\s*\d+[\.\)-]\s*(.*)', str(response), re.MULTILINE)
            if not method_steps_list:
                method_steps_list = [line.strip() for line in str(response).split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '*')]
            print("【识别出的方法步骤】:", method_steps_list)

        except Exception as e:
            print(f"提取方法步骤失败: {e}")

        # 2.2 Agent针对每一个步骤，进行RAG检索和详细描述
        if method_steps_list:
            print("\n--- 步骤 2.2: 针对每个步骤进行详细分析 ---")
            for i, step_description in enumerate(method_steps_list):
                print(f"\n  正在详细分析步骤 {i+1}/{len(method_steps_list)}: '{step_description}'...")
                detailed_step_prompt = (f"对于论文中提到的方法步骤 '{step_description}'，根据给定的论文中内容，详细准确地描述这个步骤的具体操作和技术细节，请使用中文给出回答。")
                
                try:
                    response = self.query_engine.query(detailed_step_prompt)
                    detailed_description = str(response)

                    final_report["关键做法"].append({
                        "步骤": step_description,
                        "详细描述": detailed_description,
                        "参考段落": [source_node.node.get_text() for source_node in response.source_nodes]
                    })
                except Exception as e:
                    final_report["关键做法"].append({
                        "步骤": step_description,
                        "详细描述": f"详细分析失败: {e}"
                    })
                    print(f"  【分析失败】: {e}")

        # 3.1 Agent首先提取结论的宏观列表
        print("\n--- 步骤 3.1: 提取关键结论的宏观列表 ---")
        conclusion_list_prompt = ("请总结并列出这篇论文中所有的关键结论或发现，请使用中文给出回答。")
        conclusion_list = []
        try:
            response = self.agent.chat(conclusion_list_prompt)
            conclusion_list = re.findall(r'^\s*\d+[\.\)-]\s*(.*)', str(response), re.MULTILINE)
            if not conclusion_list:
                # 如果正则没匹配到，尝试用换行符分割作为备用方案
                conclusion_list = [line.strip() for line in str(response).split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '*')]
            
            print("【识别出的关键结论】:")
            for i, conclusion in enumerate(conclusion_list):
                print(f"  {i+1}. {conclusion}")

        except Exception as e:
            print(f"提取关键结论列表失败: {e}")
            final_report["关键结论"].append({"结论": "提取失败", "详细阐述": str(e)})

        # 3.2 Agent针对每一个结论，进行RAG检索和详细阐述
        if conclusion_list:
            print("\n--- 步骤 3.2: 针对每个结论进行详细分析 ---")
            for i, conclusion_item in enumerate(conclusion_list):
                print(f"\n  正在详细分析结论 {i+1}/{len(conclusion_list)}: '{conclusion_item}'...")
                
                # 构建一个更精确的prompt，引导Agent进行RAG
                detailed_conclusion_prompt = (f"对于论文中提到的结论 '{conclusion_item}'，根据给定的论文中内容，详细准确地阐述这个结论并分析其意义，请使用中文给出回答。")
                
                try:
                    response = self.query_engine.query(detailed_conclusion_prompt)
                    detailed_description = str(response)
                    print(len(response.source_nodes))
                    
                    final_report["关键结论"].append({
                        "结论": conclusion_item,
                        "详细阐述": detailed_description,
                        "参考段落": [source_node.node.get_text() for source_node in response.source_nodes]
                    })
                except Exception as e:
                    final_report["关键结论"].append({
                        "结论": conclusion_item,
                        "详细阐述": f"详细分析失败: {e}"
                    })
                    print(f"  【分析失败】: {e}")

        return final_report
