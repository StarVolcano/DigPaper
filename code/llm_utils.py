from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock


def define_openailike_llm(model_name, base_url, api_key):
    llm = OpenAILike(
        model=model_name,
        api_base=base_url,
        api_key=api_key,
        is_chat_model=True,
    )
    return llm


def define_ollama_llm(llm_name):
    llm = Ollama(
        model=llm_name,
        request_timeout=180.0,
        context_window=5000,
    )
    return llm


def get_mllm_response(mllm, system_prompt, text, image_path):

    if system_prompt != None:
        messages = [
            ChatMessage(
                role="system",
                content=system_prompt
            ),
            ChatMessage(
                role="user",
                blocks=[
                    ImageBlock(path=image_path),
                    TextBlock(text=text),
                ],
            )
        ]

    else:
        messages = [
            ChatMessage(
                role="user",
                blocks=[
                    TextBlock(text=text),
                    ImageBlock(path=image_path),
                ],
            )
        ]
    resp = mllm.chat(messages).message.content

    return resp

