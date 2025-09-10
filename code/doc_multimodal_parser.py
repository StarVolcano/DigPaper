import os
import re
import json
import pdb
import fitz
import requests
import pytesseract
from tqdm import tqdm
from PIL import Image

from image_utils import get_center, crop_box, find_closest_text
from prompts import SYSTEM_PROMPT_FIGURE, PROMPT_GENERATE_FIGURES_CAPTION, SYSTEM_PROMPT_TABLE, PROMPT_GENERATE_TABLES_CAPTION
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage
from llm_utils import define_ollama_llm, get_mllm_response


# PDF转图片
def convert_pdf_to_images(pdf_path, output_folder):
    """将PDF的每一页转换为PNG图片"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        # 使用更高的DPI来获得更清晰的图像，这对于模型识别很重要
        pix = page.get_pixmap(dpi=150) 
        output_path = os.path.join(output_folder, f"page_{i+1}.png")
        pix.save(output_path)
        image_paths.append(output_path)
    doc.close()
    print(f"Successfully converted {len(image_paths)} pages to images in '{output_folder}'")
    return image_paths


# 调用LayoutLMv3的API分析文档图片
def analyze_page_images(image_paths, api_url, print_results=False): 

    payload = {"image_paths": image_paths}
    response = requests.post(api_url, json=payload)

    # 打印结构化推理结果
    if print_results:
        for result in response.json():
            print(f"Image: {result['image_path']}")
            if "error" in result:
                print("  Error:", result["error"])
            else:
                print("  Classes:", result["classes"])
                print("  Boxes:", result["boxes"])
                print("  Scores:", result["scores"])
                print("  Vis Path:", result["visualized_path"])

    return response.json()


def concat_image_and_caption_robust(image_region: Image.Image, caption_region: Image.Image, save_path: str):
    """
    将主图和图例图区域垂直拼接，保证宽度一致且图例完整显示。
    """
    # 获取图像原始宽度
    target_width = image_region.width

    # 计算缩放比例（将 caption 的宽度调整为 image 区域一致）
    cap_w, cap_h = caption_region.size
    if cap_w != target_width:
        new_caption_height = int(cap_h * (target_width / cap_w))
        caption_region = caption_region.resize((target_width, new_caption_height), Image.LANCZOS)

    # 创建新图像（高 = 主图高 + 图例高）
    total_height = image_region.height + caption_region.height
    new_image = Image.new("RGB", (target_width, total_height), color="white")

    # 粘贴两个区域
    new_image.paste(image_region, (0, 0))
    new_image.paste(caption_region, (0, image_region.height))

    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new_image.save(save_path)
    return new_image


def process_layout_result(pred_result, out_dir):
    image_path = pred_result["image_path"]
    image = Image.open(image_path).convert("RGB")

    boxes = pred_result["boxes"]
    classes = pred_result["classes"]

    table_or_figures = [i for i, c in enumerate(classes) if c in [3, 4]]
    texts = [i for i, c in enumerate(classes) if c == 0]

    results = []

    # 如果当前页面没有表格或图片，返回0
    if len(table_or_figures) == 0:
        print(f"Found no tables or figures in {image_path}.")
        return 0

    for i in table_or_figures:
        box = boxes[i]
        class_name = "table" if classes[i] == 3 else "figure"

        if not texts:
            continue

        nearest_text_index = find_closest_text(box, [boxes[t] for t in texts])
        caption_box = boxes[texts[nearest_text_index]]

        image_region = crop_box(image, box)
        caption_region = crop_box(image, caption_box)
        caption_text = pytesseract.image_to_string(caption_region, lang="eng")

        save_path = f"{out_dir}/all_figures_tables/{os.path.basename(image_path)}_{class_name}{i}.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # image_region.save(save_path)
        concat_image_and_caption_robust(image_region, caption_region, save_path)

        results.append({
            "image_path": save_path,
            "caption_text": caption_text,
            "class": class_name,
        })
    return results


def batch_extract_tables_figures(layout_results, out_dir):

    merged_results = []
    for i, layout_result in enumerate(layout_results):
        merge_result = process_layout_result(layout_result, out_dir)
        if merge_result != 0:
            for item in merge_result:
                merged_results.append(item)
    return merged_results


def generate_rich_description(mllm, merged_result):

    image_type = merged_result['class']
    image_path = merged_result['image_path']
    image_caption = merged_result['caption_text']

    if image_type == 'figure':
        system_prompt = SYSTEM_PROMPT_FIGURE
        prompt = PromptTemplate(PROMPT_GENERATE_FIGURES_CAPTION).format(image_caption)
        response = get_mllm_response(mllm, system_prompt, prompt, image_path)
    
    elif image_type == 'table':
        system_prompt = SYSTEM_PROMPT_TABLE
        prompt = PromptTemplate(PROMPT_GENERATE_TABLES_CAPTION).format(image_caption)
        response = get_mllm_response(mllm, system_prompt, prompt, image_path)

    return response


def generate_rich_descriptions(llm, mllm, translate_prompt, merged_results, save_dir):
    for i, merged_result in tqdm(enumerate(merged_results),desc=f'Generating descrption for {len(merged_results)} figure or table...'):
        rich_description = generate_rich_description(mllm, merged_result)
        merged_results[i]["rich_description"] = rich_description

        input_msg = ChatMessage.from_str(translate_prompt + rich_description)
        rich_description_cn = llm.chat([input_msg]).message.content
        merged_results[i]["rich_description_cn"] = rich_description_cn

    save_path = f"{save_dir}/descrption_for_figures_tables.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=4)
    print(f'Json of figures and tables saved in {save_path}.')
