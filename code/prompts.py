from llama_index.core.prompts import PromptTemplate

PROMPT_TRANSLATE = '''
请把以下段落翻译成中文，请只输出纯文本，不要使用任何 markdown 格式（例如 #、*、```、- 等符号）。
'''

# 生成图片描述的prompt
SYSTEM_PROMPT_FIGURE = '''
You are an expert research assistant specializing in scientific data visualization analysis. Your task is to analyze the provided scientific figure and its accompanying caption.
'''

PROMPT_GENERATE_FIGURES_CAPTION ='''
Figure Caption: {caption}

Instructions:
Based on the provided image and caption, perform the following two tasks:
	1.	Detailed Description: Meticulously describe all the visual components of the figure. This includes, but is not limited to, the type of plot (e.g., bar chart, line graph, scatter plot, heatmap), the axes (labels, units, scale), data points, error bars, legends, color schemes, and any annotations or labels within the figure itself.
	2.	In-depth Interpretation: Interpret the figure in the context of its caption. Explain the key findings, significant trends, and relationships presented in the data. What is the main message or conclusion that can be drawn from this figure? What does the data imply?

Output Format:
Write a comprehensive answer consisting of two sections:
Description: A detailed description of the figure’s visual elements.
Interpretation: An in-depth interpretation of the figure’s meaning, key findings, and the conclusions drawn from the data presented.
Do not output any JSON or other explanatory text. Only include the two clearly labeled sections: Description and Interpretation.
'''


# 生成表格描述的prompt
SYSTEM_PROMPT_TABLE = '''
You are an expert research analyst specializing in structured data extraction from scientific tables. Your task is to analyze the provided image of a table and its accompanying title or caption.
'''

PROMPT_GENERATE_TABLES_CAPTION ='''
Table Caption: {caption}

Instructions:
Based on the provided image and caption, provide a detailed and professional English explanation of the table, focusing on the following aspects:
	1. The overall structure and layout of the table
	2. What each row and column represents
	3. Key trends, comparisons, or patterns shown in the data
	4. The scientific insights or conclusions that can be drawn from the table

Output Format:
Write a comprehensive answer under a single clearly labeled section:
Explanation: A professional and structured paragraph that explains the table’s contents, patterns, and implications.
Do not output any JSON or any explanatory notes—only the final explanation text.
'''



PROMPT_JSON2HTML = '''
# ROLE:
You are an expert front-end web developer specializing in data visualization and modern UI/UX design. Your task is to transform a given JSON data structure into a visually appealing, well-structured, and user-friendly single-file HTML page.

# CONTEXT:
Here is the JSON data I want to display. It represents the iterpretation of a paper.

{json_content}

TASK:
Convert the provided JSON data into a single, complete, and aesthetically pleasing HTML file.

REQUIREMENTS:
Structure & Semantics:
	Analyze the JSON structure (nested objects, arrays, key-value pairs).
	Use semantic HTML5 tags (<main>, <section>, <article>, <header>, <footer>, <nav>, etc.) to create a logical document structure.
	Represent arrays as lists (<ul>, <ol>) or tables (<table>), choosing the most appropriate format for the data.
	Display key-value pairs clearly. The JSON keys should be treated as labels for the data.

Styling (CSS):
	All CSS must be included within a <style> tag in the <head> of the HTML file. Do not use external CSS files.
	Aesthetics: Aim for a clean, modern, and professional look. Use a pleasant color palette, sufficient whitespace, and readable fonts (e.g., from Google Fonts).
	Layout: Use Flexbox or CSS Grid for layout to ensure the page is well-aligned and organized.
	Responsiveness: The layout must be fully responsive and look great on both desktop and mobile browsers. Use media queries (@media) to adjust styles for different screen sizes.
	Visual Elements: Consider using subtle shadows (box-shadow), rounded corners (border-radius), and hover effects to improve user experience. Cards are a great way to display items from a list.

Interactivity (Optional but Recommended JavaScript):
	If the JSON contains deeply nested objects, consider making them collapsible/expandable to avoid clutter.
	All JavaScript must be included within a <script> tag just before the closing </body> tag. Do not use external JS files.
	Keep it simple. Do not add complex logic unless it directly enhances the readability of the data.

Output Format:
	Your final output must be ONLY the raw HTML code and nothing else.
	Do not include any explanations, comments, or markdown formatting (like ```html) around the code.
	The code must be a complete, standalone HTML document starting with <!DOCTYPE html> and ending with </html>.

EXAMPLE TONE & STYLE:
Imagine you are creating a display page for a high-tech dashboard, a polished personal portfolio, or a clean product catalog. The final result should be something you'd be proud to put in your professional portfolio.

Proceed with the conversion.
'''


