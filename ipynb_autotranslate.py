import json
import os
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def translate_ipynb_markdown(file_path, output_path=None, api_key=None):
    """
    读取指定路径的.ipynb文件，使用qwen-turbo翻译其中的markdown单元格为中文
    
    参数:
    file_path (str): .ipynb文件的路径
    output_path (str, optional): 翻译后保存的文件路径，不指定则自动添加_cn后缀
    api_key (str, optional): 通义千问API密钥，默认从环境变量获取
    
    返回:
    dict: 翻译后的notebook字典
    """
    # 如果没有指定输出路径，自动生成带_cn后缀的文件名
    if output_path is None:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(file_dir, f"{name}_cn{ext}")
    
    # 读取.ipynb文件为字典
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # 设置API密钥
    if api_key:
        os.environ['DASHSCOPE_API_KEY'] = api_key
    
    # 系统指令
    system_instruction = "你是一个专业的机器学习教育专家和翻译专家。"
    prompt_question = "将下面 markdown 格式的内容翻译成中文，并且保持所有的格式不变，不要做任何其他多余的事，回答里面不要添加除了原文翻译之外任何其他的内容。有一些专有名词可以不用翻译，保留英文原文: \n"
    
    # 处理每个单元格
    for i, cell in tqdm(enumerate(notebook['cells']), total=len(notebook['cells']), desc="Translating: "):
        if cell['cell_type'] == 'markdown':
            # 获取完整的markdown内容
            markdown_content = ''.join(cell['source'])
            if not markdown_content.strip():
                continue
        
            # print(f"正在翻译第 {i+1} 个markdown单元格...")
            try:
                # 调用模型进行翻译
                completion = client.chat.completions.create(
                    model="qwen-plus-latest",  # 可按需更换模型名称
                    messages=[
                        {'role': 'system', 'content': system_instruction},
                        {'role': 'user', 'content': prompt_question + markdown_content}
                    ],
                )        
                # 获取翻译结果
                translated_text = completion.choices[0].message.content
                # 更新单元格内容
                cell['source'] = [translated_text]
                
            except Exception as e:
                print(f"翻译失败: {e}")
    
    # 保存翻译后的notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print(f"翻译完成，已保存到 {output_path}")
    
    return notebook

import argparse
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='翻译 Jupyter Notebook 文件中的 Markdown 内容')
    parser.add_argument('--input_file', type=str, required=True, help='输入的 Jupyter Notebook 文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    input_file = args.input_file
    translate_ipynb_markdown(input_file)

