import json
from pathlib import Path
import os
import sys

current_dir = Path.cwd() 
project_root = os.path.dirname(os.path.dirname(current_dir))     # 项目根目录
sys.path.insert(0, project_root)

from config.config import INSTRUCTION, SYSTEM_MESSAGE

def rag_lexicon_prompt(content, contexts):
    prompt = INSTRUCTION +"\n以下是要分析的文本：" + content + "\n相关知识：" + contexts + "输出格式示例：你|坦白吧，放过人家|non-hate|non-hate [END]"
    return prompt

def rag_selftrain_prompt(content, contexts):
    example_content = contexts['content']
    example_output = contexts['output']
    prompt =  INSTRUCTION+ "\n示例：" + "\n输入：" +example_content +"\n输出：" + example_output + "\n以下是要分析的文本：" + content 
    return prompt