import json
from pathlib import Path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))          # src/fine_Tuning
project_root = os.path.dirname(os.path.dirname(current_dir))     # 项目根目录
sys.path.insert(0, project_root)

from config.config import SYSTEM_MESSAGE,INSTRUCTION
train_data_path = os.path.join(project_root, "data/train_data.json")
GROUPS = ["LGBTQ", "Region", "Sexism", "Racism", "others", "non-hate"]
# SYSTEM_MESSAGE = "你是一个内容审查专家，擅长分析中文仇恨言论,擅长从文本中提取所有仇恨言论四元组。"

# INSTRUCTION =(
#     "请从下面的文本中抽取一个或多个四元组。\n"
#     "每个四元组的格式为：评论对象|对象观点|仇恨群体|是否仇恨\n"
#     "严格要求：\n"
#     "  1. 无论文本是否包含仇恨言论，必须至少输出一个四元组，不能只输出 [END] 或其它解释性文字。\n"
#     "  2. 四元组中四个字段都必须出现，用英文竖线 | 分隔，不能缺少任意一项，同时不要使用换行分割。\n"
#     "  3. 如果文本不包含仇恨言论，则“仇恨群体”和“是否仇恨”都填 non-hate。\n"
#     "  4. 只输出四元组本身，不要输出任何解释性文字（例如“没有明显仇恨言论”之类的话）。\n"
#     "  5. 最后一个四元组后加 [END]，如果存在多个四元组，采用[SEP]分隔，如果只有一个四元组，不需要使用 [SEP]。整个输出必须只占一行，输出中绝对不能包含换行符或回车符；如果有多个四元组，一律写在同一行.\n"
#     "字段含义说明：\n"
#     "  - 评论对象：文本中被评论的实体，例如“这人”“我”。\n"
#     "  - 对象观点：针对评论对象的核心观点或评价，例如“有毛病”“含义深重”。\n"
#     "  - 仇恨群体：只能从以下类别选择（LGBTQ、Region、Sexism、Racism、others、non-hate），可以选择一个或者多个，多个时使用英文逗号分割。\n"
#     "  - 是否仇恨：只能填 hate 或 non-hate。\n"
# )

GROUP_EXAMPLES_ONE = """输出：你|坦白吧，放过人家|non-hate|non-hate [END]"""

GROUP_EXAMPLES_SIX = """
1、输入：我的天啊，求求你坦白吧，放过人家
   输出：你|坦白吧，放过人家|non-hate|non-hate [END]
2、输入：我190+，但是这种墓垢还是铁鲨鼻
   输出：墓垢|铁鲨鼻|Sexism|hate [END]
3、输入：笑死，要是刷男女的没人有意见，真就高贵的同性恋呗??
   输出：同性恋|高贵|LGBTQ|hate [END]
4、输入：钓鱼的吧，除非让你全家女性照顾黑人
   输出：你全家女性|照顾黑人|Sexism, Racism|hate [END]
5、输入：广州就是黑河南、黑北方的总部
   输出：广州|黑河南、黑北方的总部|Region|hate [END]
6、输入：无艾不是真基
   输出：基|无艾不是真基|LGBTQ, others|hate [END]

"""
GROUP_EXAMPLES_TWELVE = """
1、输入:我的天啊，求求你坦白吧，放过人家
   输出:你|坦白吧，放过人家|non-hate|non-hate [END]
2、输入:有意思~~请问你哪里得出结论我是山东人~~
   输出:你|哪里得出结论我是山东人|non-hate|non-hate [END]
3、输入:我190+，但是这种墓垢还是铁鲨鼻
   输出:墓垢|铁鲨鼻|Sexism|hate [END]
4、输入:更何况你鬼母阿娜本身就是个二年级辍学的屑
   输出:你鬼母阿娜|二年级辍学的屑|Sexism|hate [END]
5、输入:笑死，要是刷男女的没人有意见，真就高贵的同性恋呗??
   输出:同性恋|高贵|LGBTQ|hate [END]
6、输入:反社会分子别冒充同性恋了
   输出:反社会分子|别冒充同性恋了|LGBTQ|hate [END]
7、输入:钓鱼的吧，除非让你全家女性照顾黑人
   输出:你全家女性|照顾黑人|Sexism, Racism|hate [END]
8、输入:这些黑猴子也真不怕死。
   输出:这些黑猴子|也真不怕死|Racism|hate [END]
9、输入:广州就是黑河南、黑北方的总部
   输出:广州|黑河南、黑北方的总部|Region|hate [END]
10、输入:贵州人表示在浙江贵州人的身份经常感受到来自浙江人的恶意
    输出:贵州人|经常感受到来自浙江人的恶意|Region|hate [END]
11、输入:无艾不是真基
    输出:基|无艾不是真基|LGBTQ, others|hate [END]
12、输入:他们宣传他们的，咱们抵制就好了
    输出:他们|咱们抵制就好了|others|hate [END]
"""
def json_load(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def zero_prompts(content):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": INSTRUCTION + "\n请分析以下文本：" + content},
    ]

def few_shot_prompts(content, example_num):
    if example_num == 1:
        group_examples = GROUP_EXAMPLES_ONE
    elif example_num == 6:
        group_examples = GROUP_EXAMPLES_SIX
    else:
        group_examples = GROUP_EXAMPLES_TWELVE

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": INSTRUCTION + "\n以下是一些示例\n" + group_examples + " \n请分析以下文本：" + content}
        ]

# def build_cot_prompt(content: str) -> list[dict[str, str]]:
#     user_content = (
#         INSTRUCTION +
#         "\n请按以下步骤分析进行，不需要输出分析内容：\n"
#         "步骤1：判断文本是否包含仇恨言论。\n"
#         "步骤2：识别文本中涉及的目标群体或实体。\n"
#         "步骤3：提取针对每个目标的具体观点。\n"
#         "步骤4：对每个目标确定仇恨群体类别和是否仇恨。\n"
#         "步骤5：按照格式输出最终四元组。\n\n"
#         f'待分析文本："{content}"'
#     )
#     return [
#         {"role": "system", "content": SYSTEM_MESSAGE},
#         {"role": "user", "content": user_content},
#     ]
    

def build_prompt(content, mode, num = 0):
    if mode == "1":
        return zero_prompts(content)
    else:
        return few_shot_prompts(content, num)
    
