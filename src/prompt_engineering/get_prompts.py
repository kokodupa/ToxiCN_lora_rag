import json
from pathlib import Path


SYSTEM_MESSAGE = (
    "你是一个内容审查专家，擅长分析中文仇恨言论。"
    "请从文本中提取所有仇恨言论四元组。"
)

TASK_DESCRIPTION = (
    "请从下面的文本中抽取一个或多个四元组。\n"
    "每个四元组的格式为：评论对象|对象观点|仇恨群体|是否仇恨\n"
    "说明：\n"
    "  - 评论对象：文本中被评论的实体，若无则填 NULL\n"
    "  - 对象观点：针对评论对象的核心观点，尽量简洁\n"
    "  - 仇恨群体：只能从以下类别选择（LGBTQ、Region、Sexism、Racism、others、non-hate）\n"
    "  - 是否仇恨：只能填 hate 或 non-hate\n"
    "多个四元组之间用 [SEP] 分隔，最后一个四元组后加 [END]。\n"
)

GROUPS = ["LGBTQ", "Region", "Sexism", "Racism", "others", "non-hate"]


def _load_group_examples() -> dict[str, dict[str, str]]:
    """Return one dataset example per group to fuel few-shot prompting."""
    data_path = Path(__file__).resolve().parents[2] / "data" / "train_output.json"
    if not data_path.exists():
        return {}

    try:
        dataset = json.loads(data_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    examples: dict[str, dict[str, str]] = {}
    for record in dataset:
        content = record.get("content", "")
        for quad in record.get("quadruples", []):
            group = quad.get("group")
            if group not in GROUPS or group in examples:
                continue
            target = quad.get("target", "NULL") or "NULL"
            argument = quad.get("argument", "NULL") or "NULL"
            hateful = quad.get("hateful", "non-hate") or "non-hate"
            formatted_output = f"{target}|{argument}|{group}|{hateful} [END]"
            examples[group] = {"input": content, "output": formatted_output}
            if len(examples) == len(GROUPS):
                return examples
    return examples


GROUP_EXAMPLES = _load_group_examples()


def zero_prompts(content):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": TASK_DESCRIPTION + "\n请分析以下文本：" + content},
    ]

def few_shot_prompts(content):
    if not GROUP_EXAMPLES:
        return zero_prompts(content)

    example_text = ""
    example_index = 1
    for group in GROUPS:
        example = GROUP_EXAMPLES.get(group)
        if not example:
            continue
        example_text += (
            f"示例 {example_index}（{group}）：\n"
            f"输入：{example['input']}\n"
            f"输出：{example['output']}\n\n"
        )
        example_index += 1
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": TASK_DESCRIPTION + "\n以下是一些示例\n" + example_text + "请分析以下文本：" + content}
        ]

def build_cot_prompt(content: str) -> list[dict[str, str]]:
    user_content = (
        TASK_DESCRIPTION
        + "\n请按以下步骤分析：\n"
        "步骤1：判断文本是否包含仇恨言论，说明理由。\n"
        "步骤2：识别文本中涉及的目标群体或实体。\n"
        "步骤3：提取针对每个目标的具体观点。\n"
        "步骤4：对每个目标确定仇恨群体类别和是否仇恨。\n"
        "步骤5：按照格式输出最终四元组。\n\n"
        f'待分析文本："{content}"'
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def build_prompt(content, type):
    if type == "1":
        return zero_prompts(content)
    elif type == 2:
        return few_shot_prompts(content)
    else:
        return build_cot_prompt(content)
