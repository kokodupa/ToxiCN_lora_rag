#调用api
BASE_URL = ''
API_KEY = ''
MODEL_NAME = 'gpt-4o'

#lora
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]
BIAS = "none"

#训练参数
NUM_TRAIN_EPOCHS = 10
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 32
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 500
LOGGING_STEPS = 50
SEED = 42
MAX_LENGTH = 2048
#路径
OUTPUT_DIR = 'output'
OUTPUT_DATA_PATH = 'data/output'
TRAIN_DATA_PATH = 'data/train_data.json'
TEST_DATA_PATH = 'data/test_data.json'

#提示词
SYSTEM_MESSAGE = "你是一个内容审查专家，擅长分析中文仇恨言论,擅长从文本中提取所有仇恨言论四元组。"
INSTRUCTION = """
    请从下面的文本中抽取所有仇恨言论四元组，
    格式为：评论对象|对象观点|仇恨群体|是否仇恨，
    其中：
    评论对象：文本中被评论的实体；
    对象观点：针对评论对象的核心观点；
    仇恨群体：只能从以下类别选择（LGBTQ、Region、Sexism、Racism、others、non-hate），可以选择一个或者多个，多个时使用逗号分割；
    是否仇恨为（hate、non-hate），
    多个四元组用 [SEP] 分隔，末尾加 [END]。
    以下是文本：
"""

# SwanLab配置
SWANLAB_PROJECT = "qwen-medical-lora"
SWANLAB_RUN_NAME = "qwen2.5-7B-Instruct-lora-experiment"

#RAG配置
RAG_DATA_PATH = "data/annotated lexicon.json"
RAG_SYSTEM_MESSAGE = ""
RAG_INSTRUCTION = """"""