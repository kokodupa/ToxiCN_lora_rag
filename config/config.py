#调用api
BASE_URL = ''
API_KEY = ''
MODEL_NAME = 'gpt-4o'

#lora
LORA_R = 8                
LORA_ALPHA = 16           
LORA_DROPOUT = 0.1         
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]  
BIAS = "none"

#训练参数
NUM_TRAIN_EPOCHS = 4 #5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 500
LOGGING_STEPS = 50
SEED = 42
# MAX_LENGTH = 512
MAX_LENGTH = 1024
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"      # 余弦退火
EARLY_STOPPING_PATIENCE = 3
#路径
OUTPUT_DIR = 'output'
OUTPUT_DATA_PATH = 'data/output'
TRAIN_DATA_PATH = 'data/train_data.json'
TEST_DATA_PATH = 'data/test_data.json'

#提示词
SYSTEM_MESSAGE = "你是一个内容审查专家，擅长分析中文仇恨言论,擅长从文本中提取所有仇恨言论四元组。"
INSTRUCTION = (
    "请从下面的文本中抽取一个或多个四元组。\n"
    "每个四元组的格式为：评论对象|对象观点|仇恨群体|是否仇恨\n"
    "严格要求：\n"
    "  1. 无论文本是否包含仇恨言论，必须至少输出一个四元组，不能只输出 [END] 或其它解释性文字。\n"
    "  2. 四元组中四个字段都必须出现，用英文竖线 | 分隔，不能缺少任意一项，同时不要使用换行分割。\n"
    "  3. 如果文本不包含仇恨言论，则“仇恨群体”和“是否仇恨”都填 non-hate。\n"
    "  4. 只输出四元组本身，不要输出任何解释性文字（例如“没有明显仇恨言论”之类的话）。\n"
    "  5. 最后一个四元组后加 [END]，如果存在多个四元组，采用[SEP]分隔，如果只有一个四元组，不需要使用 [SEP]。整个输出必须只占一行，输出中绝对不能包含换行符或回车符；如果有多个四元组，一律写在同一行.\n"
    "字段含义说明：\n"
    "  - 评论对象：文本中被评论的实体，例如“这人”“我”。\n"
    "  - 对象观点：针对评论对象的核心观点或评价，例如“有毛病”“含义深重”。\n"
    "  - 仇恨群体：只能从以下类别选择（LGBTQ、Region、Sexism、Racism、others、non-hate），可以选择一个或者多个，多个时使用英文逗号分割。\n"
    "  - 是否仇恨：只能填 hate 或 non-hate。\n"
)
GROUP_EXAMPLES = """
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

#RAG配置
RAG_DATA_PATH = "data/annotated lexicon.json"
RAG_SYSTEM_MESSAGE = ""
RAG_INSTRUCTION = """"""