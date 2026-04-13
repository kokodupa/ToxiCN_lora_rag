import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))          # src/fine_Tuning
project_root = os.path.dirname(os.path.dirname(current_dir))     # 项目根目录
sys.path.insert(0, project_root)
from src.data import load_json
from config.config import SYSTEM_MESSAGE
data_path = os.path.join(project_root, "data")
data_output_path = os.path.join(project_root, "data/output")
model_path = os.path.abspath(os.path.join(project_root, "model/Qwen2.5-7B-Instruct"))

llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    dtype="float16",
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=False,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)


def message_build(prompt):
    
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt}
        ]
def qwen_model_batch(messages_list):
    prompts = []
    for msg in messages_list:
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    # print(prompts)
    outputs = llm.generate(prompts, sampling_params)
    responses = [out.outputs[0].text for out in outputs]
    return responses

# 删除了未使用的 qwen_model 和 api_model 函数（如有需要可保留，但需修正）

def run_inference(test_data_path, save_path, batch_size=4):
    # test_data_path = os.path.join(data_path, data_name)
    print(f"测试数据路径{test_data_path}")

    test_data = load_json(test_data_path)
    # test_data = test_data[:6]  # 如果需要全量数据，请注释掉这一行
    pred_data = []
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        messages_batch = [message_build(d['prompt']) for d in batch]
        # print(messages_batch)
        responses = qwen_model_batch(messages_batch)
        for d, resp in zip(batch, responses):
            # pred_quads = str_to_quadruples(resp)
            pred_data.append({
                "id": d["id"],
                "content": d["content"],
                'ground_truth': d['output'],
                'output': resp,
            })
    
        # 全部处理完后一次性保存
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False, indent=2)

# test_data_path = os.path.join(data_path, "rag_self_test.json")
# save_data_path = os.path.join(data_path, "output", "result_base_self_train.json")
# run_inference(test_data_path, save_data_path, batch_size=8)
    
