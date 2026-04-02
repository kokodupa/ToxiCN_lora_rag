import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from data import load_json, str_to_quadruples
from evaluate import evaluate
from prompt_engineering.get_prompts import build_prompt

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "../../model/Qwen2.5-7B-Instruct"))

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

def qwen_model_batch(messages_list):
    prompts = []
    for msg in messages_list:
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    outputs = llm.generate(prompts, sampling_params)
    responses = [out.outputs[0].text for out in outputs]
    return responses

# 删除了未使用的 qwen_model 和 api_model 函数（如有需要可保留，但需修正）

def run_inference(mode, batch_size=4):
    test_data = load_json("../data/test_data.json")
    # test_data = test_data[:6]  # 如果需要全量数据，请注释掉这一行
    gold_data = [{"id": d["id"], "content": d["content"], "quadruples": d["quadruples"]} for d in test_data]
    pred_data = []
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        messages_batch = [build_prompt(d["content"], mode) for d in batch]
        responses = qwen_model_batch(messages_batch)
        for d, resp in zip(batch, responses):
            pred_quads = str_to_quadruples(resp)
            pred_data.append({
                "id": d["id"],
                "content": d["content"],
                'output': resp,
                "quadruples": pred_quads
            })
    
        # 全部处理完后一次性保存
        with open(f"../data/{mode}_prompt_output.json", "w", encoding="utf-8") as f:
            json.dump(pred_data, f, ensure_ascii=False, indent=2)
    
    return evaluate(gold_data, pred_data)

# "零样本"
# zero_results = run_inference(mode = '1')
# print(f"Hard Score: {zero_results['hard_f1']:.4f}")
# print(f"Soft Score: {zero_results['soft_f1']:.4f}")
# print(f"Score: {zero_results['avg_f1']:.4f}")