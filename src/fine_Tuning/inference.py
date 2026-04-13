import argparse
import sys
import os
from pathlib import Path
import json
import torch
# import swanlab
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    PeftModel,
    prepare_model_for_kbit_training  # 用于量化训练准备
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForSeq2Seq,
    pipeline
)
from datasets import Dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))          # src/fine_Tuning
project_root = os.path.dirname(os.path.dirname(current_dir))     # 项目根目录
sys.path.insert(0, project_root)

# from data import load_json
from config.config import LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,BIAS
from config.config import OUTPUT_DIR,TRAIN_DATA_PATH,TEST_DATA_PATH, SYSTEM_MESSAGE, INSTRUCTION
from config.config import NUM_TRAIN_EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, MAX_LENGTH, MAX_GRAD_NORM, SAVE_STEPS, LOGGING_STEPS, SEED
# from config.config import TRAIN_MODEL_PATH
from config.config import OUTPUT_DATA_PATH


# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
output_dir = os.path.join(project_root, OUTPUT_DIR)
# lora_adapter_path = os.path.join(output_dir, "final_lora_adapter")
model_path = os.path.abspath(os.path.join(project_root, "./model/Qwen2.5-7B-Instruct"))
# test_data_path = os.path.join(project_root, TEST_DATA_PATH)

class QwenLoraVLLMInference:
    def __init__(self, base_model_path, lora_path, tensor_parallel_size=1, max_model_len=512):
        """
        :param base_model_path: 基础模型路径
        :param lora_path: LoRA适配器路径
        :param tensor_parallel_size: GPU数量（单卡设为1）
        :param max_model_len: 最大序列长度
        """
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        # 初始化 vLLM 模型
        self.llm = LLM(
            model=base_model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_lora=True,          # 启用LoRA
        )
        # 加载 tokenizer（vLLM 内部也有 tokenizer，但我们需要应用 chat_template）
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 构建 LoRA 请求对象
        self.lora_request = LoRARequest("qwen_lora", 1, lora_path)

    def apply_chat_template(self, question):
        """将单个问题应用对话模板，返回 prompt 字符串"""
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def batch_generate(self, questions, max_new_tokens=512, temperature=0.0, top_p=0.9):
        """
        批量生成
        :param questions: 问题列表
        :return: 生成的回答列表
        """
        # 将所有问题转换为 prompt
        prompts = [self.apply_chat_template(q) for q in questions]

        # 设置采样参数
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p if temperature > 0 else 1.0,  # 当 temperature=0 时，top_p 无效
            stop_token_ids=[self.tokenizer.eos_token_id],
            repetition_penalty=1.1,
        )

        # 使用 vLLM 生成，并指定 LoRA
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self.lora_request
        )
        # 提取生成的文本（只取新生成部分，vLLM 返回完整文本，需要减去 prompt 长度？）
        # vLLM 的 output.outputs[0].text 默认是去掉 prompt 的，直接是生成的回复
        responses = [output.outputs[0].text.strip() for output in outputs]
        return responses


def inference_main(test_data_path, lora_model_path, output_json_path):
    # 加载测试数据
    print("加载测试数据...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # 初始化推理引擎
    print("初始化推理引擎...")
    infer = QwenLoraVLLMInference(
        base_model_path=model_path,
        lora_path=lora_model_path,
        tensor_parallel_size=1,       # 单卡 A10，设为1即可
        max_model_len=2048            # 根据你的最大长度调整
    )

    # 提取所有问题
    print("提取所有问题...")
    # data_list = data_list[:5]
    
    dataid = [item["id"] for item in data_list]
    questions = [item["content"] for item in data_list]
    prompts = [item["prompt"] for item in data_list]
    ground_truths = [item.get("output", "") for item in data_list]

    # 批量推理（可设置批次大小，vLLM 内部会自动分批，也可以自己分批）
    # print("批量推理...")
    # batch_size = 8   # 根据显存调整
    # results = []
    # for i in tqdm(range(0, len(questions), batch_size), desc="批量推理"):
    #     batch_questions = questions[i:i+batch_size]
    #     batch_preds = infer.batch_generate(batch_questions, max_new_tokens=512, temperature=0.0)
    #     for j, pred in enumerate(batch_preds):
    #         idx = i + j
    #         results.append({
    #             "id": dataid[idx],
    #             "content": questions[idx],
    #             "ground_truth": ground_truths[idx],
    #             "output": pred
    #         })
    print("批量推理...")
    batch_size = 8
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="批量推理"):
        batch_prompts = prompts[i:i+batch_size]
        batch_preds = infer.batch_generate(batch_prompts, max_new_tokens=512, temperature=0.0)
        for j, pred in enumerate(batch_preds):
            idx = i + j
            results.append(
                {
                    "id": dataid[idx],
                    "content": questions[idx],
                    "ground_truth": ground_truths[idx],
                    "output": pred
                }
            )

    # 保存结果
    print("保存结果...")
    data_output_dir = os.path.join(project_root, OUTPUT_DATA_PATH)
    save_path = os.path.join(data_output_dir, output_json_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果保存至 {save_path}")

