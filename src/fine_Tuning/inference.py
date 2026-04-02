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
lora_adapter_path = os.path.join(output_dir, "final_lora_adapter")
model_path = os.path.abspath(os.path.join(project_root, "./model/Qwen2.5-7B-Instruct"))
test_data_path = os.path.join(project_root, TEST_DATA_PATH)

class QwenLoraVLLMInference:
    def __init__(self, base_model_path, lora_path, tensor_parallel_size=1, max_model_len=2048):
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
            {"role": "system", "content": SYSTEM_MESSAGE + "\n" + INSTRUCTION},
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


if __name__ == "__main__":
    # 加载测试数据
    print("加载测试数据...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # 初始化推理引擎
    print("初始化推理引擎...")
    infer = QwenLoraVLLMInference(
        base_model_path=model_path,
        lora_path=lora_adapter_path,
        tensor_parallel_size=1,       # 单卡 A10，设为1即可
        max_model_len=2048            # 根据你的最大长度调整
    )

    # 提取所有问题
    print("提取所有问题...")
    dataid = [item["id"] for item in data_list]
    questions = [item["content"] for item in data_list]
    ground_truths = [item.get("output", "") for item in data_list]

    # 批量推理（可设置批次大小，vLLM 内部会自动分批，也可以自己分批）
    print("批量推理...")
    batch_size = 8   # 根据显存调整
    results = []
    for i in tqdm(range(0, len(questions), batch_size), desc="批量推理"):
        batch_questions = questions[i:i+batch_size]
        batch_preds = infer.batch_generate(batch_questions, max_new_tokens=512, temperature=0.0)
        for j, pred in enumerate(batch_preds):
            idx = i + j
            results.append({
                "id": dataid[idx],
                "content": questions[idx],
                "ground_truth": ground_truths[idx],
                "output": pred
            })

    # 保存结果
    print("保存结果...")
    data_output_dir = os.path.join(project_root, OUTPUT_DATA_PATH)
    save_path = os.path.join(data_output_dir, "test_results_vllm.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果保存至 {save_path}")

#版本2
# class QwenLoraInference:
#     def __init__(self, base_model_path=base_model_path, lora_path=lora_adapter_path):
#         print(f"加载基础模型: {base_model_path}")
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             base_model_path, use_fast=False, trust_remote_code=True
#         )
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_path,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#         )
#         self.model = PeftModel.from_pretrained(base_model, lora_path, device_map="auto")
#         self.model.eval()
#         print("LoRA 适配器加载完成")

#     def generate_response(self, question, max_new_tokens=300, temperature=0.0, do_sample=False):
#         # 注意：必须与训练时的 system 消息一致
#         messages = [
#             {"role": "system", "content": SYSTEM_MESSAGE + "\n" + INSTRUCTION},
#             {"role": "user", "content": question}
#         ]
#         text = self.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

#         with torch.inference_mode():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 do_sample=do_sample,
#                 top_p=0.9 if do_sample else None,
#                 repetition_penalty=1.1,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 pad_token_id=self.tokenizer.pad_token_id,
#             )
#         # 只解码新生成的部分
#         input_len = inputs.input_ids.shape[1]
#         generated = outputs[0][input_len:]
#         response = self.tokenizer.decode(generated, skip_special_tokens=True)
#         return response.strip()

# if __name__ == "__main__":
#     # 加载测试数据
#     test_data_path = os.path.join(project_root, TEST_DATA_PATH)  
#     with open(test_data_path, "r", encoding="utf-8") as f:
#         data_list = json.load(f)

#     infer = QwenLoraInference()
#     results = []
#     for item in tqdm(data_list, desc="推理中"):
#         question = item["content"]
#         pred = infer.generate_response(question)
#         results.append({
#             "content": question,
#             "ground_truth": item.get("output", ""),
#             "prediction": pred
#         })

#     # 保存结果
#     save_path = os.path.join(output_dir, "test_results.json")
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)
#     print(f"结果保存至 {save_path}")

#版本1
# output_path = os.path.join(project_root, OUTPUT_DIR)

# class QwenLoraInference:
#     """LoRA微调后的推理类"""
    
#     def __init__(self, base_model_name: str = None, lora_adapter_path: str = None):
#         """
#         初始化推理模型
        
#         参数:
#             base_model_name: 基础模型名称或路径
#             lora_adapter_path: LoRA适配器路径
#         """
#         base_model_name = os.path.abspath(os.path.join(project_root, "./model/Qwen2.5-7B-Instruct"))

#         if lora_adapter_path is None:
#             lora_adapter_path = os.path.join(output_path, "final_lora_adapter")
        
#         print(f"加载基础模型: {base_model_name}")
#         print(f"加载LoRA适配器: {lora_adapter_path}")
        
#         # 加载基础模型
          
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             base_model_name,
#             use_fast=False,
#             trust_remote_code=True
#         )
        
#         self.base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#         )
        
#         # 加载LoRA适配器
#         self.model = PeftModel.from_pretrained(
#             self.base_model,
#             lora_adapter_path,
#             device_map="auto"
#         )
        
#         # 设置为评估模式
#         self.model.eval()
#         self.base_model.eval()
#         print("推理模型加载完成！")
    
#     def generate_response(self,question: str, model: str = 'ft', max_new_tokens: int = 1324,temperature: float = 0.7,do_sample: bool = True) -> str:
#         """
#         生成回答
        
#         参数:
#             question: 用户问题
#             max_new_tokens: 最大生成token数
#             temperature: 温度参数（控制随机性）
#             do_sample: 是否使用采样
        
#         返回:
#             模型生成的回答
#         """
#         model = self.model if model=='ft' else self.base_model
#         # 构建对话
#         messages = [
#             {"role": "system", "content": SYSTEM_MESSAGE + "\n" + INSTRUCTION},
#             {"role": "user", "content": question}
#         ]
        
#         # 应用聊天模板
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
        
#         # 编码输入
#         inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        
#         # 生成参数
#         generation_config = {
#             "max_new_tokens": max_new_tokens,
#             "temperature": temperature,
#             "do_sample": do_sample,
#             "top_p": 0.9,
#             "repetition_penalty": 1.1,
#             "eos_token_id": self.tokenizer.eos_token_id,
#             "pad_token_id": self.tokenizer.pad_token_id,
#         }
        
#         # 生成回答
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 **generation_config
#             )
        
#         # 解码并提取助手回复
#         full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # 提取助手部分（从"assistant"标签后开始）
#         assistant_marker = "<|im_start|>assistant\n"
#         if assistant_marker in full_response:
#             response = full_response.split(assistant_marker)[-1]
#         else:
#             response = full_response
        
#         return response.strip()
    
#     def batch_predict(self, questions, max_new_tokens= 512) :
#         """
#         批量预测
        
#         参数:
#             questions: 问题列表
#             max_new_tokens: 最大生成token数
        
#         返回:
#             回答列表
#         """
#         responses = []
#         for i, question in enumerate(questions):
#             print(f"处理问题 {i+1}/{len(questions)}: {question[:50]}...")
#             response = self.generate_response(question, max_new_tokens)
#             responses.append(response)
#         return responses
    

# if __name__ == "__main__":
#     inference_model = QwenLoraInference()

#     print("加载测试数据...")
#     test_data_path = os.path.join(project_root, TEST_DATA_PATH)
#     with open(test_data_path, "r", encoding="utf-8") as f:
#         data_list = json.load(f)

#     test_questions = [item["content"] for item in data_list]

#     print("\n测试推理结果:")
#     print("-" * 50)
    
#     test_results = []
#     for i, question in enumerate(test_questions):
#         print(f"\n问题 {i+1}: {question}")
#         response = inference_model.generate_response(question, max_new_tokens=300)
#         print(f"回答: {response}")
        
#         # 记录到SwanLab
#         test_results.append({
#             "content": question,
#             "output": response
#         })

#     with open(os.path.join(output_path, "test_results.json"), "w", encoding="utf-8") as f:
#         json.dump(test_results, f, ensure_ascii=False, indent=2)
    
#     print(f"\n测试结果已保存到: {os.path.join(output_path, 'test_results.json')}")