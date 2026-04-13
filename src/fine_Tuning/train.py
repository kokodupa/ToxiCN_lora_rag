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
    pipeline,
    EarlyStoppingCallback
) 
from datasets import Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))          # src/fine_Tuning
project_root = os.path.dirname(os.path.dirname(current_dir))     # 项目根目录
sys.path.insert(0, project_root)
from config.config import LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,BIAS,LR_SCHEDULER_TYPE,WEIGHT_DECAY
from config.config import OUTPUT_DIR,TRAIN_DATA_PATH,TEST_DATA_PATH, SYSTEM_MESSAGE, INSTRUCTION
from config.config import NUM_TRAIN_EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, MAX_LENGTH, MAX_GRAD_NORM, SAVE_STEPS, LOGGING_STEPS, SEED

    

def preprocess_function(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_tokenized = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding=False, return_tensors=None)
    prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH, padding=False, return_tensors=None)

    input_ids = full_tokenized["input_ids"]
    prompt_len = len(prompt_tokenized["input_ids"])

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels
    }



"""
加载模型
"""
def load_model_and_tokenizer():
    print("开始加载模型和分词器...")
    model_path = os.path.abspath(os.path.join(project_root, "./model/Qwen2.5-7B-Instruct"))

    #加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 填充侧设置为右侧

    bnb_config = None
    torch_dtype = torch.bfloat16

    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配到可用设备
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_cache=False,  # 训练时关闭缓存以节省显存
    )

    # 6. 配置LoRA
    print("配置LoRA适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias=BIAS,
        modules_to_save=None,  # 可指定需要全参数训练的模块
    )

    # 7. 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    
    # 8. 启用梯度检查点（进一步节省显存）
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # 9. 打印可训练参数信息
    model.print_trainable_parameters()
    
    print("模型和分词器加载完成！")
    print("=" * 50)
    
    return model, tokenizer


def train_model(model, tokenizer, train_dataset, eval_dataset, lora_save_path):
    print("开始配置训练参数...")
    output_path = os.path.join(project_root, OUTPUT_DIR)
    # 训练参数配置
    training_args = TrainingArguments(
        # 输出目录
        output_dir= output_path,

        # 训练参数
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,

        # 优化器设置
        optim = 'adamw_torch',
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.03,  # 预热比例
        lr_scheduler_type= LR_SCHEDULER_TYPE, 
        
        # 评估与保存策略
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_total_limit=3,             # 最多保存3个检查点
        load_best_model_at_end=True,    # 训练结束后加载最佳模型
        metric_for_best_model="eval_loss",  # 根据验证损失选择最佳模型
        
        # 日志与报告
        logging_strategy="steps",       # 保持不变
        
        # 精度与硬件优化
        bf16=torch.cuda.is_bf16_supported(),  # 4090支持bf16
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,    # 梯度检查点节省显存
        
        #其他
        dataloader_num_workers=4,
        remove_unused_columns=True,
        group_by_length=True,           # 按长度分组提高效率
        dataloader_pin_memory=True,
        
        # 新版本可能需要添加的额外参数
        logging_first_step=True,        # 记录第一步的日志
        greater_is_better=False,        # eval_loss越小越好
        
    )
    
    # 数据收集器（动态填充）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # 对齐到8的倍数，GPU效率更高
        return_tensors="pt"
    )
    sample = eval_dataset[0]
    print(sample)
    print("input_ids:", sample["input_ids"][:50])
    print("labels:", sample["labels"][:50])
    print("有效 token 数:", sum(1 for l in sample["labels"] if l != -100))
    # 统计验证集中有效 token 为 0 的样本
    zero_valid_samples = []
    for idx, sample in enumerate(eval_dataset):
        valid_count = sum(1 for l in sample["labels"] if l != -100)
        if valid_count == 0:
            zero_valid_samples.append((idx, valid_count, sample["input_ids"][:10]))  # 记录索引和部分input_ids

    print(f"验证集总样本数: {len(eval_dataset)}")
    print(f"有效 token 为 0 的样本数: {len(zero_valid_samples)}")
    if zero_valid_samples:
        print(f"首个问题样本索引: {zero_valid_samples[0][0]}")
        # 可选：打印该样本的原始输入和输出
        orig_idx = zero_valid_samples[0][0]
        print("原始 prompt:", raw_eval[orig_idx]["input"][:200])
        print("原始 output:", raw_eval[orig_idx]["output"][:200])
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0))
    
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型（LoRA适配器）
    print("保存LoRA适配器...")
    model.save_pretrained(os.path.join(output_path, lora_save_path))
    tokenizer.save_pretrained(os.path.join(output_path, lora_save_path))
    
    # 保存训练历史
    trainer.save_model(os.path.join(output_path, lora_save_path,'train'))
    trainer.save_state()
    
    print(f"训练完成！模型保存在: {os.path.join(output_path, lora_save_path,'train')}")
    
    return trainer


def train_main(train_data_path, lora_save_path):
    print("=" * 60)
    print("Qwen3-1.7B LoRA微调脚本")
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print("=" * 60)
    
    print("\n[步骤2/5] 加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer()

    print("加载原始数据...")

    with open(train_data_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    formatted = []
    # for item in data_list:
    #     formatted.append({
    #         "input": item.get("content", ""),
    #         "output": item.get("output", "")
    #     })
    for item in data_list:
        formatted.append({
            "input": item.get("prompt", ""),
            "output": item.get("output", "")
        })

    raw_dataset = Dataset.from_list(formatted)
    print(f"原始样本数量: {len(raw_dataset)}")

    # 划分训练集和验证集
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=SEED)
    raw_train = split_dataset["train"]
    raw_eval = split_dataset["test"]
    print(f"训练集样本数: {len(raw_train)}，验证集样本数: {len(raw_eval)}")
    print(raw_train[0])
    # 分别做预处理
    train_dataset = raw_train.map(
        lambda example: preprocess_function(example, tokenizer),
        remove_columns=raw_train.column_names,
        batched=False
    )
    eval_dataset = raw_eval.map(
        lambda example: preprocess_function(example, tokenizer),
        remove_columns=raw_eval.column_names,
        batched=False
    )
    print("\n[步骤4/5] 训练LoRA适配器...")
    train_model(model, tokenizer, train_dataset, eval_dataset, lora_save_path)



    