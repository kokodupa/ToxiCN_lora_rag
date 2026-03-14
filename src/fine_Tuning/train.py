import argparse
import os
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import Dataset
from data import load_json


INSTRUCTION = """
    请从下面的文本中抽取所有仇恨言论四元组，
    格式为：评论对象|对象观点|仇恨群体|是否仇恨，
    仇恨群体类别包括（LGBTQ、Region、Sexism、Racism、others、non-hate），
    是否仇恨为（hate、non-hate），
    多个四元组用 [SEP] 分隔，末尾加 [END]。\n
    以下是文本：
"""


train_json = load_json("../../data/train.json")
train_data = []
for d in train_json:
    train_data.append({
        "input": d["content"], 
        "output": d["output"]
        })
dataset = Dataset.from_list(train_data)

#分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
special_tokens = ["[SEP]", "[END]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

def preprocess(example):
    texts = []
    for inp, out in zip(example["input"], example["output"]):
        user_content = INSTRUCTION + inp
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    
    model_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=2048,      
        padding=False,    
        return_tensors=None 
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

encoded_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names 
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))


    