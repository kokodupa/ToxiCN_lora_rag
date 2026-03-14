import pandas as pd
import os
from openai import OpenAI
import time
import requests
import json
import io
import sys
from transformers import pipeline

from data import load_json
from get_prompts import build_prompt
from data import str_to_quadruples
from evaluate import evaluate
import yaml
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_config(config_path='../config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config
  

def model(config, message):
    url = config['api']['url']
    api_key = config['api']['key']
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": config['api']['model'],
        "messages": message,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        json = response.json()
        return json["choices"][0]["message"]["content"]
    else :
        return "Error:", response.status_code, response.text
    

def run_inference(type):
    test_data = load_json("data/test.json")

    config = load_config()
    gold_data = []
    pred_data = []
    for d in test_data:
        content = d["content"]
        id = d["id"]
        gold_quads = d["quadruples"]
        gold_data.append(
            {"id": id, 
             "content": content, 
             "quadruples": gold_quads,
             })
        
        message = build_prompt(content, type)
        try:
            pred_output = model(config, message)
        except Exception as exc:
            print(f"  错误：{exc}")  
            response_text = ""

        pred_quads = str_to_quadruples(pred_output)
        pred_data.append(
            {"id": id, 
             "content": content, 
             "quadruples": pred_quads,
             })
        with open("", "w", encoding="utf-8") as f:
            json.dump(pred_data, f, ensure_ascii=False, indent=2)
    
    return evaluate(gold_data, pred_data)


    
    
    


