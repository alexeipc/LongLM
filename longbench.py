# transfromers version 4.38.2
# this example is tested with 4 RTX3090s, 24GB memory each
# Edit 1.0
import re
import string
from collections import Counter
from fuzzywuzzy import fuzz

import warnings
import argparse
from datasets import load_dataset
import torch
from rouge import Rouge

import os

# Create the folder results if it does not exist
folder_path = "results"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Load a transformer model with a specified auth token.")
parser.add_argument("--auth_token", type=str, required=True, help="Hugging Face authentication token")
args = parser.parse_args()
torch.cuda.memory_summary()

import torch 
import json
import time
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import SelfExtend 


window_size = 1024
group_size = 32
use_flash = True

# model_lists = ['google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', ]
model_lists = ['mistralai/Mistral-7B-Instruct-v0.3']
auth_token = args.auth_token

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def gen_prompt(context, input, test_name):
    prompts = {
        "qasper": f"Article: {context}\n\n Answer the question based on the above article as concisely as you can, using a single list or word if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "narrativeqa": f"Story: {context}\n\n Now, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "multifieldqa_en": f"Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "hotpotqa": f"{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": f"{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": f"The following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "lcc": f"Please complete the code given below. \n{context}Next line of code:\n",
    }
    prompt = prompts[test_name]
    prompt = f"[INST]{prompt}[/INST]"
    return prompt


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def load_model_and_tokenizer(model_name):
    print("Start loading model ",model_name)
    if 'Mistral' in model_name:
        # Disable Mistral's sliding window
        config = AutoConfig.from_pretrained(model_name)
        config.sliding_window = None
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2", device_map="auto", use_auth_token=auth_token)

    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    
    print("Tokenizer loaded")
    model.eval()
    
    #SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn")
    
    print("Finished loading")
    
    return model, tokenizer

for model_name in model_lists:
    print("=========="*2 + "**SelfExtend using flash_attn**" + "=========="*2)
    
    datasets = ["qasper"]
    results_json = []
    
    for dataset in datasets:
        torch.cuda.empty_cache()
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        
        print("---------------------------------\n")
        print(dataset)
        print("---------------------------------\n")

        total_score = 0
        expected_score = 0
        
        result = []
        
        for trial in range(0,1):
            print("Trial", trial)
            for i in range(len(data['context'])):
                model, tokenizer = load_model_and_tokenizer(model_name)
                
                expected_score += 1
                context = data['context'][i]
                input_text = data['input'][i]
                expected_answers = data["answers"][i]

                prompt = gen_prompt(context, input_text, dataset)
                
                max_length = 3500
                
                tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids[0]
                
                if len(tokenized_prompt) > max_length:
                    half = int(max_length/2)
                    prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                    
                input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids
                
                with torch.no_grad():
                    # print(input_ids.shape)
                    tokens = model.generate(input_ids, max_new_tokens=64,  num_beams=1,
                                            do_sample=False,
                                            temperature=1.0,
                                            use_cache = True)
                answer = tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
                
                score = 0
                for expected_answer in expected_answers:
                    score = max(score, dataset2metric[dataset](answer, expected_answer))
                
                print("--------------------------------------------")
                print("Expected", expected_answers)
                print("Answer:",answer)
                print("Score",score)
                total_score += score

                '''print(f"Expected answer: {expected_answers}")
                print(f"{input_text} type: {question_type}")
                print(f"Model's answer: {answer}")
                print(f"Score: {score}")'''
                
                result.append({
                    "expected_answers": expected_answers,
                    "model_answer": answer,
                    "score": score
                })
            
        results_json.append({
            "test_name": dataset,
            "score": (total_score/expected_score * 100),
            "details": result 
        })
            
        print(f"Total score: {total_score/expected_score * 100}")
    
        with open(f'results/result-{model_name.replace("/","-")}-{dataset}.json', 'w') as json_file:
            json.dump(results_json, json_file, indent=4)  
        
        results_json = []
            

        
