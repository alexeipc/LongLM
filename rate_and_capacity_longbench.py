# transfromers version 4.38.2
# this example is tested with 4 RTX3090s, 24GB memory each
# Edit 1.0
import re
import string
from collections import Counter

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
parser.add_argument("--start_cap", type=int, required=True)
parser.add_argument("--end_cap", type=int, required=True)

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
capacity_range = [2**i for i in range(args.start_cap, args.end_cap + 1)]
rate_range = [x * 0.05 for x in range(1, 10)]
dataset = "qasper"


# model_lists = ['google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', ]
model_lists = ['meta-llama/Llama-2-7b-chat-hf']
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

def qa_f1_score(prediction, ground_truth, type):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

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
    }
    prompt = prompts[test_name]
    prompt = f"[INST]{prompt}[/INST]"
    return prompt

for model_name in model_lists:
    accuracy_arry = []
    
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
    print("Finished loading")
    
    
    for capacity in capacity_range:
        tmp = []
        for rate in rate_range:
            file_name = "passkey_examples.jsonl"
            print("=========="*2 + "**SelfExtend using flash_attn**" + "=========="*2)
            SelfExtend.apply(model, capacity, window_size, rate, enable_flash_attention=use_flash, flash_attention_impl="flash_attn") ## flash_attention_impl="triton" or "flash_attn"
            correct_cnt = 0
            total_cnt = 0;
            for line in open(file_name, "r"):
                total_cnt += 1
                example = json.loads(line)
                prompt_postfix = "What is the pass key? The pass key is "
                prompt = example["input"] + prompt_postfix
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                #print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
                #print( "Passkey target:", example["target"] )

                start_time = time.time()
                tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
                end_time = time.time()
                answer = tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
                answer = answer.replace("\n", "\\n")
                break
            
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            total_score = 0
            expected_score = 0    
            result = []
            
            start_time = time.time()
            
            for i in range(len(data['context'])):
                expected_score += 1
                context = data['context'][i]
                input_text = data['input'][i]
                question_type = "na"
                expected_answers = data["answers"][i]

                prompt = gen_prompt(context, input_text, dataset)
                
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
                with torch.no_grad():
                    # print(input_ids.shape)
                    tokens = model.generate(input_ids, max_new_tokens=128, use_cache = True)
                answer = tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
                
                score = 0
                for expected_answer in expected_answers:
                    score = max(score, qa_f1_score(answer, expected_answer,question_type))
                
                total_score += score
            
            end_time = time.time()
            
            tmp.append(total_score / expected_score)
            print(f"(rate = {rate}, capacity = {capacity}): {total_score / expected_score}")
            print(f"Running time: {end_time - start_time:.2f}")
        
        accuracy_arry.append(tmp)
        
print(accuracy_arry)
        