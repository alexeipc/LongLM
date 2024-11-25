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
capacity_range = range(8,32)
rate_range = [x * 0.05 for x in range(1, 10)]

# model_lists = ['google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', ]
model_lists = ['meta-llama/Llama-2-7b-chat-hf']
auth_token = args.auth_token

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
            # model = model.cuda()
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
                
                if (answer == example["target"]):
                    correct_cnt += 1;
                    
                #answer= f"SelfExtended-{model_name}:\n     [ {answer} ]"
                #print(answer)
                #print( answer, example["target"])
                #print( f"Runing Time: {end_time - start_time:.2f} sec" )
                #print( "-----------------------------------\n" )
            
            print(f"(rate = {rate}, capacity = {capacity}): {correct_cnt / total_cnt * 100} %")
            tmp.append(correct_cnt / total_cnt)
        
        accuracy_arry.append(tmp)
        
print(accuracy_arry)
        