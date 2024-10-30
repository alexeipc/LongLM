# transfromers version 4.38.2
# this example is tested with 4 RTX3090s, 24GB memory each
# Edit 1.0
import re
import string
from collections import Counter

import warnings
import argparse
from datasets import load_dataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Load a transformer model with a specified auth token.")
parser.add_argument("--auth_token", type=str, required=True, help="Hugging Face authentication token")
args = parser.parse_args()

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

def eval(pred, answer, type):
    def yes_no_type():
            pred_lower = [token.lower() for token in pred]
            answer_lower = [token.lower() for token in answer]
            
            # Check for presence of "yes", "no", or consider it unanswerable
            if "yes" in pred_lower:
                pred_main = "yes"
            elif "no" in pred_lower:
                pred_main = "no"
            else:
                pred_main = "unanswerable"
            
            if "yes" in answer_lower:
                answer_main = "yes"
            elif "no" in answer_lower:
                answer_main = "no"
            else:
                answer_main = "unanswerable"
            
            # Convert to binary for F1 calculation
            pred_binary = 1 if pred_main == "yes" else (0 if pred_main == "no" else -1)
            answer_binary = 1 if answer_main == "yes" else (0 if answer_main == "no" else -1)
            
            return 1 if pred_binary == answer_binary else 0

    def list_type():
            # Extract unique items in both pred and answer
            common = Counter(pred) & Counter(answer)
            num_same = sum(common.values())
            #print(num_same,len(answer))
            # F1 score calculation for sets
            return num_same/len(answer)

    def other_type():
            # Assume F1 score calculation on tokenized text for other types
            return f1_score(pred, answer)

    if (type == 'yes_no'):
            return yes_no_type()
    elif (type == 'list'):
        return list_type()
    else:
        return other_type()

def classify_question_type(question):
    # Define keywords for classification, including multi-word starters
    yes_no_starters = {"is", "are", "do", "does", "was", "were", "did"}
    list_starters = {"what", "which", "on which", "how many", "in what"}
    
    # Convert the question to lowercase and strip leading/trailing whitespace
    question = question.strip().lower()
    
    # Check if the question starts with any of the multi-word or single-word starters
    for starter in list_starters:
        if question.startswith(starter):
            return "list"
    
    for starter in yes_no_starters:
        if question.startswith(starter):
            return "yes_no"
    
    # Default to "other" if no match is found
    return "other"


def qa_f1_score(prediction, ground_truth, type):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return eval(prediction_tokens, ground_truth_tokens, type)

def gen_prompt(context, input):
        prompt = f"Article: {context}\n\n Answer the question based on the above article as concisely as you can, using a single list or word if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
        prompt = f"[INST]{prompt}[/INST]"
        return prompt

for model_name in model_lists:
    print("Start loading model ",model_name)
    if 'Mistral' in model_name:
        # Disable Mistral's sliding window
        config = AutoConfig.from_pretrained(model_name)
        config.sliding_window = None
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2", use_auth_token=auth_token).cuda()

    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    print("Tokenizer loaded")
    model.eval()
    print("Finished loading")
    file_name = "passkey_examples.jsonl"
    
    print("=========="*2 + "**SelfExtend using flash_attn**" + "=========="*2)
    SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn") ## flash_attention_impl="triton" or "flash_attn"
    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
        print( "Passkey target:", example["target"] )

        start_time = time.time()
        tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
        end_time = time.time()
        answer = prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.replace("\n", "\\n")
        answer= f"SelfExtended-{model_name}:\n     [ {answer} ]"
        print( answer )
        print( f"Runing Time: {end_time - start_time:.2f} sec" )
        print( "-----------------------------------\n" )
        break;
    
    datasets = ["qasper"]
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        
        print("---------------------------------\n")
        print(dataset)
        print("---------------------------------\n")

        total_score = 0
        expected_score = 0
        
        for i in range(len(data['context'])):
        #for i in range(4,10):
            expected_score += 1
            context = data['context'][i]
            input_text = data['input'][i]
            question_type = classify_question_type(input_text)
            expected_answers = data["answers"][i]

            prompt = gen_prompt(context, input_text)
            
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            tokens = model.generate(input_ids, max_new_tokens=128)
            answer = tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
            
            score = 0
            for expected_answer in expected_answers:
                score = max(score, qa_f1_score(answer, expected_answer,question_type))
            
            total_score += score

            print(f"Expected answer: {expected_answers}")
            print(f"{input_text} type: {question_type}")
            print(f"Model's answer: {answer}")
            print(f"Score: {score}")
        
        print(total_score)
        print(f"Total score: {total_score/expected_score * 100}")