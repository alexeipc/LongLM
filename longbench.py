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

template = '''Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)".'''

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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import SelfExtend

print('t')

window_size = 1024
group_size = 32
use_flash = True

# model_lists = ['google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', ]
model_lists = ['mistralai/Mistral-7B-Instruct-v0.3']
auth_token = args.auth_token

def process_math(response):
    match = re.search(r'The answer is (\S+)', response)
    if not match:
        response = response.split('\n\n')[0]
        response = response.split(' ')[::-1]
        flag = False
        ret = ''
        for i in range(len(response)):
            s = response[i]
            for i in range(len(s)):
                if s[i].isdigit():
                    flag = True
                    ret = s
                    break
            if flag:
                break
    else:
        ret = match.group(1)
    ret1 = ''
    for i in range(len(ret)):
        if ret[i].isdigit():
            ret1 += ret[i]
        if ret[i] == ".":
            break
    return ret1


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

    return white_space_fix(remove_articles(remove_punc(s)))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_f1(predictions, references):
    f1 = 0
    for prediction, ground_truths in zip(predictions, references):
        #f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        f1 += f1_score(prediction, ground_truths)
    return 100.0 * f1 / len(predictions)


def extract_answer(response):
    # Use regular expression to replace anything that is not A, B, C or D with an empty string
    if len(response.strip()) == 0:
        return "None"
    if response in "ABCD":
        return response

    cleaned_str = ""
    for chr in response:
        if chr in "ABCD":
            cleaned_str += chr
            response = response[1:]
        else:
            break

    if len(cleaned_str) > 1:
        return ''.join(sorted(set(cleaned_str)))
    # retrieve multiple correct answers (for coursera)
    response = response.split("Question")[0]
    pattern = r"\s*[A-Z](?=[\s.)])"
    options = re.findall(pattern, response)
    cleaned_str += ' '.join(options).strip()
    cleaned_str = re.sub(r'[^A-D]', '', cleaned_str)
    s_set = set(cleaned_str)
    cleaned_str = "".join(sorted(s_set))
    if len(cleaned_str) < 1:  
        has_answer = False
        for chr in response:
            if chr in "ABCD":
                cleaned_str += chr
                response = response[1:]
                has_answer = True
            elif has_answer:
                break
    if len(cleaned_str) < 1:  
        cleaned_str = "A"
    return cleaned_str

def process_math(response):
    match = re.search(r'The answer is (\S+)', response)
    if not match:
        response = response.split('\n\n')[0]
        response = response.split(' ')[::-1]
        flag = False
        ret = ''
        for i in range(len(response)):
            s = response[i]
            for i in range(len(s)):
                if s[i].isdigit():
                    flag = True
                    ret = s
                    break
            if flag:
                break
    else:
        ret = match.group(1)
    ret1 = ''
    for i in range(len(ret)):
        if ret[i].isdigit():
            ret1 += ret[i]
        if ret[i] == ".":
            break
    return ret1

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

'''dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}'''

for model_name in model_lists:
    print("Start loading model ",model_name)
    if 'Mistral' in model_name:
        # Disable Mistral's sliding window
        config = AutoConfig.from_pretrained(model_name, use_auth_token=auth_token)
        config.sliding_window = None
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash, use_auth_token=auth_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2", device_map="auto", use_auth_token=auth_token)

    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    print("Tokenizer loaded")
    model.eval()
    print("Finished loading")
    file_name = "passkey_examples.jsonl"

    print("=========="*2 + "**SelfExtend using flash_attn**" + "=========="*2)
    SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn") ## flash_attention_impl="triton" or "flash_attn"
    model = model.cuda()
    '''
    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
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
    '''

    datasets = ["gsm100"]
    results_json = []

    for dataset in datasets:
        torch.cuda.empty_cache()
        data = load_dataset('L4NLP/LEval', dataset, split='test')

        print("---------------------------------\n")
        print(dataset)
        print("---------------------------------\n")
        total_score = 0
        expected_score = 0

        result = []

        print(data)

        rounds = len(data['instructions'])

        correct = 0

        total_questions = 0

        answer_total = list()

        truth_total = list()

        for q in range(rounds):
            instructions = data['instructions'][q]

            questions = len(instructions)

            total_questions += questions

            for instruction in range(questions):
                prompt = f"{data['input'][q]}\n{data['instructions'][q][instruction]}"

                input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.cuda()
                with torch.no_grad():
                    # print(input_ids.shape)
                     tokens = model.generate(input_ids, max_new_tokens=1000, num_beams=1,
                                            do_sample=False,
                                            temperature=1.0,
                                            use_cache = True)
                     #print(tokens)
                answer = tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
                

                pred = process_math(answer)

                print("-----------------------------------")
                #print(f"Prompt: {prompt}") # very long
                print(f"Question: {data['instructions'][q][instruction]}")
                print(f"Answer: {answer}")
                print(f"Pred: {pred}")
                print(f"Expected: {data['outputs'][q][instruction]}")
                print("-----------------------------------")

                correct_ans = data['outputs'][q][instruction]

                if pred == correct_ans:
                    correct += 1

                answer_total.append(pred)
                truth_total.append(correct_ans)

                

        score = compute_f1(answer_total, truth_total)
        print(f"Correct: {correct}")

        results_json.append({
            "test_name": dataset,
            "score": score,
            "details": result
        })

        print(f"Total score: {score}")

        with open(f'results/result-{model_name.replace("/","-")}-{dataset}.json', 'w') as json_file:
            json.dump(results_json, json_file, indent=4)

        results_json = []