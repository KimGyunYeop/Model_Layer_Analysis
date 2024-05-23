import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse

def torch_seed(random_seed=424):

    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

def trivia_preprocessing(batch):
    batch["answer"] = batch["answer"]["aliases"] + batch["answer"]["normalized_aliases"]
    
    return batch

# model_path = "meta-llama/Llama-2-7b-hf"
# model_path = "google/gemma-7b"
# model_path = "meta-llama/Meta-Llama-3-8B"
# model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_path = "meta-llama/Llama-2-7b-chat-hf"
# model_path = "meta-llama/Meta-Llama-Guard-2-8B"
argparser = argparse.ArgumentParser()
argparser.add_argument("--data_name", type=str, default="nq_open")
argparser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
argparser.add_argument("--result_file_name", type=str, default="result_df.csv")
argparser.add_argument("--test_num", type=int, default=500)
argparser.add_argument("--max_k", type=int, default=10)
argparser.add_argument("--early_stop_percent", type=float, default=0.75)
argparser.add_argument("--gpu", type=int, default=0)
args = argparser.parse_args()

torch_seed()
device = "cuda:"+str(args.gpu)
result_file_name = "result_df.csv"
test_num = args.test_num
max_k = args.max_k
add_special_token=False
early_stop_percent = args.early_stop_percent

data_name = args.data_name

model_path = args.model_path

os.makedirs(os.path.join("results", data_name, model_path), exist_ok=True)
if os.path.isfile(os.path.join("results", data_name, model_path, args.result_file_name)):
    result_df = pd.read_csv(os.path.join("results", data_name, model_path, args.result_file_name), index_col = 0)
else:
    result_df = pd.DataFrame(columns=["prune_indices", "sorted_prune_indices", "EM", "Early_stop"])

DATAINFO = {
    "nq_open": {
        "data_path": "nq_open",
        "subset": None,
        "source_col": "question",
        "target_col": "answer",
        "preprocessing": None,
    },
    "trivia_qa": {
        "data_path": "trivia_qa",
        "subset": "rc",
        "source_col": "question",
        "target_col": "answer",
        "preprocessing": trivia_preprocessing,
    }
}
data_dict = DATAINFO[data_name]

data = datasets.load_dataset(data_dict["data_path"], data_dict["subset"], cache_dir="./cache")
print(data)

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, cache_dir="./cache")
original_layers = model.model.layers
print(model.device)

tokenizer = AutoTokenizer.from_pretrained(model_path)
print(model)
print(tokenizer)

new_line_token = tokenizer("\n")['input_ids'][-1]
stopping_ids = [tokenizer.eos_token_id, new_line_token]
if "Llama-3" in model_path:
    stopping_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
gen_args = {'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': stopping_ids, 'max_new_tokens': 50, 'pad_token_id': tokenizer.eos_token_id, "early_stopping":True, "temperature":0.7}
gen_retrun_args = {"output_attentions":False, "output_hidden_states":True, "output_scores": False, "output_logits":False} #"num_return_sequences":1, 

prompt_format = "Q: {} \n A: {} \n"
input_format = "Q: {} \n A:"

prompt_data_num = 5
if prompt_data_num > 0:
    # if data_dict["preprocessing"] is not None:
    #     prompt_data = data["train"].map(data_dict["preprocessing"])
    prompt_data = data["train"][:prompt_data_num]
    print(prompt_data)
    prompt_string = " ".join([prompt_format.format(prompt_data[data_dict["source_col"]][i] , prompt_data[data_dict["target_col"]][i][0]) for i in range(prompt_data_num)])
    print(prompt_string)
    prompt_string = "Answer these questions: \n " + prompt_string
else:
    prompt_string = "Answer these questions: \n "

    
test_one_data = data["validation"][:test_num]
if data_dict["preprocessing"] is not None:
    test_one_data = test_one_data.map(data_dict["preprocessing"])

#First test without pruning
pi = "X"
if pi not in list(result_df["prune_indices"]):
    torch_seed()
    print(model)
    print("testing original_model")
    correct_counter = 0
    for test_idx in tqdm(range(test_num)):
        input_string = input_format.format(test_one_data[data_dict["source_col"]][test_idx])

        input_data = tokenizer(prompt_string+" "+input_string, return_tensors="pt", add_special_tokens=False)

        for i in input_data.keys():
            input_data[i] = input_data[i].to(model.device) 

        out = model.generate(**input_data, **gen_args, output_hidden_states=True, return_dict_in_generate=True)
        seq_len = input_data["input_ids"].shape[1]

        pred_answer = tokenizer.batch_decode(out.sequences[:,seq_len-1:], skip_special_tokens=True)[0]
        print(tokenizer.batch_decode(out.sequences, skip_special_tokens=True))

        for i in range(len(test_one_data[data_dict["target_col"]][test_idx])):
            test_one_data[data_dict["target_col"]][test_idx][i] = test_one_data[data_dict["target_col"]][test_idx][i].strip().lower()

        # print(pred_answer, test_one_data[data_dict["target_col"]][test_idx])
        for i in test_one_data[data_dict["target_col"]][test_idx]:
            if i.lower() in pred_answer.lower():
                # print("correct")
                correct_counter += 1
                break

    result_df.loc[len(result_df)] = {"prune_indices": "X", "sorted_prune_indices":"X", "EM": correct_counter/float(test_num), "Early_stop":False}
    print(result_df)
    result_df.to_csv(os.path.join("results", data_name, model_path, args.result_file_name))


tmp_original_layers = model.model.layers

for prune_k in range(max_k):
    for prev_df_idx in range(len(result_df)):
        parents_data = result_df.loc[prev_df_idx]

        if parents_data["Early_stop"]:
            continue

        exp_list = parents_data["prune_indices"]
        # parents_score = parents_data["EM"]
        
        if exp_list == "X" :
            parents_score = parents_data["EM"]
            if prune_k != 0:
                continue

        elif len(str(exp_list).split("_")) != prune_k:
            continue

        for prune_layer_idx in range(len(tmp_original_layers)):
            
            if exp_list == "X":
                prune_list = []
            else:
                prune_list = str(exp_list).split("_")

            if str(prune_layer_idx) in prune_list:
                continue

            prune_list.append(str(prune_layer_idx))

            if "_".join(sorted(prune_list)) in list(result_df["sorted_prune_indices"]):
                continue
            
            new_layers = []
            for idx, tmp_layer in enumerate(tmp_original_layers):
                if str(idx) not in prune_list:
                    new_layers.append(tmp_layer)

            model.model.layers = nn.ModuleList(new_layers)
            
            for model_layer_idx in range(len(model.model.layers)):
                model.model.layers[model_layer_idx].self_attn.layer_idx = model_layer_idx

            torch_seed()
            print(model)
            print("cur_prune_layer = ", "_".join(prune_list))
            correct_counter = 0
            early_stop = False
            for test_idx in tqdm(range(test_num)):
                input_string = input_format.format(test_one_data[data_dict["source_col"]][test_idx])

                input_data = tokenizer(prompt_string+" "+input_string, return_tensors="pt", add_special_tokens=add_special_token)

                for i in input_data.keys():
                    input_data[i] = input_data[i].to(model.device) 

                out = model.generate(**input_data, **gen_args, output_hidden_states=True, return_dict_in_generate=True)
                seq_len = input_data["input_ids"].shape[1]

                pred_answer = tokenizer.batch_decode(out.sequences[:,seq_len-1:], skip_special_tokens=True)[0]

                for i in range(len(test_one_data[data_dict["target_col"]][test_idx])):
                    test_one_data[data_dict["target_col"]][test_idx][i] = test_one_data[data_dict["target_col"]][test_idx][i].strip().lower()

                # print(pred_answer, test_one_data[data_dict["target_col"]][test_idx])
                for i in test_one_data[data_dict["target_col"]][test_idx]:
                    if i.lower() in pred_answer.lower():
                        # print("correct")
                        correct_counter += 1
                        break
                
                if test_idx == int(test_num*0.1):
                    print("10% test done")
                    print("acc = ", correct_counter/float(test_num*0.1))
                    print("parent = ", parents_score)
                    print( early_stop_percent, "% parent = ", parents_score * early_stop_percent)
                    if correct_counter/float(int(test_num*0.1)) < parents_score * early_stop_percent:
                        print("early stop")
                        early_stop=True
                        break

            result_df.loc[len(result_df)] = {"prune_indices": "_".join(prune_list), "sorted_prune_indices":"_".join(sorted(prune_list)), "EM": correct_counter/float(test_num), "Early_stop":early_stop}
            print(result_df)
            result_df.to_csv(os.path.join("results", data_name, model_path, args.result_file_name))
