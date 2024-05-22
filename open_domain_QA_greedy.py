import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np
import pandas as pd

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
torch_seed()
gpu = 0
device = "cuda:"+str(gpu)
use_l2d = True
use_cs = True
l2d_scaling = True
analysis_only_gen_token = True
selected_k = 20

data_name = "nq_open"
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

data_path = "meta-llama/Llama-2-7b-hf"
# data_path = "google/gemma-7b"
# data_path = "meta-llama/Meta-Llama-3-8B"
# data_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# data_path = "meta-llama/Llama-2-7b-chat-hf"
# data_path = "meta-llama/Meta-Llama-Guard-2-8B"
model = AutoModelForCausalLM.from_pretrained(data_path, torch_dtype=torch.bfloat16, device_map="cuda:0", cache_dir="./cache")
original_layers = model.model.layers
print(model.device)

tokenizer = AutoTokenizer.from_pretrained(data_path)
print(model)
print(tokenizer)

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

    

test_num = 100
test_one_data = data["validation"][:test_num]
if data_dict["preprocessing"] is not None:
    test_one_data = test_one_data.map(data_dict["preprocessing"])

new_line_token = tokenizer("\n")['input_ids'][-1]

l2d_list = []
cs_list = []

result_df = pd.DataFrame(columns=["prune_idices", "EM"])

torch_seed()
print(model)
correct_counter = 0
for test_idx in range(test_num):
    input_string = input_format.format(test_one_data[data_dict["source_col"]][test_idx])
    
    # print("\n\n", test_one_data[data_dict["source_col"]][test_idx])

    input_data = tokenizer(prompt_string+" "+input_string, return_tensors="pt", add_special_tokens=False)
    # print(input_data)

    for i in input_data.keys():
        # print(i)
        input_data[i] = input_data[i].to(model.device) 

    gen_args = {'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': [tokenizer.eos_token_id, new_line_token], 'max_new_tokens': 50, 'pad_token_id': tokenizer.eos_token_id, "early_stopping":True, "temperature":0.7}
    gen_retrun_args = {"output_attentions":False, "output_hidden_states":True, "output_scores": False, "output_logits":False} #"num_return_sequences":1, 
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
        
    # break

result_df.loc[len(result_df)] = {"prune_idices": "X", "EM": correct_counter/float(test_num)}
print(result_df)
result_df.to_csv("result_df.csv")


tmp_original_layers = model.model.layers

for prune_i in range(len(tmp_original_layers)):
    
    new_layers = []
    for idx, tmp_layer in enumerate(tmp_original_layers):
        if idx != prune_i:
            new_layers.append(tmp_layer)
            
    model.model.layers = nn.ModuleList(new_layers)
    
    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.layer_idx = i
    
    torch_seed()
    print(model)
    correct_counter = 0
    for test_idx in range(test_num):
        input_string = input_format.format(test_one_data[data_dict["source_col"]][test_idx])
        
        # print("\n\n", test_one_data[data_dict["source_col"]][test_idx])

        input_data = tokenizer(prompt_string+" "+input_string, return_tensors="pt", add_special_tokens=False)
        # print(input_data)

        for i in input_data.keys():
            # print(i)
            input_data[i] = input_data[i].to(model.device) 

        gen_args = {'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': [tokenizer.eos_token_id, new_line_token], 'max_new_tokens': 50, 'pad_token_id': tokenizer.eos_token_id, "early_stopping":True, "temperature":0.7}
        gen_retrun_args = {"output_attentions":False, "output_hidden_states":True, "output_scores": False, "output_logits":False} #"num_return_sequences":1, 
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
            
        # break
    
    result_df.loc[len(result_df)] = {"prune_idices": prune_i, "EM": correct_counter/float(test_num)}
    print(result_df)
    result_df.to_csv("result_df.csv")
    
    if correct_counter/float(test_num) < 0.1:
        continue
            
    for prune_j in range(len(tmp_original_layers) - 1):
        new_layers = []
        for idx, tmp_layer in enumerate(tmp_original_layers):
            if idx != prune_i and idx != prune_j:
                new_layers.append(tmp_layer)
                
        model.model.layers = nn.ModuleList(new_layers)
        
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn.layer_idx = i
        
        torch_seed()
        print(model)
        correct_counter = 0
        for test_idx in range(test_num):
            input_string = input_format.format(test_one_data[data_dict["source_col"]][test_idx])
            
            # print("\n\n", test_one_data[data_dict["source_col"]][test_idx])

            input_data = tokenizer(prompt_string+" "+input_string, return_tensors="pt", add_special_tokens=False)
            # print(input_data)

            for i in input_data.keys():
                # print(i)
                input_data[i] = input_data[i].to(model.device) 

            gen_args = {'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': [tokenizer.eos_token_id, new_line_token], 'max_new_tokens': 50, 'pad_token_id': tokenizer.eos_token_id, "early_stopping":True, "temperature":0.7}
            gen_retrun_args = {"output_attentions":False, "output_hidden_states":True, "output_scores": False, "output_logits":False} #"num_return_sequences":1, 
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
        
        
        result_df.loc[len(result_df)] = {"prune_idices": "_".join([str(prune_i), str(prune_j)]), "EM": correct_counter/float(test_num)}
        print(result_df)
        result_df.to_csv("result_df.csv")
        # print("prune_idices = ", i, j)

        # after_prune_acc = correct_counter/float(test_num)
        # print("acc = ",after_prune_acc)
    