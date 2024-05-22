import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np

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
gpu = 1
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

# print(prompt_string)
# quit()
# test_num = len(data["validation"])
test_num = 100
test_one_data = data["validation"][:test_num]
if data_dict["preprocessing"] is not None:
    test_one_data = test_one_data.map(data_dict["preprocessing"])

new_line_token = tokenizer("\n")['input_ids'][-1]

l2d_list = []
cs_list = []
correct_counter = 0
for test_idx in range(test_num):
    input_string = input_format.format(test_one_data[data_dict["source_col"]][test_idx])
    
    print("\n\n", test_one_data[data_dict["source_col"]][test_idx])

    input_data = tokenizer(prompt_string+" "+input_string, return_tensors="pt", add_special_tokens=False)
    # print(input_data)

    for i in input_data.keys():
        # print(i)
        input_data[i] = input_data[i].to(model.device) 

    # print(input_data)
    # print(input_data["input_ids"].shape)
    gen_args = {'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': [tokenizer.eos_token_id, new_line_token], 'max_new_tokens': 50, 'pad_token_id': tokenizer.eos_token_id, "early_stopping":True, "temperature":0.7}
    gen_retrun_args = {"output_attentions":False, "output_hidden_states":True, "output_scores": False, "output_logits":False} #"num_return_sequences":1, 
    # gen_args = {'max_length': 4068}
    # print(model.generate(**input_data, **gen_args, output_hidden_states=True, return_dict_in_generate=True).keys())
    # quit()
    out = model.generate(**input_data, **gen_args, output_hidden_states=True, return_dict_in_generate=True)
    # print(out.sequences.shape)
    # print(input_data["input_ids"].shape)
    seq_len = input_data["input_ids"].shape[1]

    pred_answer = tokenizer.batch_decode(out.sequences[:,seq_len-1:], skip_special_tokens=True)[0]
    # print(pred_string1)
    # pred_answer = pred_string1[0].split("Q:")[0].strip()
    # for i in range(len(out["hidden_states"])):
    #     print(out["hidden_states"][i][0].shape)
    #     print(out["hidden_states"][i][-1].shape)
    #     # for j in range(len(out["hidden_states"][i])):

    #analysis
    
    out["hidden_states"] = list(out["hidden_states"])
    print(len(out["hidden_states"]))
    for i in range(len(out["hidden_states"])):
        out["hidden_states"][i] = torch.cat(out["hidden_states"][i], dim=0)
        # print(out["hidden_states"][i].shape)
        
    if analysis_only_gen_token:
        out["hidden_states"][0] = out["hidden_states"][0][:,-2:,:]
        
    out["hidden_states"] = torch.cat(out["hidden_states"], dim=1)
    
    print(out["hidden_states"].shape)
    pred_with_hidden = model.lm_head(out["hidden_states"][-1,:,:])
    # print(pred_with_hidden.shape)

    pred_token = torch.argmax(pred_with_hidden, dim=-1)
    print(pred_token.shape)

    pred_string1 = tokenizer.batch_decode(pred_token.unsqueeze(0)[:,:], skip_special_tokens=True)
    print("pred_string:", pred_string1)

    for i in range(out["hidden_states"].size(0)):
        pred_with_hidden = model.lm_head(out["hidden_states"][i,:,:])

        pred_token = torch.argmax(pred_with_hidden, dim=-1)

        pred_string1 = tokenizer.batch_decode(pred_token.unsqueeze(0)[:,:], skip_special_tokens=True)
        print(i, pred_string1)
    
    # a = out["hidden_states"][:-1,:,:] - out["hidden_states"][1:,:,:]
    # print(a.shape)
    # a = torch.sum(a * a, dim=-1)
    # print(a.shape)
    # a = torch.sqrt(a)
    # print(torch.mean(a, dim=-1))

    print("L2 disttance of n to n+1 hidden states")
    l2d = torch.mean((out["hidden_states"][:-1,:,:] - out["hidden_states"][1:,:,:]).pow(2).sum(2).sqrt() ,dim=-1)
    
    if l2d_scaling:
        vector_distance = torch.mean((out["hidden_states"][:-1,:,:]).pow(2).sum(2).sqrt(), dim=-1)
        l2d = l2d/vector_distance
    # print((out["hidden_states"][:-1,:,:] - out["hidden_states"][1:,:,:]).pow(2).sum(2).sqrt().shape)
    print(l2d)
    l2d_list.append(l2d)
    

    print("cosine similarity betwwen n and n+1 hidden states")
    cs = F.cosine_similarity(out["hidden_states"][:-1,:,:], out["hidden_states"][1:,:,:], dim=-1)
    cs = cs.mean(dim=-1)
    print(cs)
    cs_list.append(cs)


    for i in range(len(test_one_data[data_dict["target_col"]][test_idx])):
        test_one_data[data_dict["target_col"]][test_idx][i] = test_one_data[data_dict["target_col"]][test_idx][i].strip().lower()

    print(pred_answer, test_one_data[data_dict["target_col"]][test_idx])
    for i in test_one_data[data_dict["target_col"]][test_idx]:
        if i.lower() in pred_answer.lower():
            print("correct")
            correct_counter += 1
            break
        
    # input()
        
    # if str(test_one_data["answer"][test_idx]) in pred_string[0].split(":")[-1].lower():
    #     correct_counter += 1
    # quit()

print("final_l2d:")
final_l2d = torch.mean(torch.stack(l2d_list), dim=0)    
print(final_l2d)

print("final_cs:")
final_cs = torch.mean(torch.stack(cs_list), dim=0)  
print(final_cs)

before_prune_acc = correct_counter/float(test_num)
print("acc = ",before_prune_acc)

top_k_l2d_idx = torch.topk(final_l2d, k=selected_k).indices
print(top_k_l2d_idx)
top_k_cs_idx = torch.topk(final_cs, k=selected_k, largest=False).indices
print(top_k_cs_idx)

selcted_layer_idx = []
for i in range(selected_k):
    if use_l2d:
        if top_k_l2d_idx[i].item() not in selcted_layer_idx:
            selcted_layer_idx.append(top_k_l2d_idx[i].item())
            
        if len(selcted_layer_idx) == selected_k:
            break
    
    if use_cs:
        if top_k_cs_idx[i].item() not in selcted_layer_idx:
            selcted_layer_idx.append(top_k_cs_idx[i].item())
        
        if len(selcted_layer_idx) == selected_k:
            break

print(selcted_layer_idx)

new_layers = []
for i in range(len(model.model.layers)):
    if i in selcted_layer_idx:
        new_layers.append(model.model.layers[i])    
    
model.model.layers = nn.ModuleList(new_layers)

for i in range(len(model.model.layers)):
    model.model.layers[i].self_attn.layer_idx = i

torch_seed()
print(model)
correct_counter = 0
for test_idx in range(test_num):
    input_string = input_format.format(test_one_data[data_dict["source_col"]][test_idx])
    
    print("\n\n", test_one_data[data_dict["source_col"]][test_idx])

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

    print(pred_answer, test_one_data[data_dict["target_col"]][test_idx])
    for i in test_one_data[data_dict["target_col"]][test_idx]:
        if i.lower() in pred_answer.lower():
            print("correct")
            correct_counter += 1
            break
        


print("\nfinal_l2d:")  
print(final_l2d)
print(top_k_l2d_idx)

print("\nfinal_cs:")
print(final_cs)
print(top_k_cs_idx)

print(f"\nselected_top{len(selcted_layer_idx)}_layer:")
print(selcted_layer_idx)
print("not selected layer:")
for i in range(model.config.num_hidden_layers):
    if i not in selcted_layer_idx:
        print(i, end=" ")
        
print("\nbefore acc = ",before_prune_acc)
after_prune_acc = correct_counter/float(test_num)
print("acc = ",after_prune_acc)