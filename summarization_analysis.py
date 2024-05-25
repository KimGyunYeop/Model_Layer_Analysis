import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np

from matplotlib import pyplot as plt
import os

import evaluate
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

# data_path = "meta-llama/Llama-2-7b-hf"
# data_path = "meta-llama/Llama-2-13b-hf"
# data_path = "meta-llama/Llama-2-70b-hf"
# data_path = "google/gemma-7b"
# data_path = "meta-llama/Meta-Llama-3-8B"
# data_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# data_path = "meta-llama/Llama-2-7b-chat-hf"
# data_path = "meta-llama/Meta-Llama-Guard-2-8B"
argparser = argparse.ArgumentParser()
argparser.add_argument("--data_name", type=str, default="cnndm")
argparser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
argparser.add_argument("--result_file_name", type=str, default="result_df.csv")
argparser.add_argument("--test_num", type=int, default=1000)
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument("--auto", default=False, action="store_true")
args = argparser.parse_args()

torch_seed()
gpu = args.gpu
device = "cuda:"+str(gpu)
if args.auto:
    device = "auto"
use_l2d = True
use_cs = True
l2d_scaling = True
analysis_only_gen_token = True
test_num = args.test_num
add_special_tokens = False
data_name = "cnndm"

result_txt = ""
if not analysis_only_gen_token:
    result_txt += "allTokens"
if add_special_tokens:
    result_txt += "addSpecialTokens"

result_txt += str(test_num)

DATAINFO = {
    "cnndm": {
        "data_path": "abisee/cnn_dailymail",
        "subset": "3.0.0",
        "source_col": "article",
        "target_col": "highlights",
        "preprocessing": None,
    }
}
data_dict = DATAINFO[data_name]

data = datasets.load_dataset(data_dict["data_path"], data_dict["subset"], cache_dir="./cache")
print(data)

# data_path = "meta-llama/Llama-2-7b-hf"
# data_path = "meta-llama/Llama-2-13b-hf"
# data_path = "meta-llama/Llama-2-70b-hf"
# data_path = "google/gemma-7b"
# data_path = "meta-llama/Meta-Llama-3-8B"
data_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# data_path = "meta-llama/Llama-2-7b-chat-hf"
# data_path = "meta-llama/Meta-Llama-Guard-2-8B"
data_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(data_path, torch_dtype=torch.bfloat16, device_map=device, cache_dir="./cache")
print(model.device)

tokenizer = AutoTokenizer.from_pretrained(data_path)
print(model)
print(tokenizer)

stopping_ids = [tokenizer.eos_token_id]
# new_line_token = tokenizer("\n")['input_ids'][-1]
# stopping_ids.append(new_line_token)
# new_line_token2 = tokenizer(" \n")['input_ids'][-1]
# stopping_ids.append(new_line_token2)
if "Llama-3" in data_path:
    stopping_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
gen_args = {'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': stopping_ids, 'max_new_tokens': 300, 'pad_token_id': tokenizer.eos_token_id, "early_stopping":True, "temperature":0.7}

prompt_format = """
### Input:
{}

### Summary:
{}


"""
input_format = """
### Input:
{}

### Summary:

"""

prompt_data_num = 0
if prompt_data_num > 0:
    # if data_dict["preprocessing"] is not None:
    #     prompt_data = data["train"].map(data_dict["preprocessing"])
    prompt_data = data["train"][:prompt_data_num]
    print(prompt_data)
    prompt_string = " ".join([prompt_format.format(prompt_data[data_dict["source_col"]][i].strip() , prompt_data[data_dict["target_col"]][i][0]) for i in range(prompt_data_num)])
    print(prompt_string)
    prompt_string = "Instruction: \nSummarize the following conversation." + prompt_string
else:
    prompt_string = "Instruction: \nSummarize the following conversation.\n"


# Model Anlysis
q_proj = []
k_proj = []
v_proj = []
o_proj = []
mlp_gate_proj = []
mlp_up_proj = []
mlp_down_proj = []
for i in model.model.layers:
    # print(i.self_attn.q_proj.weight.data)
    # print(i.self_attn.k_proj.weight.data)
    # print(i.self_attn.v_proj.weight.data)
    # print(i.self_attn.o_proj.weight.data)
    
    # print(i.mlp.gate_proj.weight.data)
    # print(i.mlp.up_proj.weight.data)
    # print(i.mlp.down_proj.weight.data)

    q_proj.append(torch.mean(torch.abs(i.self_attn.q_proj.weight.data)).item())
    k_proj.append(torch.mean(torch.abs(i.self_attn.k_proj.weight.data)).item())
    v_proj.append(torch.mean(torch.abs(i.self_attn.v_proj.weight.data)).item())
    o_proj.append(torch.mean(torch.abs(i.self_attn.o_proj.weight.data)).item())
    mlp_gate_proj.append(torch.mean(torch.abs(i.mlp.gate_proj.weight.data)).item())
    mlp_up_proj.append(torch.mean(torch.abs(i.mlp.up_proj.weight.data)).item())
    mlp_down_proj.append(torch.mean(torch.abs(i.mlp.down_proj.weight.data)).item())
    
print("q_proj_abs", q_proj)
print("k_proj_abs", k_proj)
print("v_proj_abs", v_proj)
print("o_proj_abs", o_proj)
print("mlp_gate_proj_abs", mlp_gate_proj)
print("mlp_up_proj_abs", mlp_up_proj)
print("mlp_down_proj_abs", mlp_down_proj)

os.makedirs("figs", exist_ok=True)
os.makedirs( os.path.join("figs", data_path, result_txt), exist_ok=True)
plt.plot(q_proj, label="q_proj")
plt.savefig(os.path.join("figs", data_path, result_txt, "q_proj.png"))
plt.clf()
plt.plot(k_proj, label="k_proj")
plt.savefig(os.path.join("figs", data_path, result_txt, "k_proj.png"))
plt.clf()
plt.plot(v_proj, label="v_proj")
plt.savefig(os.path.join("figs", data_path, result_txt, "v_proj.png"))
plt.clf()
plt.plot(o_proj, label="o_proj")
plt.savefig(os.path.join("figs", data_path, result_txt, "o_proj.png"))
plt.clf()
plt.plot(mlp_gate_proj, label="mlp_gate_proj")
plt.savefig(os.path.join("figs", data_path, result_txt, "mlp_gate_proj.png"))
plt.clf()
plt.plot(mlp_up_proj, label="mlp_up_proj")
plt.savefig(os.path.join("figs", data_path, result_txt, "mlp_up_proj.png"))
plt.clf()
plt.plot(mlp_down_proj, label="mlp_down_proj")
plt.savefig(os.path.join("figs", data_path, result_txt, "mlp_down_proj.png"))
plt.clf()


# Experiment Analysis
test_one_data = data["validation"][:test_num]
if data_dict["preprocessing"] is not None:
    test_one_data = test_one_data.map(data_dict["preprocessing"])

new_line_token = tokenizer("\n")['input_ids'][-1]

l2d_list = []
scaled_l2d_list = []
cs_list = []
self_token_attention_scores_list = []
correct_counter = 0
counting_gen = 0
targets = []
preds = []
for test_idx in range(test_num):
    input_string = input_format.format(test_one_data[data_dict["source_col"]][test_idx]).strip()
    targets.append(test_one_data[data_dict["target_col"]][test_idx].strip())
    
    # print("\n\n", test_one_data[data_dict["source_col"]][test_idx])

    input_data = tokenizer(prompt_string+" "+input_string, return_tensors="pt", add_special_tokens=add_special_tokens)
    # print(input_data)

    for i in input_data.keys():
        # print(i)
        input_data[i] = input_data[i].to(model.device) 
        
    out = model.generate(**input_data, **gen_args, output_hidden_states=True, return_dict_in_generate=True, output_attentions=True)
    seq_len = input_data["input_ids"].shape[1]

    pred_answer = tokenizer.batch_decode(out.sequences[:,seq_len:], skip_special_tokens=True)[0].strip()
    preds.append(pred_answer.strip())
    # print(out.keys())
    # print(len(out["attentions"]))
    # print(len(out["attentions"][0]))
    # print(out["attentions"][0][0].shape)
    cur_self_token_attention_scores = []
    counting_gen += len(out["attentions"])
    for i in range(len(out["attentions"])):
        cur_layer_self_token_attention_scores = []
        for j in range(len(out["attentions"][i])):
            cur_layer_self_token_attention_scores.append(torch.mean(out["attentions"][i][j][:,:,-1,-1]).item())
        
        cur_self_token_attention_scores.append(cur_layer_self_token_attention_scores)

    if not analysis_only_gen_token:
        prev_self_token_attention_scores = []
        for k in range(out["attentions"][0][j].size(-1) - 1):
            cur_layer_self_token_attention_scores = []
            for j in range(len(out["attentions"][0])):
                cur_layer_self_token_attention_scores.append(torch.mean(out["attentions"][0][j][:,:,k,k]).item())

            prev_self_token_attention_scores.append(cur_layer_self_token_attention_scores)
        cur_self_token_attention_scores = torch.cat([torch.tensor(prev_self_token_attention_scores), torch.tensor(cur_self_token_attention_scores)], dim=0).tolist()
    self_token_attention_scores_list.extend(cur_self_token_attention_scores)
    
    
    out["hidden_states"] = list(out["hidden_states"])
    for i in range(len(out["hidden_states"])):
        out["hidden_states"][i] = torch.cat(out["hidden_states"][i], dim=0)
        
    
    if analysis_only_gen_token:
        out["hidden_states"][0] = out["hidden_states"][0][:,-1:,:]
        
    out["hidden_states"] = torch.cat(out["hidden_states"], dim=1)
    
    pred_with_hidden = model.lm_head(out["hidden_states"][-1,:,:])

    pred_token = torch.argmax(pred_with_hidden, dim=-1)

    # pred_string1 = tokenizer.batch_decode(pred_token.unsqueeze(0)[:,:], skip_special_tokens=True)
    # print("pred_string:", pred_string1)

    # for i in range(out["hidden_states"].size(0)):
    #     pred_with_hidden = model.lm_head(out["hidden_states"][i,:,:])

    #     pred_token = torch.argmax(pred_with_hidden, dim=-1)

    #     pred_string1 = tokenizer.batch_decode(pred_token.unsqueeze(0)[:,:], skip_special_tokens=True)
    #     print(i, pred_string1)
    

    print("L2 disttance of n to n+1 hidden states")
    # l2d = torch.mean((out["hidden_states"][:-1,:,:] - out["hidden_states"][1:,:,:]).pow(2).sum(2).sqrt() ,dim=-1)
    
    # vector_distance = torch.mean((out["hidden_states"][:-1,:,:]).pow(2).sum(2).sqrt(), dim=-1)
    l2d = (out["hidden_states"][:-1,:,:] - out["hidden_states"][1:,:,:]).pow(2).sum(2).sqrt().T
    
    vector_distance = (out["hidden_states"][:-1,:,:]).pow(2).sum(2).sqrt().T
    scaled_l2d = l2d/vector_distance
        
    print(torch.mean(l2d, dim=0))
    l2d_list.append(l2d)
    scaled_l2d_list.append(scaled_l2d)
    

    print("cosine similarity betwwen n and n+1 hidden states")
    cs = F.cosine_similarity(out["hidden_states"][:-1,:,:], out["hidden_states"][1:,:,:], dim=-1).T
    # cs = cs.mean(dim=-1)
    print(cs.mean(dim=0))
    cs_list.append(cs)

    print("reference:\n", targets[-1])
    print("prediction:\n", pred_answer)

rouge = evaluate.load('rouge')
results = rouge.compute(predictions=preds, references=targets)
print(results)

print("counting_gen:",counting_gen)
num_layer = torch.cat(l2d_list, dim=0).shape[1]

print("final_l2d:")
# final_l2d = torch.mean(torch.stack(l2d_list), dim=0)  
print(torch.cat(l2d_list, dim=0).shape)
final_l2d = torch.mean(torch.cat(l2d_list, dim=0), dim=0)   
print(final_l2d)

plt.plot(final_l2d.tolist())
plt.title("L2 distance")
plt.savefig(os.path.join("figs", data_path, result_txt, "l2d.png"))
plt.xticks(range(1, num_layer+1))
plt.clf()

plt.figure(figsize=(16,8))
plt.boxplot(torch.cat(l2d_list, dim=0).T.tolist())
plt.title("l2d_boxplot")
plt.xticks(range(1, num_layer+1))
plt.savefig(os.path.join("figs", data_path, result_txt, "l2d_boxplot.png"))
plt.clf()


print("final_scaled_l2d:")
print(torch.cat(scaled_l2d_list, dim=0).shape)
final_scaled_l2d = torch.mean(torch.cat(scaled_l2d_list, dim=0), dim=0)    
print(final_scaled_l2d)

plt.plot(final_scaled_l2d.tolist())
plt.title("scaled L2 distance")
plt.savefig(os.path.join("figs", data_path, result_txt, "scaled_l2d.png"))
plt.xticks(range(1, num_layer+1))
plt.clf()

plt.figure(figsize=(16,8))
plt.boxplot(torch.cat(scaled_l2d_list, dim=0).T.tolist())
plt.title("scaled_l2d_boxplot")
plt.xticks(range(1, num_layer+1))
plt.savefig(os.path.join("figs", data_path, result_txt, "scaled_l2d_boxplot.png"))
plt.clf()

print("final_cs:")
print(torch.cat(cs_list, dim=0).shape)
final_cs = torch.mean(torch.cat(cs_list, dim=0), dim=0)  
print(final_cs)

plt.plot(final_cs.tolist(), label="cs")
plt.title("cosine similarity")
plt.savefig(os.path.join("figs", data_path, result_txt, "cs.png"))
plt.xticks(range(1, num_layer+1))
plt.clf()

plt.figure(figsize=(16,8))
plt.boxplot(torch.cat(cs_list, dim=0).T.tolist())
plt.title("cs_boxplot")
plt.xticks(range(1, num_layer+1))
plt.savefig(os.path.join("figs", data_path, result_txt, "cs_boxplot.png"))
plt.clf()

print("final_rev_cs:")
print((1 - torch.cat(cs_list, dim=0)).shape)
final_rev_cs = torch.mean(1 - torch.cat(cs_list, dim=0), dim=0)  
print(final_rev_cs)

plt.plot(final_rev_cs.tolist(), label="reverse_cs")
plt.title("reverse_cosine similarity")
plt.savefig(os.path.join("figs", data_path, result_txt, "reverse_cs.png"))
plt.xticks(range(1, num_layer+1))
plt.clf()

plt.figure(figsize=(16,8))
plt.boxplot((1 - torch.cat(cs_list, dim=0)).T.tolist())
plt.title("reverse_cs_boxplot")
plt.xticks(range(1, num_layer+1))
plt.savefig(os.path.join("figs", data_path, result_txt, "reverse_cs_boxplot.png"))
plt.clf()

print("final_self_token_attnetion:")
print(torch.tensor(self_token_attention_scores_list).shape)
final_self_token_attnetion_scores = torch.mean(torch.tensor(self_token_attention_scores_list), dim=0)  
print(final_self_token_attnetion_scores)

plt.plot(final_self_token_attnetion_scores.tolist())
plt.title("self_token_attnetion_scores")
plt.savefig(os.path.join("figs", data_path, result_txt, "self_token_attnetion_scores.png"))
plt.xticks(range(1, num_layer+1))
plt.clf()

plt.figure(figsize=(16,8))
plt.boxplot(torch.tensor(self_token_attention_scores_list).T.tolist())
plt.title("self_token_attnetion_scores_boxplot")
plt.xticks(range(1, num_layer+1))
plt.savefig(os.path.join("figs", data_path, result_txt, "self_token_attnetion_scores_boxplot.png"))
plt.clf()

before_prune_acc = correct_counter/float(test_num)
print("acc = ",before_prune_acc)
