import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

gpu = 0
device = "cuda:"+str(gpu)

data = datasets.load_dataset("hotpot_qa","fullwiki")
print(data)

train_dataset = data['train']
validation_dataset = data['validation']
test_dataset = data['test']

# fewshot inference of boolq dataset in llama-7b
# data_path = "huggyllama/llama-7b"
data_path = "meta-llama/Meta-Llama-Guard-2-8B"
# model = AutoModelForCausalLM.from_pretrained(data_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
# model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto", torch_dtype=torch.bfloat16)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16)
print(model.device)
# model.to(device)
tokenizer = AutoTokenizer.from_pretrained(data_path)

prompt_format = "context: {} question: {} answer: {}"
input_format = "context: {} question: {} answer:"

prompt_data_num = 3
prompt_data = data["train"][:prompt_data_num]
print(prompt_data)
prompt_string = " \n ".join([prompt_format.format(prompt_data["passage"][i] ,prompt_data["question"][i], prompt_data["answer"][i]) for i in range(3)])
print(prompt_string)

test_num = len(data["validation"])
test_num = 100
test_one_data = data["validation"][:test_num]

correct_counter = 0
for test_idx in range(test_num):
    input_string = input_format.format(test_one_data["passage"][test_idx], test_one_data["question"][test_idx])

    input_data = tokenizer(prompt_string+" "+input_string, return_tensors="pt")

    for i in input_data.keys():
        # print(i)
        input_data[i] = input_data[i].to(model.device) 

    # print(input_data)
    out = model.generate(**input_data, max_new_tokens=300)
    # print(out)

    print(tokenizer.batch_decode(input_data["input_ids"]))

    pred_str = tokenizer.batch_decode(out)
    print(tokenizer.batch_decode(out))
    print(test_one_data["answer"][test_idx])

    if str(test_one_data["answer"][test_idx]).lower() in pred_str[0].split(":")[-1].lower():
        correct_counter += 1

print("exact match = ",correct_counter/float(test_num))

