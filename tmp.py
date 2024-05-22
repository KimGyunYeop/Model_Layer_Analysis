from transformers import AutoTokenizer
import transformers
import torch
import datasets

model = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="./cache")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    model_kwargs={"cache_dir":"./cache"},
)

print(pipeline.model)
print(pipeline.tokenizer)

data = datasets.load_dataset("nq_open", cache_dir="./cache")

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

prompt_format = "Q: {} \n A: {} \n"
input_format = "Q: {} \n A:"

prompt_data_num = 5
if prompt_data_num > 0:
    prompt_data = data["train"][:prompt_data_num]
    print(prompt_data)
    prompt_string = " ".join([prompt_format.format(prompt_data["question"][i] , prompt_data["answer"][i][0]) for i in range(prompt_data_num)])
    prompt_string = "Answer these questions: \n " + prompt_string
    print(prompt_string)
else:
    prompt_string = "Answer these questions: \n "

# print(prompt_string)
# quit()
# test_num = len(data["validation"])
test_num = 100
test_one_data = data["validation"][:test_num]

correct_counter = 0
for test_idx in range(test_num):
    input_string = prompt_string+" "+input_format.format(test_one_data["question"][test_idx])
    
    sequences = pipeline(
        input_string,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    
    print(input_string)
    print(test_one_data["answer"][test_idx])
    for seq in sequences:
        print(f"\nResult: {seq['generated_text']}")
    quit()