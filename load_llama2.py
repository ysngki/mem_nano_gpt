from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast
from my_modeling_llama import LlamaForCausalLM
import torch
from my_utils import get_seq_train_batch
import numpy as np
import os

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# load model and specify GPU
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map='auto')
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map={'': 'cuda:0'}, torch_dtype=torch.float16)
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to("cuda:0")
# model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map={'': 1})
# model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map={'': 0})


train_data = np.memmap("./data/llama_openwebtext/train.bin", dtype=np.uint16, mode='r')
data_pointer, x, y, attention_mask, seg_length_list = get_seq_train_batch(train_data, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 256, 128, 'cuda:0', 'cuda', True)

# output = model(x[0], attention_mask=attention_mask[0], labels=y[0])
# print(output.loss)

# output = model(x[0], labels=y[0])
# print(output.loss)



# print(type(tokenizer)) # <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>
# print(type(model)) # <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>

print(model.hf_device_map)

# inferece
tokenizer.pad_token = tokenizer.eos_token

prompt = ['The first name of the current US president is "']
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
inputs.to('cuda')

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=128)
print(generate_ids)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

for r in result:
    print(r)
    print("-"*40)