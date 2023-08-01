from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map='auto')

# load model and specify GPU
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map={'': 'cuda:1'}, torch_dtype=torch.float16)
# model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map={'': 1})
# model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map={'': 0})

# print(type(tokenizer)) # <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>
# print(type(model)) # <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>

print(model.hf_device_map)

# inferece
tokenizer.pad_token = tokenizer.eos_token

prompt = ["Hey, are you consciours? Can you talk to me?", 'i am a superstar']
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
inputs.to('cuda')

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

for r in result:
    print(r)
    print("-"*40)