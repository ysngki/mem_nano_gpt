# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
compile = False
wandb_log = True
wandb_project = 'predict'
wandb_run_name='1st_kl_1seg_llama'
wandb_notes='读取1个句子，每次预测下一个句子，并且有kl loss来指导预测。读取之前训练的一个句子的'

peft_method = "prompt"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
init_from = 'scratch'
pretrained_model_name = 'meta-llama/Llama-2-7b-hf'
load_name = 'predict_1st_kl_1seg.pt'
ckpt_name = 'predict_1st_kl_1seg_llama.pt'

dataset = 'llama_openwebtext'

batch_size = 3
block_size = 512
min_block_size = 256
gradient_accumulation_steps = 16
gpu_num = 1

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
warmup_iters = 2000

# eval stuff
eval_interval = 200
eval_iters = 5
log_interval = 1
always_save_checkpoint = False

# weight decay
weight_decay = 1e-1

# mem model related
evolver_n_layer = 3
evolver_n_head = 32
evolver_n_embd = 4096
evolver_n_intermediate = 8192
evolver_n_mem = 10

evolver_pad_token_id = 0
evolver_gpt2_token_id_offset = 20 # the token id produced by gpt2 tokenizer should added by this offset

# mem train related
segment_num = 1 # if > 1, train memory
remember_prob = 95
seed=12306