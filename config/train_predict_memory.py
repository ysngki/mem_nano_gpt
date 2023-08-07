# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
compile = True
wandb_log = True
wandb_project = 'predict'
wandb_run_name='resume_1st_kl_load_2seg'
wandb_notes='读取4个句子，每次预测下一个句子，并且有kl loss来指导预测。读取之前训练的一个句子的'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
init_from = 'resume'
pretrained_model_name = 'gpt2'
load_name = '1st_kl_load_2seg.pt'
ckpt_name = 'resume_1st_kl_load_2seg.pt'

batch_size = 4
block_size = 512
min_block_size = 256
gradient_accumulation_steps = 8
gpu_num = 3

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
warmup_iters = 2000

# eval stuff
eval_interval = 100
eval_iters = 25
log_interval = 1
always_save_checkpoint = False

# weight decay
weight_decay = 1e-1

# mem model related
evolver_n_layer = 6
evolver_n_head = 12
evolver_n_embd = 768
evolver_n_intermediate = 3072
evolver_n_mem = 10

evolver_pad_token_id = 0
evolver_gpt2_token_id_offset = 20 # the token id produced by gpt2 tokenizer should added by this offset

num_target_model_layer = 12

# mem train related
segment_num = 2 # if > 1, train memory
remember_prob = 95
seed=12306