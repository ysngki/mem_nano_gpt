# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
compile = False
wandb_log = False
wandb_project = 'mem'
wandb_run_name='first_train_mem'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
init_from = 'gpt2'

batch_size = 4
block_size = 512
min_block_size = 256
gradient_accumulation_steps = 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# mem related
evolver_n_layer = 6
evolver_n_head = 12
evolver_n_embd = 768
evolver_n_intermediate = 3072
evolver_n_mem = 100

evolver_pad_token_id = 0
evolver_gpt2_token_id_offset = 20 # the token id produced by gpt2 tokenizer should added by this offset

segment_num = 12 # if > 1, train memory

num_target_model_layer = 12