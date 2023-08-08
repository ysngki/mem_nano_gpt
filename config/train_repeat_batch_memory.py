# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
compile = True
wandb_log = True
wandb_project = 'repeat_mem'
wandb_run_name='batch_kl_4_seg'
wandb_notes='读取4个句子，读一个复述一个，模拟梯度传播。'
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
pretrained_model_name = 'gpt2'
init_from = 'scratch' # 'scratch' or 'resume'
ckpt_name = 'repeat_batch_kl_4_seg.pt'
# load_name = 'repeat_batch_no_revise_kl_4_seg.pt'

seed=12306
batch_size = 6
block_size = 512
min_block_size = 256
gradient_accumulation_steps = 4
gpu_num = 4

always_save_checkpoint = False

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
warmup_iters = 2000

# eval stuff
eval_interval = 100
eval_iters = 25
log_interval = 1

# weight decay
weight_decay = 1e-1

# mem related
evolver_n_layer = 6
evolver_n_head = 12
evolver_n_embd = 768
evolver_n_intermediate = 3072
evolver_n_mem = 10

evolver_pad_token_id = 0
evolver_gpt2_token_id_offset = 20 # the token id produced by gpt2 tokenizer should added by this offset

segment_num = 4 # if > 1, train memory
num_target_model_layer = 12
memory_lr = 1.0 # learning rate for the memory module

remember_prob = 95