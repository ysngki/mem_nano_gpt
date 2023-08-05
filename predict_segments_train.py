"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F

from model import GPTConfig
from model import MemoryGPT as GPT
from my_configuration_roberta import MemoryRobertaConfig
from my_modeling_roberta import MemoryRobertaModel


# -----------------------------------------------------------------------------
# default config values designed to train a evolver (roberta)
evolver_n_layer = 6
evolver_n_head = 12
evolver_n_embd = 768
evolver_n_intermediate = 3072
evolver_n_mem = 50

######### no use
evolver_pad_token_id = 0
evolver_gpt2_token_id_offset = 20 # the token id produced by gpt2 tokenizer should added by this offset
#################

segment_num = 1 # if > 1, train memory
num_target_model_layer = 12

remember_prob = 95 # 大于这个的话，minibatch的记忆都会被删除
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
wandb_notes=''
seed=1337
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
pretrained_model_name = 'gpt2'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
min_block_size = 50
# model
load_name = 'place_holder'
gpt_block_size=1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    
    # 多此一举
    # # world_size number of processes will be training simultaneously, so we can scale
    # # down the desired gradient accumulation iterations per process proportionally
    # assert gradient_accumulation_steps % ddp_world_size == 0
    # gradient_accumulation_steps //= ddp_world_size
else:
    ddp_rank = 0
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# get initial data pointer for each batch in this rank: this_rank_batch_train_data_pointer
this_rank_train_data_start = ddp_rank * (len(train_data) // ddp_world_size)

if ddp_rank == (ddp_world_size-1):
    this_rank_train_data_end = len(train_data)
else:
    this_rank_train_data_end = (ddp_rank + 1) * (len(train_data) // ddp_world_size)

this_rank_data_num = this_rank_train_data_end - this_rank_train_data_start
this_rank_batch_train_data_pointer = []
actual_batch_size = batch_size * gradient_accumulation_steps
for bi in range(0, actual_batch_size):
    this_rank_batch_train_data_pointer.append(this_rank_train_data_start + bi * (this_rank_data_num // actual_batch_size))


# get data for a minibatch
def get_seq_train_batch(data, data_pointer, this_batch_seg_num, plus_one=False):
    x_list = []
    y_list = []
    seg_length_list = []

    this_batch_size = len(data_pointer)

    def get_x_y_tensor_list(batch_start_point, no_update=False):
        # random segment len for each batch item
        random_length = torch.randint(block_size - min_block_size, (this_batch_size,)) + min_block_size
        segment_ends = random_length.clone()

        for bi in range(this_batch_size):
            this_end = random_length[bi] + batch_start_point[bi] # end index
            segment_ends[bi] = this_end if this_end < len(data) else len(data) - 1
            random_length[bi] = segment_ends[bi] - batch_start_point[bi] # actual length

        # (batch size, xxx)
        x = [torch.from_numpy((data[batch_start_point[bi]:segment_ends[bi]]).astype(np.int64)) for bi in range(this_batch_size)]
        y = [torch.from_numpy((data[batch_start_point[bi] + 1:segment_ends[bi] + 1]).astype(np.int64)) for bi in range(this_batch_size)]

        # update batch_start_point
        if no_update:
            pass
        else:
            for bi in range(this_batch_size):
                batch_start_point[bi] = segment_ends[bi] if segment_ends[bi] < len(data) - min_block_size * segment_num else 0

        return x, y, batch_start_point, random_length
    

    fetch_seg_num = this_batch_seg_num + 1 if plus_one else this_batch_seg_num

    for seg_index in range(fetch_seg_num):
        # get data for this segment
        if seg_index == this_batch_seg_num: # plus one segment for prediction
            this_x, this_y, data_pointer, this_seg_length = get_x_y_tensor_list(data_pointer, True)
        else:
            this_x, this_y, data_pointer, this_seg_length = get_x_y_tensor_list(data_pointer)

        seg_length_list.append(this_seg_length)

        # padding to (batch size, block size)
        padding_x = this_x[0].new_full((this_batch_size, block_size), fill_value=0)
        padding_y = this_y[0].new_full((this_batch_size, block_size), fill_value=-1)

        for bi in range(this_batch_size):
            padding_x[bi][:len(this_x[bi])] = this_x[bi]
            padding_y[bi][:len(this_y[bi])] = this_y[bi]

        # return_offset_x.append(offset_padding_x)
        x_list.append(padding_x)
        y_list.append(padding_y)
    
    # (actual batch size, segment num, block size)
    x = torch.stack(x_list, dim=1)
    y = torch.stack(y_list, dim=1)
    attention_mask = y.ne(-1).int()
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, attention_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, attention_mask = x.to(device), y.to(device), attention_mask.to(device)
    
    # (segment num, actual batch size)
    seg_length_list = torch.stack(seg_length_list, dim=0)

    # seg_length_list: (segment num, actual batch size); x,y,attention_mask shape: (actual batch size, segment num, block size)
    return data_pointer, x, y, attention_mask, seg_length_list


# --------------------------------------------------------------------------
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
pretrained_model_config = None

# load pretrained model
if "gpt" in pretrained_model_name:
    print(f"Initializing from OpenAI GPT-2 weights: {pretrained_model_name}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    pretrained_model = GPT.from_pretrained(pretrained_model_name, override_args)
    pretrained_model.to(device)
    # backbone forzen
    for p in pretrained_model.parameters():
        p.requires_grad_(False)
    
    pretrained_model_config = pretrained_model.config
else:
    raise Exception(f"Unrecognized pretrained model {pretrained_model_name}")

pretrained_model.to(device)

# backbone forzen
for p in pretrained_model.parameters():
    p.requires_grad_(False)

# --------------------------------------------------------------------------
# create my evolver 
if init_from == 'scratch':
    evolver_config = MemoryRobertaConfig(vocab_size=pretrained_model_config.vocab_size + evolver_gpt2_token_id_offset, num_hidden_layers=evolver_n_layer,
                                        num_attention_heads=evolver_n_head, hidden_size=evolver_n_embd, max_position_embeddings=block_size, intermediate_size=evolver_n_intermediate,
                                        pad_token_id=evolver_pad_token_id, gpt2_token_id_offset=evolver_gpt2_token_id_offset, num_memory=evolver_n_mem,
                                        num_target_model_layer=num_target_model_layer, no_embeddings=True)
    evolver_model = MemoryRobertaModel(evolver_config)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint. 
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_config = checkpoint['evolver_config']
    # create the model
    evolver_model = MemoryRobertaModel(checkpoint_config)
    state_dict = checkpoint['evolver_model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    evolver_model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from == 'load':
    evolver_config = MemoryRobertaConfig(vocab_size=pretrained_model_config.vocab_size + evolver_gpt2_token_id_offset, num_hidden_layers=evolver_n_layer,
                                        num_attention_heads=evolver_n_head, hidden_size=evolver_n_embd, max_position_embeddings=block_size, intermediate_size=evolver_n_intermediate,
                                        pad_token_id=evolver_pad_token_id, gpt2_token_id_offset=evolver_gpt2_token_id_offset, num_memory=evolver_n_mem,
                                        num_target_model_layer=num_target_model_layer, no_embeddings=True)
    evolver_model = MemoryRobertaModel(evolver_config)

    # resume training from a checkpoint. 
    ckpt_path = os.path.join(out_dir, load_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['evolver_model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    evolver_model.load_state_dict(state_dict)

evolver_model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = evolver_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    pretrained_model = torch.compile(pretrained_model) # requires PyTorch 2.0
    evolver_model = torch.compile(evolver_model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
    # model = DDP(model, device_ids=[ddp_local_rank]) 
    evolver_model = DDP(evolver_model, device_ids=[ddp_local_rank], broadcast_buffers=False, find_unused_parameters=False) # https://github.com/pytorch/pytorch/issues/22095

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    pretrained_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = pretrained_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    pretrained_model.train()
    return out


@torch.no_grad()
def estimate_predict_loss():
    out = {}
    evolver_model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            data = train_data
        elif split == 'val':
            data = val_data
        else:
            raise NotImplementedError
        
        losses = torch.zeros(eval_iters)
        gpt_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            this_iter_loss = torch.tensor(0.0, device=device)
            this_iter_gpt_loss = torch.tensor(0.0, device=device)

            # random start
            random_data_start_pointer = []
            for _ in range(0, actual_batch_size):
                random_data_start_pointer.append(random.randint(0, len(data) - block_size * segment_num - 1))

            # fetch data for this batch
            random_data_start_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(data, random_data_start_pointer, segment_num, True) # fetch the very first batch

            # empty memory
            input_memory_list = [None for _ in range(gradient_accumulation_steps)]

            for micro_step in range(gradient_accumulation_steps):
                this_micro_X = X[batch_size*micro_step : batch_size*(1+micro_step)]
                this_micro_Y = Y[batch_size*micro_step : batch_size*(1+micro_step)]
                this_micro_attention_mask = attention_mask[batch_size*micro_step : batch_size*(1+micro_step)]
                this_micro_seg_length_list = seg_length_list[:, batch_size*micro_step : batch_size*(1+micro_step)] # (seg num, micro batch size)

                # get data for first segment
                this_x = this_micro_X[:, 0, :]
                this_y = this_micro_Y[:, 0, :]
                this_attention_mask = this_micro_attention_mask[:, 0, :]
                this_seg_len = this_micro_seg_length_list[0]

                # get memory of last step
                input_memory = input_memory_list[micro_step]

                target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)
                
                for si in range(segment_num):
                    output_embeds = pretrained_model(idx=this_x, input_parameter=target_model_parameter, output_embeds=True)

                    # X -> memory
                    input_memory = evolver_model(inputs_embeds=output_embeds, attention_mask=this_attention_mask, input_memory=input_memory)["memory_output"]

                    # last memory -> X
                    target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

                    # get data for next segment
                    next_x = this_micro_X[:, si + 1, :]
                    next_y = this_micro_Y[:, si + 1, :]
                    next_attention_mask = this_micro_attention_mask[:, si + 1, :]
                    next_seg_len = this_micro_seg_length_list[si+1]

                    # predict next segment with memory
                    _, loss = pretrained_model(next_x, next_y, target_model_parameter)
                    this_iter_loss = loss + this_iter_loss

                    _, gpt_loss  = pretrained_model(next_x, next_y)
                    this_iter_gpt_loss = gpt_loss + this_iter_gpt_loss

                    # assignment
                    this_x = next_x
                    this_y = next_y
                    this_attention_mask = next_attention_mask
                    this_seg_len = next_seg_len

            this_iter_loss = this_iter_loss / (gradient_accumulation_steps * segment_num)
            this_iter_gpt_loss = this_iter_gpt_loss / (gradient_accumulation_steps * segment_num)

            losses[k] = this_iter_loss.item()
            gpt_losses[k] = this_iter_gpt_loss.item()
        out[split] = losses.mean()
        out[split + "_gpt"] = gpt_losses.mean()
    evolver_model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, notes=wandb_notes)

# training loop
this_rank_batch_train_data_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(train_data, this_rank_batch_train_data_pointer, segment_num, True) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_evolver_model = evolver_model.module if ddp else evolver_model # unwrap DDP container if needed
running_mfu = -1.0

input_memory_list = [None for _ in range(gradient_accumulation_steps)]

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_predict_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "val/train_loss": losses['train'],
                "val/val_loss": losses['val'],
                "val/train_gpt_loss": losses['train_gpt'],
                "val/val_gpt_loss": losses['val_gpt'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'evolver_model': raw_evolver_model.state_dict(),
                    'evolver_config': evolver_config,
                    'optimizer': optimizer.state_dict(),
                    'pretrained_model_config': pretrained_model_config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    
    # record loss in this iteration
    evolver_model.train()
    pretrained_model.eval()

    lossf = 0.0
    gpt_lossf = 0.0

    predict_lossf = 0.0
    revise_lossf = 0.0
    all_kl_lossf = 0.0
    context_gpt_lossf = 0.0

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        # record loss in this micro step
        all_loss = torch.tensor(0.0, device=device, requires_grad=True)
        gpt_loss = torch.tensor(0.0, device=device)

        predict_loss = torch.tensor(0.0, device=device)
        revise_loss = torch.tensor(0.0, device=device)
        all_kl_loss = torch.tensor(0.0, device=device)
        context_gpt_loss = torch.tensor(0.0, device=device)
        
        # get memory of last step
        input_memory = input_memory_list[micro_step]

        this_micro_X = X[batch_size*micro_step : batch_size*(1+micro_step)]
        this_micro_Y = Y[batch_size*micro_step : batch_size*(1+micro_step)]
        this_micro_attention_mask = attention_mask[batch_size*micro_step : batch_size*(1+micro_step)]
        this_micro_seg_length_list = seg_length_list[:, batch_size*micro_step : batch_size*(1+micro_step)] # (seg num, micro batch size)

        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            evolver_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            past_segments_x = []
            past_segments_y = []

            sampled_segments_index = set(random.sample(range(segment_num), max(int(segment_num * 0.5), 1)))
            sampled_segments_index = []
            trained_seg_num = segment_num + len(sampled_segments_index)

            # get data for first segment
            this_x = this_micro_X[:, 0, :]
            this_y = this_micro_Y[:, 0, :]
            this_attention_mask = this_micro_attention_mask[:, 0, :]
            this_seg_len = this_micro_seg_length_list[0]

            # generate parameters for first segment
            with torch.no_grad():
                target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

            for si in range(segment_num):
                # 保存数据用于复习
                if si in sampled_segments_index:
                    past_segments_x.append(this_x)
                    past_segments_y.append(this_y)

                # read this segment and update memory ------------------------------------------
                # generate input embeddings by pretrained model
                with torch.no_grad():
                    output_embeds = pretrained_model(idx=this_x, input_parameter=target_model_parameter, output_embeds=True)

                # X -> memory
                input_memory = evolver_model(inputs_embeds=output_embeds, attention_mask=this_attention_mask, input_memory=input_memory)["memory_output"]
                #--------------------------------------------------------------

                last_target_model_parameter = target_model_parameter

                # last memory -> X
                target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

                # get data for next segment
                next_x = this_micro_X[:, si + 1, :]
                next_y = this_micro_Y[:, si + 1, :]
                next_attention_mask = this_micro_attention_mask[:, si + 1, :]
                next_seg_len = this_micro_seg_length_list[si+1]

                # predict next segment with memory
                logits_with_memory, loss = pretrained_model(next_x, next_y, target_model_parameter)
                all_loss = loss + all_loss
                predict_loss = loss + predict_loss

                # reference
                with torch.no_grad():
                    _, loss = pretrained_model(next_x, next_y)
                    gpt_loss = loss + gpt_loss

                #############################
                # select vaild logits (non-padding) from logits_with_memory for kl divergence
                logits_with_memory = logits_with_memory.view(-1, logits_with_memory.shape[-1]) # shape=(batch_size*seq_len, vocab_size)
                logits_with_memory = logits_with_memory[next_attention_mask.reshape(-1) == 1] # shape=(non-padding, vocab_size)

                # predict with context (teacher) and last memory
                with torch.no_grad():
                    bsz = this_x.shape[0]
                    two_seg_block_size = this_x.shape[1] + next_x.shape[1]

                    # concatenate two segments to get context
                    x_container = this_x.new_full((bsz, two_seg_block_size), fill_value=0)
                    for bi in range(bsz):
                        x_container[bi, :this_seg_len[bi]] = this_x[bi, :this_seg_len[bi]]
                        x_container[bi, this_seg_len[bi]:this_seg_len[bi] + next_seg_len[bi]] = next_x[bi, :next_seg_len[bi]]
                    
                    y_container = this_y.new_full(x_container.size(), fill_value=-1)
                    next_segment_mask = next_attention_mask.new_full(x_container.size(), fill_value=0)
                    for bi in range(bsz):
                        y_container[bi, this_seg_len[bi]:this_seg_len[bi] + next_seg_len[bi]] = next_y[bi, :next_seg_len[bi]]

                        next_segment_mask[bi, this_seg_len[bi]:this_seg_len[bi] + next_seg_len[bi]] = 1
                    
                    # predict
                    logits_with_context, loss = pretrained_model(x_container, y_container, last_target_model_parameter) # shape of logits_with_context: (batch_size, two_seg_block_size, vocab_size)
                    context_gpt_loss = loss + context_gpt_loss
                
                # select logits of second segment from logits_with_context
                logits_with_context = logits_with_context.view(-1, logits_with_context.shape[-1]) # shape=(batch_size*two_seg_block_size, vocab_size)
                logits_with_context = logits_with_context[next_segment_mask.view(-1) == 1] # shape=(non-padding, vocab_size)
                
                # calculate KL divergence between logits_with_memory and logits_with_context
                kl_loss = F.kl_div(F.log_softmax(logits_with_memory, dim=-1), F.softmax(logits_with_context, dim=-1), reduction='batchmean')
                #############################
                all_loss = kl_loss + all_loss
                all_kl_loss = kl_loss + all_kl_loss

                # assignment
                this_x = next_x
                this_y = next_y
                this_attention_mask = next_attention_mask
                this_seg_len = next_seg_len

            # 复习一下past_segments
            for (this_x, this_y) in zip(past_segments_x, past_segments_y):
                _, loss = pretrained_model(this_x, this_y, target_model_parameter)
                all_loss = loss + all_loss
                revise_loss = loss + revise_loss

                with torch.no_grad():
                    _, loss = pretrained_model(this_x, this_y)
                    gpt_loss = loss + gpt_loss

            ###
            all_loss = all_loss / (gradient_accumulation_steps * trained_seg_num) # scale the loss to account for gradient accumulation
            gpt_loss = gpt_loss / (gradient_accumulation_steps * trained_seg_num)

            predict_loss = predict_loss / (gradient_accumulation_steps * segment_num)
            revise_loss = revise_loss / (gradient_accumulation_steps * len(past_segments_x))
            all_kl_loss = all_kl_loss / (gradient_accumulation_steps * segment_num)
            context_gpt_loss = context_gpt_loss / (gradient_accumulation_steps * segment_num)

            lossf += all_loss.item()
            gpt_lossf += gpt_loss.item()

            predict_lossf += predict_loss.item()
            revise_lossf += revise_loss.item()
            all_kl_lossf += all_kl_loss.item()
            context_gpt_lossf += context_gpt_loss.item()

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        this_rank_batch_train_data_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(train_data, this_rank_batch_train_data_pointer, segment_num, True)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(all_loss).backward()

        # input_memory_list[micro_step] = input_memory.detach()
        input_memory_list[micro_step] = None

        # # 重启记忆
        # for bi in range(input_memory.shape[0]):
        #     if random.randint(1, 100) > remember_prob:
        #         input_memory_list[micro_step][bi] = raw_evolver_model.initial_memory

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(evolver_model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_evolver_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, gpt_loss {gpt_lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "train/predict_loss": predict_lossf,
                "train/revise_loss": revise_lossf,
                "train/context_gpt_loss": context_gpt_lossf,
                "train/all_kl_loss": all_kl_lossf,
                "train/gpt_loss": gpt_lossf,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
