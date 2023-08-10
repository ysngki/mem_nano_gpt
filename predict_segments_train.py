"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py (function_name in all_configs.py)

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
import pickle
from contextlib import nullcontext
import random
import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
from peft import prepare_model_for_int8_training

from model import GPTConfig
from model import MemoryGPT as GPT
from my_configuration_roberta import MemoryRobertaConfig
from my_modeling_roberta import MemoryRobertaModel
from my_modeling_llama import LlamaForCausalLM

from my_utils import get_seq_train_batch, print_model_size, load_pretrained_model, get_lr, estimate_predict_loss

from config.training_config import train_config, import_function


def main():
    # initialize config
    config = train_config

    # load config
    if len(sys.argv) < 2:
        print("No config function specified. Using default config.")
    else:
        module_name = "config.all_configs"
        function_name = sys.argv[1]

        update_config_function = import_function(module_name, function_name)
        config = update_config_function(config)

    # set frequently used parameter
    device = config.device

    # --------------------------------------------------------------------------
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        ddp_rank = 0
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
    
    config.device = device
    # --------------------------------------------------------------------------
    # setup
    torch.manual_seed(config.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # --------------------------------------------------------------------------
    # poor man's data loader
    data_dir = os.path.join('data', config.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # get initial data pointer for each batch in this rank: this_rank_batch_train_data_pointer
    this_rank_train_data_start = ddp_rank * (len(train_data) // ddp_world_size)

    if ddp_rank == (ddp_world_size-1):
        this_rank_train_data_end = len(train_data)
    else:
        this_rank_train_data_end = (ddp_rank + 1) * (len(train_data) // ddp_world_size)

    this_rank_data_num = this_rank_train_data_end - this_rank_train_data_start
    this_rank_batch_train_data_pointer = []
    actual_batch_size = config.batch_size * config.gradient_accumulation_steps
    for bi in range(0, actual_batch_size):
        this_rank_batch_train_data_pointer.append(this_rank_train_data_start + bi * (this_rank_data_num // actual_batch_size))

    # --------------------------------------------------------------------------
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # load pretrained model
    pretrained_model, pretrained_model_config = load_pretrained_model(config)

    # backbone forzen
    for p in pretrained_model.parameters():
        p.requires_grad_(False)

    print_model_size(pretrained_model, pretrained_model_config, ddp_rank)

    # --------------------------------------------------------------------------
    # create my evolver
    input_memory_list = [None for i in range(config.gradient_accumulation_steps)]

    if config.init_from == 'scratch' or config.init_from == "load":
        evolver_config = MemoryRobertaConfig(vocab_size=pretrained_model_config.vocab_size, num_hidden_layers=config.evolver_n_layer,
                                            num_attention_heads=config.evolver_n_head, hidden_size=config.evolver_n_embd, max_position_embeddings=config.block_size, intermediate_size=config.evolver_n_intermediate,
                                            pad_token_id=config.evolver_pad_token_id, num_memory=config.evolver_n_mem,
                                            num_target_model_layer=pretrained_model_config.num_hidden_layers, no_embeddings=True, target_hidden_size=pretrained_model_config.hidden_size)
        evolver_model = MemoryRobertaModel(evolver_config)
    elif config.init_from == 'resume':
        print(f"Resuming training from {config.out_dir}")
        # resume training from a checkpoint. 
        ckpt_path = os.path.join(config.out_dir, config.load_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        evolver_config = checkpoint['evolver_config']
        # create the model
        evolver_model = MemoryRobertaModel(evolver_config)
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
    
    if config.init_from == 'load':
        # resume training from a checkpoint. 
        ckpt_path = os.path.join(config.out_dir, config.load_name)
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

    # --------------------------------------------------------------------------
    # optimizer related
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

    # optimizer
    optimizer = evolver_model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)

    if config.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        input_memory_list = checkpoint['input_memory_list']

    checkpoint = None # free up memory

    # compile the model
    if config.compile:
        print("compiling the model... (takes a ~minute)")
        pretrained_model = torch.compile(pretrained_model) # requires PyTorch 2.0
        evolver_model = torch.compile(evolver_model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
        # model = DDP(model, device_ids=[ddp_local_rank]) 
        evolver_model = DDP(evolver_model, device_ids=[ddp_local_rank], broadcast_buffers=False, find_unused_parameters=False) # https://github.com/pytorch/pytorch/issues/22095

    # --------------------------------------------------------------------------
    # logging
    if config.wandb_log and master_process:
        import wandb
        if wandb_id == "":
            wandb_id = wandb.util.generate_id()
        config['wandb_id'] = wandb_id
        wandb.init(id=wandb_id, resume='allow', project=config.wandb_project, name=config.wandb_run_name, config=config, notes=config.wandb_notes)

    # --------------------------------------------------------------------------
    # training loop
    this_rank_batch_train_data_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(train_data, this_rank_batch_train_data_pointer, config.segment_num, config.block_size, config.min_block_size, device, device_type, True) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_evolver_model = evolver_model.module if ddp else evolver_model # unwrap DDP container if needed
    running_mfu = -1.0

    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config.eval_interval == 0 and master_process:
            losses = estimate_predict_loss(evolver_model, pretrained_model, train_data, val_data, actual_batch_size, config)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if config.wandb_log:
                log_data = {
                    "iter": iter_num,
                    "val/train_loss": losses['train'],
                    "val/val_loss": losses['val'],
                    "val/train_gpt_loss": losses['train_gpt'],
                    "val/val_gpt_loss": losses['val_gpt'],
                    "val/train_context_loss": losses['train_context'],
                    "val/val_context_loss": losses['val_context'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                }
                for si in range(config.segment_num):
                    log_data[f"val/train_segment{si}_loss"] = losses['train_segment'][si]
                    log_data[f"val/val_segment{si}_loss"] = losses['val_segment'][si]

                wandb.log(log_data)
            if losses['val'] < best_val_loss or config.always_save_checkpoint:
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
                        'input_memory_list': input_memory_list,
                    }
                    print(f"saving checkpoint to {config.out_dir}")
                    torch.save(checkpoint, os.path.join(config.out_dir, config.ckpt_name))
        if iter_num == 0 and config.eval_only:
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
        for micro_step in range(config.gradient_accumulation_steps):
            # record loss in this micro step
            all_loss = torch.tensor(0.0, device=device, requires_grad=True)
            gpt_loss = torch.tensor(0.0, device=device)

            predict_loss = torch.tensor(0.0, device=device)
            revise_loss = torch.tensor(0.0, device=device)
            all_kl_loss = torch.tensor(0.0, device=device)
            context_gpt_loss = torch.tensor(0.0, device=device)
            
            # get memory of last step
            input_memory = input_memory_list[micro_step]

            this_micro_X = X[config.batch_size*micro_step : config.batch_size*(1+micro_step)]
            this_micro_Y = Y[config.batch_size*micro_step : config.batch_size*(1+micro_step)]
            this_micro_attention_mask = attention_mask[config.batch_size*micro_step : config.batch_size*(1+micro_step)]
            this_micro_seg_length_list = seg_length_list[:, config.batch_size*micro_step : config.batch_size*(1+micro_step)] # (seg num, micro batch size)

            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                evolver_model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
            with ctx:
                past_segments_x = []
                past_segments_y = []

                sampled_segments_index = set(random.sample(range(config.segment_num), max(int(config.segment_num * 0.5), 1)))
                sampled_segments_index = []
                trained_seg_num = config.segment_num + len(sampled_segments_index)

                # get data for first segment
                this_x = this_micro_X[:, 0, :]
                this_y = this_micro_Y[:, 0, :]
                this_attention_mask = this_micro_attention_mask[:, 0, :]
                this_seg_len = this_micro_seg_length_list[0]

                # generate parameters for first segment
                with torch.no_grad():
                    target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

                for si in range(config.segment_num):
                    # 保存数据用于复习
                    if si in sampled_segments_index:
                        past_segments_x.append(this_x)
                        past_segments_y.append(this_y)

                    # read this segment and update memory ------------------------------------------
                    # generate input embeddings by pretrained model
                    with torch.no_grad():
                        output_embeds = pretrained_model(input_ids=this_x, output_embeds=True)

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
                    logits_with_memory, loss = pretrained_model(input_ids=next_x, labels=next_y, input_parameter=target_model_parameter, peft=config.peft_method)
                    all_loss = loss + all_loss
                    predict_loss = loss + predict_loss

                    # reference
                    with torch.no_grad():
                        _, loss = pretrained_model(next_x, labels=next_y)
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
                        logits_with_context, loss = pretrained_model(x_container, labels=y_container, input_parameter=last_target_model_parameter, peft=config.peft_method) # shape of logits_with_context: (batch_size, two_seg_block_size, vocab_size)
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
                    _, loss = pretrained_model(this_x, labels=this_y, input_parameter=target_model_parameter, peft=config.peft_method)
                    all_loss = loss + all_loss
                    revise_loss = loss + revise_loss

                    with torch.no_grad():
                        _, loss = pretrained_model(this_x, labels=this_y)
                        gpt_loss = loss + gpt_loss

                ###
                all_loss = all_loss / (config.gradient_accumulation_steps * trained_seg_num) # scale the loss to account for gradient accumulation
                gpt_loss = gpt_loss / (config.gradient_accumulation_steps * trained_seg_num)

                predict_loss = predict_loss / (config.gradient_accumulation_steps * config.segment_num)
                revise_loss = revise_loss / (config.gradient_accumulation_steps * len(past_segments_x))
                all_kl_loss = all_kl_loss / (config.gradient_accumulation_steps * config.segment_num)
                context_gpt_loss = context_gpt_loss / (config.gradient_accumulation_steps * config.segment_num)

                lossf += all_loss.item()
                gpt_lossf += gpt_loss.item()

                predict_lossf += predict_loss.item()
                revise_lossf += revise_loss.item()
                all_kl_lossf += all_kl_loss.item()
                context_gpt_lossf += context_gpt_loss.item()

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            this_rank_batch_train_data_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(train_data, this_rank_batch_train_data_pointer, config.segment_num, config.block_size, config.min_block_size, device, device_type, True)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(all_loss).backward()

            # input_memory_list[micro_step] = input_memory.detach()
            input_memory_list[micro_step] = None

            # # 重启记忆
            # for bi in range(input_memory.shape[0]):
            #     if random.randint(1, 100) > remember_prob:
            #         input_memory_list[micro_step][bi] = raw_evolver_model.initial_memory

        # clip the gradient
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(evolver_model.parameters(), config.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_evolver_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, gpt_loss {gpt_lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            if config.wandb_log:
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
        if iter_num > config.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
