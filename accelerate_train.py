
import time
from contextlib import nullcontext
import random

from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

from my_configuration_roberta import MemoryRobertaConfig
from my_modeling_roberta import MemoryRobertaModel

from my_utils import print_model_size, load_pretrained_model, get_lr, accelerate_estimate_predict_loss
from data_utils import PredictionSequentialDataset, InfiniteBatchSampler


def predict_accelerate_main(config):
    # split_batches: fetch a batch of data and split it according to the number of processes, so the batch size of loacl device is batch_size (of dataloader) / num_processes
    accelerator = Accelerator(split_batches=True, gradient_accumulation_steps=config.gradient_accumulation_steps)

    # set frequently used parameter
    device = accelerator.device
    config.device = device
    master_process = accelerator.is_main_process
    local_master_process = accelerator.is_local_main_process
    process_rank = accelerator.process_index

    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
    
    tokens_per_iter = config.gradient_accumulation_steps * accelerator.num_processes * config.batch_size * config.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    # --------------------------------------------------------------------------
    # setup
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device.type else 'cpu' # for later use in torch.autocast
    # --------------------------------------------------------------------------
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    mini_iter_num = 0 # considering accumulation
    best_val_loss = 1e9

    # load pretrained model
    pretrained_model, pretrained_model_config = load_pretrained_model(config)

    # backbone forzen
    for p in pretrained_model.parameters():
        p.requires_grad_(False)

    print_model_size(pretrained_model, pretrained_model_config, process_rank)

    # --------------------------------------------------------------------------
    # my dataloader
    data_dir = os.path.join('data', config.dataset)
    dummy_sampler = InfiniteBatchSampler(config.batch_size * accelerator.num_processes)

    numpy_train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    train_dataset = PredictionSequentialDataset(numpy_train_data, config.block_size, config.min_block_size, config.segment_num, config.batch_size * accelerator.num_processes)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size * accelerator.num_processes, sampler=dummy_sampler, pin_memory=True, shuffle=False)

    numpy_val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    val_dataset = PredictionSequentialDataset(numpy_val_data, config.block_size, config.min_block_size, config.segment_num, config.batch_size * accelerator.num_processes)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size * accelerator.num_processes, sampler=dummy_sampler, pin_memory=True, shuffle=False)

    # todo collect_fn

    # --------------------------------------------------------------------------
    # create my evolver
    input_memory = None

    if config.init_from == 'scratch' or config.init_from == "load":
        evolver_config = MemoryRobertaConfig(vocab_size=pretrained_model_config.vocab_size, num_hidden_layers=config.evolver_n_layer,
                                            num_attention_heads=config.evolver_n_head, hidden_size=config.evolver_n_embd, max_position_embeddings=config.block_size, intermediate_size=config.evolver_n_intermediate,
                                            pad_token_id=config.evolver_pad_token_id, num_memory=config.evolver_n_mem,
                                            num_target_model_layer=pretrained_model_config.num_hidden_layers, no_embeddings=True, target_hidden_size=pretrained_model_config.hidden_size)
        evolver_model = MemoryRobertaModel(evolver_config)
    elif config.init_from == 'resume':
        # although model will be loaded by accelerate later, I still load it here to get the config
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
        mini_iter_num = checkpoint['mini_iter_num']
        best_val_loss = checkpoint['best_val_loss']
        config.wandb_id = checkpoint['config'].wandb_id

        input_memory = checkpoint['input_memory']
        if checkpoint.get('batch_data_pointer', None) is not None:
            train_dataloader.dataset.batch_data_pointer = checkpoint['batch_data_pointer']
    
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

    # --------------------------------------------------------------------------
    # optimizer related
    # optimizer
    optimizer = evolver_model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)

    evolver_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        evolver_model, optimizer, train_dataloader, val_dataloader
    )

    if config.init_from == 'resume':
        accelerator.load_state(os.path.join(config.out_dir, "acc_" + config.load_name))

    checkpoint = None # free up memory

    # --------------------------------------------------------------------------
    # logging
    if config.wandb_log and master_process:
        import wandb
        if config.wandb_id == "":
            wandb_id = wandb.util.generate_id()
            config.wandb_id = wandb_id
        # wandb.init(id=wandb_id, resume='allow', project=config.wandb_project, name=config.wandb_run_name, config=config.__dict__, notes=config.wandb_notes)
        wandb.init(id=config.wandb_id, resume='allow', project=config.wandb_project, name=config.wandb_run_name, config=vars(config), notes=config.wandb_notes)

    # --------------------------------------------------------------------------
    # training loop
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    running_mfu = -1.0

    lossf = 0.0
    gpt_lossf = 0.0

    predict_lossf = 0.0
    all_kl_lossf = 0.0
    context_gpt_lossf = 0.0
    
    for batch in train_dataloader:
        # reset memory for each mini batch ?
        input_memory = None

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config.eval_interval == 0 and (mini_iter_num % config.gradient_accumulation_steps == 0):
            losses = accelerate_estimate_predict_loss(accelerator, evolver_model, pretrained_model, val_dataloader, config)
            # logging
            if master_process:
                print(f"step {iter_num}: val loss {losses['val']:.4f}")
                if config.wandb_log:
                    log_data = {
                        "iter": iter_num,
                        # "val/train_loss": losses['train'],
                        "val/val_loss": losses['val'],
                        # "val/train_gpt_loss": losses['train_gpt'],
                        "val/val_gpt_loss": losses['val_gpt'],
                        # "val/train_context_loss": losses['train_context'],
                        "val/val_context_loss": losses['val_context'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    }
                    for si in range(config.segment_num):
                        # log_data[f"val/train_segment{si}_loss"] = losses['train_segment'][si]
                        log_data[f"val/val_segment{si}_loss"] = losses['val_segment'][si]

                    wandb.log(log_data)
            
            # saving
            if losses['val'] < best_val_loss or config.always_save_checkpoint:
                # save_directory = ""
                # accelerator.save_model(evolver_model, save_directory)x
                save_model = accelerator.unwrap_model(evolver_model)
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'evolver_model': save_model.state_dict(),
                        'evolver_config': evolver_config,
                        'optimizer': optimizer.state_dict(),
                        'pretrained_model_config': pretrained_model_config,
                        'iter_num': iter_num,
                        'mini_iter_num': mini_iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                        'input_memory': input_memory,
                        'batch_data_pointer': train_dataloader.dataset.batch_data_pointer,
                    }

                    if config.always_save_checkpoint:
                        latest_ckpt_name = "latest_" + config.ckpt_name
                        if master_process:
                            print(f"saving checkpoint to {config.out_dir}")
                            torch.save(checkpoint, os.path.join(config.out_dir, latest_ckpt_name))
                        accelerator.save_state(output_dir=os.path.join(config.out_dir, "acc_" + latest_ckpt_name))

                    if losses['val'] < best_val_loss:
                        if master_process:
                            print(f"saving checkpoint to {config.out_dir}")
                            torch.save(checkpoint, os.path.join(config.out_dir, config.ckpt_name))
                        accelerator.save_state(output_dir=os.path.join(config.out_dir, "acc_" + config.ckpt_name))

        if iter_num == 0 and config.eval_only:
            break
        
        evolver_model.train()
        pretrained_model.eval()

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        with accelerator.accumulate(evolver_model):

            input_ids, labels, attention_mask, segment_lengths = batch

            # record loss in this micro step
            all_loss = torch.tensor(0.0, device=device, requires_grad=True)
            gpt_loss = torch.tensor(0.0, device=device)

            predict_loss = torch.tensor(0.0, device=device)
            all_kl_loss = torch.tensor(0.0, device=device)
            context_gpt_loss = torch.tensor(0.0, device=device)

            this_micro_X = input_ids
            this_micro_Y = labels
            this_micro_attention_mask = attention_mask
            this_micro_seg_length_list = segment_lengths

            # get data for first segment
            this_x = this_micro_X[:, 0, :]
            this_y = this_micro_Y[:, 0, :]
            this_attention_mask = this_micro_attention_mask[:, 0, :]
            this_seg_len = this_micro_seg_length_list[:, 0]

            # generate parameters for first segment
            with torch.no_grad():
                target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

            for si in range(config.segment_num):
                # read this segment and update memory ------------------------------------------
                # generate input embeddings by pretrained model
                with torch.no_grad():
                    output_embeds = pretrained_model(input_ids=this_x, output_embeds=True, return_dict=False)
                    output_embeds = output_embeds.to(evolver_model.dtype)

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
                next_seg_len = this_micro_seg_length_list[:, si+1]

                # predict next segment with memory
                logits_with_memory, loss = pretrained_model(input_ids=next_x, labels=next_y, input_parameter=target_model_parameter, peft=config.peft_method, return_dict=False)
                all_loss = loss + all_loss
                predict_loss = loss + predict_loss

                # reference
                with torch.no_grad():
                    _, loss = pretrained_model(next_x, labels=next_y, return_dict=False)
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
                    logits_with_context, loss = pretrained_model(x_container, labels=y_container, input_parameter=last_target_model_parameter, peft=config.peft_method, return_dict=False) # shape of logits_with_context: (batch_size, two_seg_block_size, vocab_size)
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

            ###
            all_loss = all_loss / config.segment_num
            gpt_loss = gpt_loss / config.segment_num

            predict_loss = predict_loss / config.segment_num
            all_kl_loss = all_kl_loss / config.segment_num
            context_gpt_loss = context_gpt_loss / config.segment_num

            lossf += all_loss.item() / config.gradient_accumulation_steps
            gpt_lossf += gpt_loss.item() / config.gradient_accumulation_steps

            predict_lossf += predict_loss.item() / config.gradient_accumulation_steps
            all_kl_lossf += all_kl_loss.item() / config.gradient_accumulation_steps
            context_gpt_lossf += context_gpt_loss.item() / config.gradient_accumulation_steps

            accelerator.backward(all_loss)

            # clip the gradient
            if config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(evolver_model.parameters(), config.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        

        mini_iter_num += 1

        # timing and logging
        if mini_iter_num % config.gradient_accumulation_steps == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % config.log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = accelerator.unwrap_model(evolver_model).estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, gpt_loss {gpt_lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
                if config.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": lossf,
                        "train/predict_loss": predict_lossf,
                        "train/context_gpt_loss": context_gpt_lossf,
                        "train/all_kl_loss": all_kl_lossf,
                        "train/gpt_loss": gpt_lossf,
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    })   
        
        if mini_iter_num % config.gradient_accumulation_steps == 0:
            iter_num += 1
            local_iter_num += 1

            lossf = 0.0
            gpt_lossf = 0.0

            predict_lossf = 0.0
            all_kl_lossf = 0.0
            context_gpt_lossf = 0.0     

        # termination conditions
        if iter_num > config.max_iters:
            break
