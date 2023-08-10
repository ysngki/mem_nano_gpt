from dataclasses import dataclass
from typing import ClassVar
import torch
import importlib


def import_function(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function
    except ImportError:
        print(f"Error: Module '{module_name}' not found.")
    except AttributeError:
        print(f"Error: Function '{function_name}' not found in module '{module_name}'.")


@dataclass
class train_config:
    # -----------------------------------------------------------------------------
    # default config values designed to train a evolver (roberta)
    evolver_n_layer = 6
    evolver_n_head = 12
    evolver_n_embd = 768
    evolver_n_intermediate = 3072
    evolver_n_mem = 50

    peft_method = "prompt"

    # memory train parameters
    segment_num = 1 # if > 1, train memory
    remember_prob = 95 # 大于这个的话，minibatch的记忆都会被删除

    # -----------------------------------------------------------------------------
    # train
    init_from = 'scratch' # 'scratch' or 'resume'
    pretrained_model_name = 'gpt2'

    # I/O
    cache_dir="/data/yuanhang/hf_cache"
    seed=1337
    out_dir = 'out'
    ckpt_name='ckpt.pt'
    load_name = 'place_holder'

    # eval and save
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval

    # data
    dataset = 'openwebtext'
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    block_size = 1024
    min_block_size = 50
    batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size

    # wandb logging
    wandb_log = False # disabled by default
    wandb_project = 'owt'
    wandb_run_name = 'gpt2' # 'run' + str(time.time())
    wandb_notes=''
    wandb_id = "" # if set, will resume the run with this id

    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    
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
    # -----------------------------------------------------------------------------
    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?