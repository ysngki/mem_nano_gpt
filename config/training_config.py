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
    evolver_n_layer: int = 6
    evolver_n_head: int = 12
    evolver_n_embd: int = 768
    evolver_n_intermediate: int = 3072
    evolver_n_mem: int = 50

    peft_method: str = "prompt"

    # memory train parameters
    segment_num: int = 1 # if > 1, train memory
    remember_prob: int = 95 # 大于这个的话，minibatch的记忆都会被删除

    # -----------------------------------------------------------------------------
    # train
    init_from: str = 'scratch' # 'scratch' or 'resume'
    pretrained_model_name: str = 'gpt2'

    # I/O
    cache_dir: str = "/data/yuanhang/hf_cache"
    seed: int = 1337
    out_dir: str = 'out'
    ckpt_name: str = 'ckpt.pt'
    load_name: str = 'place_holder'

    # eval and save
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False # if True, script exits right after the first eval
    always_save_checkpoint: bool = True # if True, always save a checkpoint after each eval

    # data
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 5 * 8 # used to simulate larger batch sizes
    block_size: int = 1024
    min_block_size: int = 50
    batch_size: int = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size

    # wandb logging
    wandb_log: bool = False # disabled by default
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2' # 'run' + str(time.time())
    wandb_notes: str = ''
    wandb_id: str = "" # if set, will resume the run with this id

    # DDP settings
    backend: str = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device: str = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True # use PyTorch 2.0 to compile the model to be faster
    accelerate: bool = False
    
    # adamw optimizer
    learning_rate: float = 6e-4 # max learning rate
    max_iters: int = 600000 # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters: int = 2000 # how many steps to warm up for
    lr_decay_iters: int = 600000 # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # -----------------------------------------------------------------------------
    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False # do we use bias inside LayerNorm and Linear layers?
    # -----------------------------------------------------------------------------
    evolver_pad_token_id: int = 0
    evolver_gpt2_token_id_offset: int = 20 # the token id produced by gpt2 tokenizer should added by this offset
