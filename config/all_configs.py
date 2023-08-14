def predict_lora_llama_config(this_config):
    this_config.out_dir = 'llama_out'
    this_config.compile = False
    this_config.wandb_log = True
    this_config.wandb_project = 'predict'
    this_config.wandb_run_name='1st_1seg_r20_llama'
    this_config.wandb_notes='读取1个句子，每次预测下一个句子，并且有kl loss来指导预测。读取之前训练的一个句子的'

    this_config.peft_method = "lora"

    this_config.init_from = 'scratch'
    this_config.pretrained_model_name = 'meta-llama/Llama-2-7b-hf'
    this_config.accelerate = True
    # this_config.load_name = 'predict_1st_kl_1seg_m20_llama.pt'
    this_config.ckpt_name = 'predict_1st_1seg_r20_llama.pt'

    this_config.dataset = 'llama_openwebtext'

    this_config.batch_size = 1
    this_config.block_size = 512
    this_config.min_block_size = 256
    this_config.gradient_accumulation_steps = 24
    this_config.gpu_num = 2

    # this makes total number of tokens be 300B
    this_config.max_iters = 600000
    this_config.lr_decay_iters = 600000
    this_config.warmup_iters = 2000

    # eval stuff
    this_config.eval_interval = 100
    this_config.eval_iters = 10
    this_config.log_interval = 1
    this_config.always_save_checkpoint = True

    # weight decay
    this_config.weight_decay = 1e-1

    # mem model related
    this_config.evolver_n_layer = 6
    this_config.evolver_n_head = 32
    this_config.evolver_n_embd = 4096
    this_config.evolver_n_intermediate = 8192
    this_config.evolver_n_mem = 20

    # mem train related
    this_config.segment_num = 1 # if > 1, train memory
    this_config.remember_prob = 95
    this_config.seed=32306

    return this_config


def predict_llama_config(this_config):
    this_config.compile = False
    this_config.wandb_log = True
    this_config.wandb_project = 'predict'
    this_config.wandb_run_name='1st_kl_1seg_m20_llama'
    this_config.wandb_notes='读取1个句子，每次预测下一个句子，并且有kl loss来指导预测。读取之前训练的一个句子的'

    this_config.peft_method = "prompt"

    # these make the total batch size be ~0.5M
    # 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
    this_config.init_from = 'resume'
    this_config.pretrained_model_name = 'meta-llama/Llama-2-7b-hf'
    this_config.accelerate = True
    this_config.load_name = 'predict_1st_kl_1seg_m20_llama.pt'
    this_config.ckpt_name = 'predict_1st_kl_1seg_m20_llama.pt'

    this_config.dataset = 'llama_openwebtext'

    this_config.batch_size = 1
    this_config.block_size = 512
    this_config.min_block_size = 256
    this_config.gradient_accumulation_steps = 16
    this_config.gpu_num = 3

    # this makes total number of tokens be 300B
    this_config.max_iters = 600000
    this_config.lr_decay_iters = 600000
    this_config.warmup_iters = 2000

    # eval stuff
    this_config.eval_interval = 100
    this_config.eval_iters = 10
    this_config.log_interval = 1
    this_config.always_save_checkpoint = True

    # weight decay
    this_config.weight_decay = 1e-1

    # mem model related
    this_config.evolver_n_layer = 6
    this_config.evolver_n_head = 32
    this_config.evolver_n_embd = 4096
    this_config.evolver_n_intermediate = 8192
    this_config.evolver_n_mem = 20

    # mem train related
    this_config.segment_num = 1 # if > 1, train memory
    this_config.remember_prob = 95
    this_config.seed=32306

    return this_config


def predict_config(this_config):
    this_config.compile = False
    this_config.wandb_log = True
    this_config.wandb_project = 'predict'
    this_config.wandb_run_name='1st_kl_1seg_acc'
    this_config.wandb_notes='读取1个句子，每次预测下一个句子，并且有kl loss来指导预测。读取之前训练的一个句子的'

    this_config.peft_method = "prompt"

    # these make the total batch size be ~0.5M
    # 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
    this_config.init_from = 'scratch'
    this_config.pretrained_model_name = 'gpt2'
    this_config.accelerate = True
    # load_name = 'predict_1st_kl_1seg.pt'
    this_config.ckpt_name = 'predict_1st_kl_1seg_acc.pt'

    this_config.dataset = 'openwebtext'

    this_config.batch_size = 16
    this_config.block_size = 512
    this_config.min_block_size = 256
    this_config.gradient_accumulation_steps = 4
    this_config.gpu_num = 3

    # this makes total number of tokens be 300B
    this_config.max_iters = 600000
    this_config.lr_decay_iters = 600000
    this_config.warmup_iters = 2000

    # eval stuff
    this_config.eval_interval = 100
    this_config.eval_iters = 25
    this_config.log_interval = 1
    this_config.always_save_checkpoint = False

    # weight decay
    this_config.weight_decay = 1e-1

    # mem model related
    this_config.evolver_n_layer = 6
    this_config.evolver_n_head = 12
    this_config.evolver_n_embd = 768
    this_config.evolver_n_intermediate = 3072
    this_config.evolver_n_mem = 10


    # mem train related
    this_config.segment_num = 1 # if > 1, train memory
    this_config.remember_prob = 95
    this_config.seed=12306

    return this_config