def predict_config(this_config):
    this_config.compile = False
    this_config.wandb_log = False
    this_config.wandb_project = 'predict'
    this_config.wandb_run_name='1st_kl_1seg'
    this_config.wandb_notes='读取1个句子，每次预测下一个句子，并且有kl loss来指导预测。读取之前训练的一个句子的'

    this_config.peft_method = "prompt"

    # these make the total batch size be ~0.5M
    # 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
    this_config.init_from = 'scratch'
    this_config.pretrained_model_name = 'gpt2'
    # load_name = 'predict_1st_kl_1seg.pt'
    ckpt_name = 'temp.pt'

    this_config.dataset = 'openwebtext'

    this_config.batch_size = 8
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

    this_config.evolver_pad_token_id = 0
    this_config.evolver_gpt2_token_id_offset = 20 # the token id produced by gpt2 tokenizer should added by this offset

    this_config.num_target_model_layer = 12

    # mem train related
    this_config.segment_num = 2 # if > 1, train memory
    this_config.remember_prob = 95
    this_config.seed=12306

    return this_config