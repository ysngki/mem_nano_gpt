from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    OPTForCausalLM
)
import transformers

cache_dir = "/data/yuanhangyang/transformers_cache"
model_url = "facebook/opt-2.7b"

tokenizer = AutoTokenizer.from_pretrained(model_url, cache_dir=cache_dir)
config = AutoConfig.from_pretrained(model_url, cache_dir=cache_dir)
model = OPTForCausalLM.from_pretrained(model_url, config=config, cache_dir=cache_dir)

