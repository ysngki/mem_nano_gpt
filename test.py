import os
import time
from contextlib import nullcontext
import random

from accelerate import Accelerator
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F

from my_configuration_roberta import MemoryRobertaConfig
from my_modeling_roberta import MemoryRobertaModel

from my_utils import get_seq_train_batch, print_model_size, load_pretrained_model, get_lr, estimate_predict_loss
from data_utils import PredictionSequentialDataset, InfiniteBatchSampler


data = np.memmap(os.path.join("./data/llama_openwebtext", 'train.bin'), dtype=np.uint16, mode='r')

dataset = PredictionSequentialDataset(data, 512, 128, 2, 8)
sampler = InfiniteBatchSampler(8)

dataloader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=0, pin_memory=True, shuffle=False)

# for idx, batch in enumerate(dataloader.batch_sampler):
#     print(idx)
#     print(batch)
#     print("----"*10)
    
#     if idx == 2:
#         break


for idx, batch in enumerate(dataloader):
    input_ids, labels, attention_mask, segment_lengths = batch
    print(input_ids.shape, labels.shape, attention_mask.shape, segment_lengths.shape)
    print("----"*10)
    
    if idx == 2:
        break


# print(dataset.batch_data_pointer)
# for i, batch in enumerate(dataloader):
#     padding_x, padding_y, attention_mask, random_length = batch
#     len_delta = random_length[:, :-1].sum(-1)

#     print(dataset.batch_data_pointer)
#     temp_pointer = []
#     for bi in range(8):
#         temp_pointer.append(dataset.batch_data_pointer[bi].item() - len_delta[bi].item())
#     print(temp_pointer)
#     # print(batch[0].shape)
#     # print(batch[1].shape)
#     # print(batch[2].shape)
#     # print(batch[3].shape)
#     # print(batch[3])
#     if i == 1:
#         break