from torch.utils.data import Dataset, Sampler
import random
import torch
import numpy as np


class RepeatIterator:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.current = low
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current > self.high:
            self.current = self.low
    
        result = self.current
        self.current += 1
        return result


class InfiniteBatchSampler(Sampler):
    def __init__(self, whole_batch_size):
        super().__init__(None)
        self.whole_batch_size = whole_batch_size

    def __iter__(self):
        # return iter(torch.arange(0, self.whole_batch_size).repeat(10000000).tolist())
        return RepeatIterator(0, self.whole_batch_size - 1)

    def __len__(self):
        return 10000000 * self.whole_batch_size


class PredictionSequentialDataset(Dataset):
    def __init__(self, numpy_dataset, block_size, min_block_size, segment_num, whole_batch_size):
        self.data = numpy_dataset
        self.block_size = block_size
        self.min_block_size = min_block_size
        self.segment_num = segment_num

        self.whole_batch_size = whole_batch_size # world_size * mini_bsz

        self.data_end_index = len(self.data) - self.block_size * (segment_num + 1) - 1
        self.batch_data_pointer = []
        for _ in range(self.whole_batch_size):
            self.batch_data_pointer.append(random.randint(0, self.data_end_index))
    
    # need to overload
    def __len__(self):
        return 10000000 * self.whole_batch_size # a large number

    # need to overload
    def __getitem__(self, idx):
        random_length = torch.randint(self.block_size - self.min_block_size, (self.segment_num + 1,)) + self.min_block_size

        x_list = []
        y_list = []

        for si in range(self.segment_num + 1):
            this_end = random_length[si] + self.batch_data_pointer[idx] # end index
            this_end = this_end if this_end < len(self.data) else len(self.data) - 1
            random_length[si] = this_end - self.batch_data_pointer[idx] # actual length

            x_list.append(torch.from_numpy(self.data[self.batch_data_pointer[idx]:this_end].astype(np.int64)))
            y_list.append(torch.from_numpy(self.data[self.batch_data_pointer[idx] + 1:this_end + 1].astype(np.int64)))

            if si < self.segment_num:
                self.batch_data_pointer[idx] = this_end if this_end < self.data_end_index else random.randint(0, self.data_end_index)
        
        # padding to (segment_num, block size)
        padding_x = x_list[0].new_full((self.segment_num + 1, self.block_size), fill_value=0)
        padding_y = y_list[0].new_full((self.segment_num + 1, self.block_size), fill_value=-1)

        for si in range(self.segment_num + 1):
            padding_x[si][:len(x_list[si])] = x_list[si]
            padding_y[si][:len(y_list[si])] = y_list[si]
        
        attention_mask = padding_y.ne(-1).int()

        return padding_x, padding_y, attention_mask, random_length
    

# # collect_fn for PredictionSequentialDataset
# def prediction_sequential_collect_fn(batch):
#     padding_x, padding_y, attention_mask, random_length = batch
#     len_delta = random_length[:, :-1].sum(-1)

#     padding_x = padding_x[:, :padding_x.size(1) - len_delta.max()]
#     padding_y = padding_y[:, :padding_y.size(1) - len_delta.max()]
#     attention_mask = attention_mask[:, :attention_mask.size(1) - len_delta.max()]

#     return padding_x, padding_y, attention_mask, random_length