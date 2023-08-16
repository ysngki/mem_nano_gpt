import datasets
import os
import numpy as np
from tqdm import tqdm
import torch


cache_dir = "/data/yuanhang/hf_cache"

# train_data_name = ["Books3", "Github", "FreeLaw", "Wikipedia", "Gutenberg", "HackerNews", "ArXiv", "YoutubeSubtitles"]
train_data_name = ["Books3", "Github", "FreeLaw", "Wikipedia"]

for d_name in train_data_name:
    d_path = "awettig/Pile-%s-0.5B-6K-opt" % (d_name)
    data = datasets.load_dataset(
            d_path,
            cache_dir=cache_dir,
        )

    for split, dset in data.items():
        total_batches = 128

        # iter this split to get length as whole_length
        whole_length = 0

        for batch_idx in tqdm(range(total_batches), desc=f'get length'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['attention_mask'])
            assert torch.tensor(arr_batch).ne(1).sum().item() == 0
            whole_length += len(arr_batch)
            
        print(f"{d_name} {split} whole_length: {whole_length}")

        # write to mmap
        arr_len = whole_length
        filename = os.path.join("./opt_datasets", f'{d_name}_{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        total_batches = 128
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['input_ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
# concatenate mmap
for split in ['train', 'test']:
    whole_length = 0
    for d_name in train_data_name:
        filename = os.path.join("./opt_datasets", f'{d_name}_{split}.bin')
        dtype = np.uint16
        
        m = np.memmap(filename, dtype=dtype, mode='r')
        whole_length += len(m)
    
    print(f"whole_length: {whole_length}")
    
    filename = f'{split}.bin'
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(whole_length,))
    
    # concatenate
    idx = 0
    for d_name in train_data_name:
        filename = os.path.join("./opt_datasets", f'{d_name}_{split}.bin')
        dtype = np.uint16
        
        m = np.memmap(filename, dtype=dtype, mode='r')
        arr[idx : idx + len(m)] = m
        idx += len(m)
