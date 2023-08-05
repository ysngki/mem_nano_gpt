import numpy as np
import torch


# get data for a minibatch
def get_seq_train_batch(data, data_pointer, this_batch_seg_num, block_size, min_block_size, device, device_type, plus_one=False):
    x_list = []
    y_list = []
    seg_length_list = []

    this_batch_size = len(data_pointer)

    def get_x_y_tensor_list(batch_start_point, no_update=False):
        # random segment len for each batch item
        random_length = torch.randint(block_size - min_block_size, (this_batch_size,)) + min_block_size
        segment_ends = random_length.clone()

        for bi in range(this_batch_size):
            this_end = random_length[bi] + batch_start_point[bi] # end index
            segment_ends[bi] = this_end if this_end < len(data) else len(data) - 1
            random_length[bi] = segment_ends[bi] - batch_start_point[bi] # actual length

        # (batch size, xxx)
        x = [torch.from_numpy((data[batch_start_point[bi]:segment_ends[bi]]).astype(np.int64)) for bi in range(this_batch_size)]
        y = [torch.from_numpy((data[batch_start_point[bi] + 1:segment_ends[bi] + 1]).astype(np.int64)) for bi in range(this_batch_size)]

        # update batch_start_point
        if no_update:
            pass
        else:
            for bi in range(this_batch_size):
                batch_start_point[bi] = segment_ends[bi] if segment_ends[bi] < len(data) - min_block_size * this_batch_seg_num else 0

        return x, y, batch_start_point, random_length
    

    fetch_seg_num = this_batch_seg_num + 1 if plus_one else this_batch_seg_num

    for seg_index in range(fetch_seg_num):
        # get data for this segment
        if seg_index == this_batch_seg_num: # plus one segment for prediction
            this_x, this_y, data_pointer, this_seg_length = get_x_y_tensor_list(data_pointer, True)
        else:
            this_x, this_y, data_pointer, this_seg_length = get_x_y_tensor_list(data_pointer)

        seg_length_list.append(this_seg_length)

        # padding to (batch size, block size)
        padding_x = this_x[0].new_full((this_batch_size, block_size), fill_value=0)
        padding_y = this_y[0].new_full((this_batch_size, block_size), fill_value=-1)

        for bi in range(this_batch_size):
            padding_x[bi][:len(this_x[bi])] = this_x[bi]
            padding_y[bi][:len(this_y[bi])] = this_y[bi]

        # return_offset_x.append(offset_padding_x)
        x_list.append(padding_x)
        y_list.append(padding_y)
    
    # (actual batch size, segment num, block size)
    x = torch.stack(x_list, dim=1)
    y = torch.stack(y_list, dim=1)
    attention_mask = y.ne(-1).int()
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, attention_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, attention_mask = x.to(device), y.to(device), attention_mask.to(device)
    
    # (segment num, actual batch size)
    seg_length_list = torch.stack(seg_length_list, dim=0)

    # seg_length_list: (segment num, actual batch size); x,y,attention_mask shape: (actual batch size, segment num, block size)
    return data_pointer, x, y, attention_mask, seg_length_list