import torch

def generate_sequentially_grouping_position(q_max, window_size):
    print(f"Hello {q_max}")
    max_freq = 0
    count = 0
    while count + max_freq < q_max:
        max_freq += 1
        count += max_freq

    group_key_position = [max_freq - i for i in range(max_freq) for _ in range(i+1)]
    group_key_position = group_key_position[:q_max]
    group_query_position = [pos + window_size for pos in group_key_position]

    # Trim and reserve the list
    group_query_position = group_query_position[::-1]
    group_key_position = group_key_position[::-1]
    print(len(group_query_position))

    group_query_position = torch.tensor([group_query_position], device = "cuda")
    group_key_position = torch.tensor([group_key_position], device = "cuda")
    
    return group_query_position, group_key_position