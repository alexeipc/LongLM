import torch

def generate_sequentially_grouping_position(q_max, window_size):
    #print(f"Hello {q_max}")
    max_freq = 0
    count = 0
    while count < q_max:
        max_freq += 1
        count += max_freq

    group_key_position = [max_freq - i for i in range(max_freq) for _ in range(i+1)]
    group_key_position = group_key_position[:q_max]
    group_query_position = [pos + window_size for pos in group_key_position]

    # Trim and reserve the list
    group_query_position = group_query_position[::-1]
    group_key_position = group_key_position[::-1]

    group_query_position = torch.tensor([group_query_position], device = "cuda")
    group_key_position = torch.tensor([group_key_position], device = "cuda")
    
    return group_query_position, group_key_position

def generate_exponentially_grouping_position(q_max, window_size, base=2):
    max_freq = 1
    count = 0
    group_key_position = []
    value = 0

    # Generate positions with exponential groupings
    while count < q_max:
        group_key_position.extend([value] * max_freq)
        count += max_freq
        max_freq *= base
        value += 1

    # Trim to the exact number of required positions
    group_key_position = group_key_position[:q_max]
    group_query_position = [pos + window_size for pos in group_key_position]

    # Reverse lists to match required order
    group_query_position = torch.tensor([group_query_position[::-1]], device="cuda")
    group_key_position = torch.tensor([group_key_position[::-1]], device="cuda")

    return group_query_position, group_key_position