import torch
import math
from torch.utils.cpp_extension import load

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

async_generator_module = load(
    name="async_generator", 
    sources=["self_extend_patch/attn_method/logistic.cu"], 
    build_directory="build",
    verbose=True
)
print("Load module successfully")
def generate_logistically_grouping_position(q_max, window_size, rate = 0.01, capacity=32, device="cuda"):
    group_query_position = torch.zeros(q_max, dtype=torch.int32, device=device)  
    group_key_position = torch.zeros(q_max, dtype=torch.int32, device=device)
    
    async_generator_module.async_generator(group_query_position, group_key_position, q_max, window_size, rate, capacity)

    group_query_position = group_query_position.unsqueeze(0)
    group_key_position = group_key_position.unsqueeze(0) 
    #print(group_key_position,group_key_position.shape)
    
    return group_query_position, group_key_position

'''def generate_logistically_grouping_position(q_max, window_size, rate=0.01, capacity=32, device="cuda"):
    def logistic_func(rate, capacity, t):
        try:
            numerator = capacity * math.exp(rate * t)
            denominator = capacity + (math.exp(rate * t) - 1)
            return math.floor(numerator / denominator)
        except OverflowError:
            return capacity
    
    max_freq = 1
    count = 0
    group_key_position = []
    value = 0

    while count < q_max:
      max_freq = logistic_func(rate, capacity, value)
      group_key_position.extend([value] * max_freq)
      count += max_freq
      value += 1

    group_key_position = group_key_position[:q_max][::-1]
    group_key_position = [value - 1 - i for i in group_key_position]

    group_query_position = [0] * len(group_key_position)

    increase_vec = [0 if (i - window_size < 0) else (window_size - group_query_position[i] + group_key_position[i - window_size]) for i in range(len(group_query_position))]
    increase_vec = torch.tensor([increase_vec], device=device)

    group_query_position = torch.tensor([group_query_position], device=device)
    group_key_position = torch.tensor([group_key_position], device=device)
    group_query_position = group_query_position + increase_vec

    return group_query_position, group_key_position'''