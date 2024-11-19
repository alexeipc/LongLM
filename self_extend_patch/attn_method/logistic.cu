#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ double gs_inverse_generating_function(int y, double rate, double capacity) {
    double numerator = std::log(y * capacity - y) - std::log(capacity - y);
    double denominator = rate;
    return (double) numerator / denominator;
}

struct Group {
    int first;
    int last;

    __device__ Group(int first, int last) 
        : first(first), last(last) // Use initializer list
    {
    }
};

__device__ void group_id(int id, int n, int capacity, Group* groups, int* presum, int* res) {
    int l = 0, r = capacity - 1;
    int p = 0;
    
    if (presum[1] == INT_MAX) {
        res[id - 1] = id - 1;
        return;
    }
    
    while (l < r) {
        int mid = (l + r) / 2;
        
        if (presum[mid] < id) {
            p = mid;
            l = mid + 1;
        }
        else r = mid;
    }
    
    int next_group_size = p + 1;
    int last = groups[p].last;
    int presumm = presum[p];
    
    int group_id = last + ceil((double)(id - presumm) / next_group_size);
    
    res[n - id] = n - group_id;
}

__global__ void gpu_key_group_id(int n, int capacity, Group* groups, int* presum, int* res) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (id < n) group_id(id + 1, n, capacity, groups, presum, res);
}

__global__ void gpu_query_group_id(int n, int window_size, int* group_query_position, int* group_key_position) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < window_size) group_query_position[i] = 0;
    else if (i < n) {
        group_query_position[i] = window_size + group_key_position[i - window_size];
    }
}

__global__ void freq_group(int capacity, double rate, Group* groups) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i >= 1 && i < capacity - 1) {
		double lower_bound = gs_inverse_generating_function(i, rate, capacity);
        double upper_bound = gs_inverse_generating_function(i + 1, rate, capacity);

		groups[i] = Group(ceil(lower_bound), floor(upper_bound));

		if (upper_bound == (double)floor(upper_bound)) groups[i].last--;
   }
}

void async_generator(torch::Tensor group_query_position, torch::Tensor group_key_position, int n, int window_size, double rate, double capacity) {
	Group* groups;
	int* presum;

	cudaMallocManaged(&groups, capacity * sizeof(Group));
	cudaMallocManaged(&presum, capacity * sizeof(int));

	freq_group<<<1, capacity - 1>>>(capacity, rate, groups);
	cudaDeviceSynchronize();
	groups[0].last = -1;

	for (int i = 1; i < capacity - 1; ++i) presum[i] = presum[i - 1] + i * (groups[i].last - groups[i].first + 1);

	gpu_key_group_id<<<(n + 49)/50, 50>>>(n, capacity, groups, presum, group_key_position.data_ptr<int>());
	cudaDeviceSynchronize();
    gpu_query_group_id<<<(n + 49)/50, 50>>>(n, window_size, group_query_position.data_ptr<int>() , group_key_position.data_ptr<int>());
    cudaDeviceSynchronize();
    cudaFree(groups);
    cudaFree(presum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("async_generator", &async_generator, "Description of your function");
}