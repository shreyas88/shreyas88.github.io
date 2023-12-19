---
title: Implementing Parallel MLP in pytorch
date: 2023-12-19 02:54:00 Z
---

Tensor parallel MLP is one of the building blocks of modern distributed transformer based models. 

Typically we see the following kinds of parallelism 
### Tensor parallelism 
Splits the tensor computation across various GPU nodes. Typically this is used within a server datacenter node since this involves all reduce operations(over nvLink network as opposed to slower interconnects) which are expensive collective communication operations.
### Pipeline parallelism
Splits the sequential layers of the transformer model across different GPU nodes. This is analogous to the pipelining concept in computer architecture.  
### Data parallelism
Split the work across the batch axis and reduce the gradients using all-reduce operation. 

## Overview
Tensor parallel MLP block is common in high performance transformer architecture. We typically have a self attention block followed by MLP blocker interspersed with the dropout/layer norm layers. Here we focus on implementing the tensor parallel MLP layer. By splitting the matrix, we can reduce the memory bandwidth requirements(memory bound) as we cut down the size of activation and weight matrix and hope to get a linear speedup by increasing the number of GPU's.

This has the following operations. 

We represent the input tensor using `(B, T, D)` where
`B: Batch size`
`T: Sequence dimension`
`D: Hidden dimension`

1. `Matrix multiplication project up` : Project from hidden dimension `D` to `4D` ie `B,T,D -> B,T,4D`
2. `Non linearity`: Gelu/relu layer applied at the hidden dimension level 
3. `Matrix multiplication project down`: Project back down from hidden dimension `4D` to `D` ie `B,T,4D -> B,T,D`

Pseudo-code for tensor parallel MLP layer
```matlab
% Split the matrix-up weight vector along the column dimension. 
% Note that splitting the input X along row and A along column 
% doesn't work due to the non-linearity after this layer. 
[Y1 Y2] = X [ A1, A2 ]

% apply the gelu independently across the tensor parallel degree
Z1,Z2 = gelu(Y1, Y2)

% split the second weight vector along the row 
% eg if orig (m,n)*(n,k) -> (m, n/2) * (n/2, k) = (m,k)
[Z1 Z2 ] [ B1
           B2 ]
Note that even though the dimension match in the final step
we still need to add the reduce the matrix across nodes
to get the final result. 
```

The following implementation is simplified as adapted from the Megatron code base simplified for educational purpose. 
1. We would be using a single serve dual GPU setup to simplify the setup.
2. We do NOT handle any weight initialization which would typically be required in real applications.
 
## Setup distributed torch application
Initialize the simplified torch distributed setup to enable collective communications. This is covered in more detailed in the [official pytorch guide](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

1. `dist_launcher` spawns multiple processes and handles the synchronization loop
2. `dist_init` sets up the distributed process group which is used for collective communication ops such as all-reduce, all-to-all etc.

```python
import os
import torch
import time
import shutil
import itertools
from pathlib import Path
import torch.distributed as dist
from torch.multiprocessing import Process


def dist_init(rank, num_procs, run_func, *func_args, **func_kwargs):
    """Initialize torch.distributed and execute the user function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8081"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(num_procs)
    os.environ.pop("NCCL_DEBUG", None)

    init_method = 'tcp://' 
    init_method +=  os.environ["MASTER_ADDR"] + ':' + os.environ["MASTER_PORT"]


    torch.distributed.init_process_group(
        backend="nccl",
        world_size=num_procs,
        rank=rank,
        init_method=init_method)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    run_func(*func_args, **func_kwargs)

    # make sure all ranks finish at the same time
    torch.distributed.barrier()
    # tear down after test completes
    torch.distributed.destroy_process_group()

def dist_launcher(num_procs, run_func, *func_args, **func_kwargs):
    """Launch processes and gracefully handle failures."""

    # Spawn all workers on subprocesses.
    processes = []
    for local_rank in range(num_procs):
        p = Process(target=dist_init,
                    args=(local_rank, num_procs, run_func, *func_args),
                    kwargs=func_kwargs)
        p.start()
        processes.append(p)

    # Now loop and wait for a test to complete. The spin-wait here isn't a big
    # deal because the number of processes will be O(#GPUs) << O(#CPUs).
    any_done = False
    while not any_done:
        for p in processes:
            if not p.is_alive():
                any_done = True
                break

    # Wait for all other processes to complete
    for p in processes:
        p.join(200)

    failed = [(rank, p) for rank, p in enumerate(processes) if p.exitcode != 0]
    for rank, p in failed:
        # If it still hasn't terminated, kill it because it hung.
        if p.exitcode is None:
            p.terminate()
            print(f"Worker {rank} hung.")
        if p.exitcode < 0:
            print(f"Worker {rank} killed by signal {-p.exitcode}")
        if p.exitcode > 0:
            print(f"Worker {rank} exited with code {p.exitcode}")
```

We can set up a dummy test loop like below which should print the output on the two processes. We will extend this harness later on.

```python
def dummy():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(torch.cuda.get_device())
    print(rank)

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    dist_launcher(2,dummy)
```

## Column parallel layer 
`[Y1 Y2] = X [ A1, A2 ]`
The forward pass is straightforward as we simply need to need to output the matrix multiplication result corresponding to split column weight matrix. 

However, in the backward pass gradient all-reduce op is needed to sum the gradient contributions from the branches going back from `X * A1` and `X * A2`. In order to override the backward pass behavior, we will need to implement the `torch.autograd.Function` function and override the backward pass.


```python
import torch
import torch.nn as nn

class LinearColumnWithAsyncGrad(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,input,weight,bias):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight.t())
        return output + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # grad_output is (batch, T, output_size_partition)
        # input is (batch, T, input_size_partition)
        input, weight = ctx.saved_tensors

        # (batch, output_size_partition) * (output_size_partition, input_size) -> (batch, input_size)   
        #  (batch, T, input_size) = (batch, T, 1) * (1, input_size)  
        grad_input = grad_output.matmul(weight)

        # Asynchronous all-reduce
        handle = torch.distributed.all_reduce(grad_input, async_op=True)

        # (batch*T, output_size_partition) * (batch*T, input_size_partition) -> (output_size_partition, input_size_partition)
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0)
        handle.wait()
        return grad_input, grad_weight, grad_bias

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, weight_per_rank, bias_per_rank):
        super(ColumnParallelLinear, self).__init__()
        self.weight = nn.Parameter(weight_per_rank)
        self.bias = nn.Parameter(bias_per_rank)

    def forward(self, input_: torch.Tensor):
        return LinearColumnWithGradReduce.apply(input_, self.weight, self.bias)


```

## Relu layer
Relu layer should continue to work as expected normally

## Row parallel layer
In the Row Parallel Layer, the weight matrix is split along the row dimension.  

```python
class LinearRowWithTensorReduce(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,input,weight,bias):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight.t()) + bias
        # all reduce along tensor parallel dimension
        return torch.distributed.all_reduce(output)
        

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.
    its second dimension as Z =   X  [ Y1
                                       Y2 ]
    """
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.
    its second dimension as Z =   X  [ Y1
                                       Y2 ]
    """
    def __init__(self, weight_per_rank, bias_per_rank):
        super(RowParallelLinear, self).__init__()
        self.weight = nn.Parameter(weight_per_rank)
        self.bias = nn.Parameter(bias_per_rank)

    def forward(self, input_: torch.Tensor):
        LinearRowWithTensorReduce.apply(input_, self.weight, self.bias)

    def forward(self, input_: torch.Tensor):
        LinearRowWithTensorReduce.apply(input_, self.weight, self.bias)
```

