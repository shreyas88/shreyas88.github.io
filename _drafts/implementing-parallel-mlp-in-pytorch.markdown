---
title: Implementing Parallel MLP in pytorch
date: 2023-12-19 02:54:00 Z
---

Tensor parallel MLP is one of the building blocks of modern distributed transformer based models. 

Typically we see the following kinds of parallelism:
1. *Tensor parallelism*: Splits the tensor computation across various GPU nodes. Typically this is used within a server datacenter node since this involves all reduce operations(over nvLink network as opposed to slower interconnects) which are expensive collective communication operations.
2. *Pipeline parallelism*: Splits the sequential layers of the transformer model across different GPU nodes. This is analogous to the pipelining concept in computer architecture.  
3. *Data parallelism*: Split the work across the batch axis and reduce the gradients using all-reduce operation. 

```python
import os
import torch
import time
import shutil
import itertools
from pathlib import Path
import torch.distributed as dist
from torch.multiprocessing import Process
import deepspeed

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
```

adsad