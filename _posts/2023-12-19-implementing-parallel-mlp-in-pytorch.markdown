---
title: Implementing Parallel MLP in pytorch
date: 2023-12-19 02:54:00 Z
---

{% highlight python %}
import os
import torch
import time
import shutil
import itertools
from pathlib import Path
import torch.distributed as dist
from torch.multiprocessing import Process
import deepspeed
DEEPSPEED_UNIT_WORKER_TIMEOUT = 120


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

{% endhighlight %}