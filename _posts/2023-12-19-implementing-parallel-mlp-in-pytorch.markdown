---
title: Implementing Parallel MLP in pytorch
date: 2023-12-19 02:54:00 Z
---


\`\`\`python

importosimport torchfrom torch.multiprocessing import Process\
\
defdist_init(rank, num_procs, run_func, \*func_args, \*\*func_kwargs):"""Initialize torch.distributed and execute the user function."""os.environ\["MASTER_ADDR"\] ="localhost"os.environ\["MASTER_PORT"\] ="8081"os.environ\["LOCAL_RANK"\] =str(rank)os.environ\["RANK"\] =str(rank)os.environ\["WORLD_SIZE"\] =str(num_procs)os.environ.pop("NCCL_DEBUG", None)\
init_method='tcp://'init_method\+=os.environ\["MASTER_ADDR"\] \+':'\+os.environ\["MASTER_PORT"\]\
\
torch.distributed.init_process_group(backend="nccl",world_size=num_procs,rank=rank,init_method=init_method)\
iftorch.cuda.is_available():torch.cuda.set_device(rank)\
run_func(\*func_args, \*\*func_kwargs)\
\# make sure all ranks finish at the same timetorch.distributed.barrier()# tear down after test completestorch.distributed.destroy_process_group()\
defdist_launcher(num_procs, run_func, \*func_args, \*\*func_kwargs):"""Launch processes and gracefully handle failures."""\
\# Spawn all workers on subprocesses.processes= \[\]forlocal_rankinrange(num_procs):p= Process(target=dist_init,args=(local_rank, num_procs, run_func, \*func_args),kwargs=func_kwargs)p.start()processes.append(p)\
\# Now loop and wait for a test to complete. The spin-wait here isn't a big# deal because the number of processes will be O(#GPUs) << O(#CPUs).any_done=Falsewhilenotany_done:forpinprocesses:ifnotp.is_alive():any_done=Truebreak\
\# Wait for all other processes to completeforpinprocesses:p.join(200)\
failed= \[(rank, p) forrank, pinenumerate(processes) ifp.exitcode !=0\]forrank, pinfailed:# If it still hasn't terminated, kill it because it hung.ifp.exitcode isNone:p.terminate()print(f"Worker {rank} hung.")ifp.exitcode <0:print(f"Worker {rank} killed by signal {-p.exitcode}")ifp.exitcode >0:print(f"Worker {rank} exited with code {p.exitcode}")\
\`\`\`