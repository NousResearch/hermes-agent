---
title: "Pytorch Fsdp"
sidebar_label: "Pytorch Fsdp"
description: "PyTorch FSDP 全分片数据并行训练专家指导 - 参数分片、混合精度、CPU 卸载、FSDP2"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pytorch Fsdp

PyTorch FSDP 全分片数据并行训练专家指导 - 参数分片、混合精度、CPU 卸载、FSDP2

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/pytorch-fsdp` 安装 |
| 路径 | `optional-skills/mlops/pytorch-fsdp` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `torch>=2.0`, `transformers` |
| 平台 | linux, macos |
| 标签 | `Distributed Training`, `PyTorch`, `FSDP`, `Data Parallel`, `Sharding`, `Mixed Precision`, `CPU Offloading`, `FSDP2`, `Large-Scale Training` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 看到的指令内容。
:::

# Pytorch-Fsdp Skill

基于官方文档生成的 pytorch-fsdp 开发综合辅助。

## 何时使用此 Skill

以下情况应触发此 skill：
- 使用 pytorch-fsdp
- 询问 pytorch-fsdp 功能或 API
- 实现 pytorch-fsdp 解决方案
- 调试 pytorch-fsdp 代码
- 学习 pytorch-fsdp 最佳实践

## 快速参考

### 常用模式

**模式 1：** 通用 Join 上下文管理器（Generic Join Context Manager）# 创建于：2025年6月6日 | 最后更新：2025年6月6日 通用 join 上下文管理器用于在输入不均匀时进行分布式训练。本页概述相关类的 API：Join、Joinable 和 JoinHook。教程请参见《使用 Join 上下文管理器进行不均匀输入的分布式训练》。class torch.distributed.algorithms.Join(joinables, enable=True, throw_on_early_termination=False, **kwargs)[source]# 该类定义通用 join 上下文管理器，允许在进程 join 后调用自定义 hook。这些 hook 应模拟未 join 进程的集合通信，以防止挂起和报错，并确保算法正确性。有关 hook 定义的详细信息，请参见 JoinHook。警告：上下文管理器要求每个参与的 Joinable 在其自身的每次迭代集合通信之前调用 notify_join_context() 方法，以确保正确性。警告：上下文管理器要求所有 JoinHook 对象中的 process_group 属性相同。如果存在多个 JoinHook 对象，则使用第一个的设备。进程组和设备信息用于检查未 join 的进程，以及在启用 throw_on_early_termination 时通知进程抛出异常，两者均使用 all-reduce。参数：joinables (List[Joinable]) – 参与的 Joinable 列表；其 hook 按给定顺序迭代。enable (bool) – 启用不均匀输入检测的标志；设为 False 将禁用上下文管理器功能，仅当用户确认输入不会不均匀时才应设置（默认：True）。throw_on_early_termination (bool) – 控制检测到不均匀输入时是否抛出异常的标志（默认：False）。示例：>>> import os >>> import torch >>> import torch.distributed as dist >>> import torch.multiprocessing as mp >>> import torch.nn.parallel.DistributedDataParallel as DDP >>> import torch.distributed.optim.ZeroRedundancyOptimizer as ZeRO >>> from torch.distributed.algorithms.join import Join >>> >>> # 在每个 spawned worker 上 >>> def worker(rank): >>> dist.init_process_group("nccl", rank=rank, world_size=2) >>> model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank]) >>> optim = ZeRO(model.parameters(), torch.optim.Adam, lr=0.01) >>> # Rank 1 比 rank 0 多一个输入 >>> inputs = [torch.tensor([1.]).to(rank) for _ in range(10 + rank)] >>> with Join([model, optim]): >>> for input in inputs: >>> loss = model(input).sum() >>> loss.backward() >>> optim.step() >>> # 所有 rank 均可到达此处，不会挂起或报错 static notify_join_context(joinable)[source]# 通知 join 上下文管理器调用进程尚未 join。然后，如果 throw_on_early_termination=True，检查是否检测到不均匀输入（即某个进程已 join），若是则抛出异常。此方法应在 Joinable 对象的每次迭代集合通信之前调用。例如，应在 DistributedDataParallel 的前向传播开始时调用。只有传入上下文管理器的第一个 Joinable 对象会在此方法中执行集合通信，其他对象调用此方法为空操作。参数：joinable (Joinable) – 调用此方法的 Joinable 对象。返回：如果 joinable 是传入上下文管理器的第一个，则返回用于通知上下文管理器进程尚未 join 的 all-reduce 异步工作句柄；否则返回 None。class torch.distributed.algorithms.Joinable[source]# 定义可 join 类的抽象基类。可 join 类（继承自 Joinable）应实现 join_hook()（返回 JoinHook 实例），以及 join_device() 和 join_process_group()（分别返回设备和进程组信息）。abstract property join_device: device# 返回执行 join 上下文管理器所需集合通信的设备。abstract join_hook(**kwargs)[source]# 返回给定 Joinable 的 JoinHook 实例。参数：kwargs (dict) – 包含在运行时修改 join hook 行为的关键字参数的字典；共享同一 join 上下文管理器的所有 Joinable 实例将收到相同的 kwargs 值。返回类型：JoinHook abstract property join_process_group: Any# 返回 join 上下文管理器本身所需集合通信的进程组。class torch.distributed.algorithms.JoinHook[source]# 定义 join hook，在 join 上下文管理器中提供两个入口点。入口点：主 hook（在存在未 join 进程时重复调用）和后置 hook（在所有进程均已 join 后调用一次）。要为通用 join 上下文管理器实现 join hook，请定义一个继承自 JoinHook 的类，并根据需要重写 main_hook() 和 post_hook()。main_hook()[source]# 在存在未 join 进程时调用此 hook，以模拟训练迭代中的集合通信。训练迭代即一次前向传播、反向传播和优化器步骤。post_hook(is_last_joiner)[source]# 在所有进程均已 join 后调用此 hook。传入额外的布尔参数 is_last_joiner，指示该 rank 是否是最后 join 的之一。参数：is_last_joiner (bool) – 如果该 rank 是最后 join 的之一则为 True；否则为 False。

```
Join
```

**模式 2：** 分布式通信包 - torch.distributed# 创建于：2017年7月12日 | 最后更新：2025年9月4日 注意：有关分布式训练所有功能的简要介绍，请参阅 PyTorch 分布式概述。后端（Backends）# torch.distributed 支持四种内置后端，各具不同能力。下表显示每种后端在 CPU 或 GPU 上可用的函数。对于 NCCL，GPU 指 CUDA GPU；对于 XCCL，GPU 指 XPU GPU。MPI 仅在构建 PyTorch 时使用的实现支持 CUDA 的情况下才支持 CUDA。后端 gloo mpi nccl xccl 设备 CPU GPU CPU GPU CPU GPU CPU GPU send ✓ ✘ ✓ ? ✘ ✓ ✘ ✓ recv ✓ ✘ ✓ ? ✘ ✓ ✘ ✓ broadcast ✓ ✓ ✓ ? ✘ ✓ ✘ ✓ all_reduce ✓ ✓ ✓ ? ✘ ✓ ✘ ✓ reduce ✓ ✓ ✓ ? ✘ ✓ ✘ ✓ all_gather ✓ ✓ ✓ ? ✘ ✓ ✘ ✓ gather ✓ ✓ ✓ ? ✘ ✓ ✘ ✓ scatter ✓ ✓ ✓ ? ✘ ✓ ✘ ✓ reduce_scatter ✓ ✓ ✘ ✘ ✘ ✓ ✘ ✓ all_to_all ✓ ✓ ✓ ? ✘ ✓ ✘ ✓ barrier ✓ ✘ ✓ ? ✘ ✓ ✘ ✓ PyTorch 内置后端# PyTorch 分布式包支持 Linux（稳定）、MacOS（稳定）和 Windows（原型）。Linux 默认构建并包含 Gloo 和 NCCL 后端（NCCL 仅在使用 CUDA 构建时包含）。MPI 是可选后端，只能在从源码构建 PyTorch 时包含（例如在已安装 MPI 的主机上构建 PyTorch）。注意：自 PyTorch v1.8 起，Windows 支持除 NCCL 外的所有集合通信后端。如果 init_process_group() 的 init_method 参数指向文件，则必须遵循以下格式：本地文件系统，init_method="file:///d:/tmp/some_file" 共享文件系统，init_method="file://////&#123;machine_name&#125;/&#123;share_folder_name&#125;/some_file" 与 Linux 平台相同，可通过设置环境变量 MASTER_ADDR 和 MASTER_PORT 启用 TcpStore。使用哪种后端？# 过去我们经常被问到："应该使用哪种后端？"。经验法则：使用 NCCL 后端进行 CUDA GPU 分布式训练。使用 XCCL 后端进行 XPU GPU 分布式训练。使用 Gloo 后端进行 CPU 分布式训练。带 InfiniBand 互连的 GPU 主机：使用 NCCL，因为它是目前唯一支持 InfiniBand 和 GPUDirect 的后端。带以太网互连的 GPU 主机：使用 NCCL，因为它目前提供最佳的分布式 GPU 训练性能，尤其适用于多进程单节点或多节点分布式训练。如果遇到 NCCL 问题，使用 Gloo 作为备选（注意 Gloo 在 GPU 上目前比 NCCL 慢）。带 InfiniBand 互连的 CPU 主机：如果 InfiniBand 启用了 IP over IB，使用 Gloo；否则使用 MPI。带以太网互连的 CPU 主机：使用 Gloo，除非有特定原因使用 MPI。常用环境变量# 选择使用的网络接口# 默认情况下，NCCL 和 Gloo 后端都会尝试自动找到合适的网络接口。如果自动检测的接口不正确，可通过以下环境变量覆盖（适用于各自后端）：NCCL_SOCKET_IFNAME，例如 export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME，例如 export GLOO_SOCKET_IFNAME=eth0 使用 Gloo 后端时，可通过逗号分隔指定多个接口，如：export GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3。后端将以轮询方式在这些接口上分发操作。所有进程必须在此变量中指定相同数量的接口。其他 NCCL 环境变量# 调试 - 如果 NCCL 失败，可设置 NCCL_DEBUG=INFO 打印明确的警告信息及基本的 NCCL 初始化信息。也可使用 NCCL_DEBUG_SUBSYS 获取 NCCL 特定方面的更多详情。例如，NCCL_DEBUG_SUBSYS=COLL 将打印集合调用的日志，在调试挂起（尤其是由集合类型或消息大小不匹配引起的挂起）时很有帮助。如果拓扑检测失败，设置 NCCL_DEBUG_SUBSYS=GRAPH 可检查详细检测结果，并在需要 NCCL 团队进一步帮助时保存为参考。性能调优 - NCCL 根据拓扑检测自动调优，以减少用户调优工作量。在某些基于 socket 的系统上，用户仍可尝试调整 NCCL_SOCKET_NTHREADS 和 NCCL_NSOCKS_PERTHREAD 以提高 socket 网络带宽。这两个环境变量已由 NCCL 针对部分云提供商（如 AWS 或 GCP）预调优。完整的 NCCL 环境变量列表请参阅 NVIDIA NCCL 官方文档。还可使用 torch.distributed.ProcessGroupNCCL.NCCLConfig 和 torch.distributed.ProcessGroupNCCL.Options 进一步调优 NCCL 通信器。在解释器中使用 help（例如 help(torch.distributed.ProcessGroupNCCL.NCCLConfig)）了解更多信息。基础知识# torch.distributed 包为在一台或多台机器上运行的多个计算节点之间的多进程并行提供 PyTorch 支持和通信原语。torch.nn.parallel.DistributedDataParallel() 类基于此功能，作为任意 PyTorch 模型的包装器提供同步分布式训练。这与 Multiprocessing 包 - torch.multiprocessing 和 torch.nn.DataParallel() 提供的并行方式不同，它支持多台网络连接的机器，且用户必须为每个进程显式启动一份主训练脚本的副本。在单机同步场景下，torch.distributed 或 torch.nn.parallel.DistributedDataParallel() 包装器相比其他数据并行方式（包括 torch.nn.DataParallel()）仍有优势：每个进程维护自己的优化器，并在每次迭代中执行完整的优化步骤。虽然这看起来冗余（因为梯度已在进程间聚合并平均，对每个进程而言是相同的），但这意味着不需要参数广播步骤，减少了节点间张量传输的时间。每个进程包含独立的 Python 解释器，消除了从单个 Python 进程驱动多个执行线程、模型副本或 GPU 时产生的额外解释器开销和"GIL 争用"。这对大量使用 Python 运行时的模型（包括带循环层或许多小组件的模型）尤为重要。初始化# 在调用任何其他方法之前，需要使用 torch.distributed.init_process_group() 或 torch.distributed.device_mesh.init_device_mesh() 函数初始化该包。两者均会阻塞直到所有进程加入。警告：初始化不是线程安全的。进程组创建应在单个线程中执行，以防止跨 rank 的 UUID 分配不一致，以及防止初始化期间可能导致挂起的竞争条件。torch.distributed.is_available()[source]# 如果分布式包可用则返回 True。否则，torch.distributed 不会暴露任何其他 API。目前，torch.distributed 在 Linux、MacOS 和 Windows 上可用。从源码构建 PyTorch 时设置 USE_DISTRIBUTED=1 以启用。目前默认值：Linux 和 Windows 为 USE_DISTRIBUTED=1，MacOS 为 USE_DISTRIBUTED=0。返回类型：bool torch.distributed.init_process_group(backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name='', pg_options=None, device_id=None)[source]# 初始化默认分布式进程组，同时也会初始化分布式包。初始化进程组有两种主要方式：显式指定 store、rank 和 world_size。指定 init_method（URL 字符串），指示在何处/如何发现对等节点。可选择性地指定 rank 和 world_size，或将所有必需参数编码到 URL 中并省略它们。如果两者均未指定，则假定 init_method 为 "env://"。参数：backend (str 或 Backend，可选) – 要使用的后端。根据构建时配置，有效值包括 mpi、gloo、nccl、ucc、xccl 或由第三方插件注册的后端。自 2.6 起，如果未提供 backend，c10d 将使用为 device_id 关键字参数（如果提供）所指示的设备类型注册的后端。目前已知的默认注册：cuda 对应 nccl，cpu 对应 gloo，xpu 对应 xccl。如果既未提供 backend 也未提供 device_id，c10d 将检测运行时机器上的加速器，并使用为该检测到的加速器（或 cpu）注册的后端。此字段可以小写字符串形式给出（例如 "gloo"），也可通过 Backend 属性访问（例如 Backend.GLOO）。如果在 nccl 后端下每台机器使用多个进程，每个进程必须对其使用的每个 GPU 拥有独占访问权，因为进程间共享 GPU 可能导致死锁或 NCCL 无效使用。ucc 后端为实验性。可通过 get_default_backend_for_device() 查询设备的默认后端。init_method (str，可选) – 指定如何初始化进程组的 URL。如果未指定 init_method 或 store，默认为 "env://"。与 store 互斥。world_size (int，可选) – 参与作业的进程数。指定 store 时必填。rank (int，可选) – 当前进程的 rank（应为 0 到 world_size-1 之间的数字）。指定 store 时必填。store (Store，可选) – 所有 worker 均可访问的键值存储，用于交换连接/地址信息。与 init_method 互斥。timeout (timedelta，可选) – 针对进程组执行的操作的超时时间。NCCL 默认值为 10 分钟，其他后端为 30 分钟。超过此时间后，集合操作将被异步中止，进程将崩溃。这是因为 CUDA 执行是异步的，继续执行用户代码不再安全，因为失败的异步 NCCL 操作可能导致后续 CUDA 操作在损坏的数据上运行。设置 TORCH_NCCL_BLOCKING_WAIT 时，进程将阻塞并等待此超时。group_name (str，可选，已弃用) – 组名。此参数被忽略。pg_options (ProcessGroupOptions，可选) – 进程组选项，指定在构建特定进程组时需要传入的额外选项。目前仅支持 nccl 后端的 ProcessGroupNCCL.Options，可指定 is_high_priority_stream 以便 nccl 后端在有计算内核等待时选择高优先级 cuda 流。其他可用的 nccl 配置选项，请参见 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t device_id (torch.device | int，可选) – 此进程将使用的单个特定设备，允许进行后端特定的优化。目前仅在 NCCL 下有两个效果：通信器立即形成（立即调用 ncclCommInit* 而非正常的延迟调用），子组将在可能时使用 ncclCommSplit 以避免不必要的组创建开销。如果想尽早了解 NCCL 初始化错误，也可使用此字段。如果提供 int，API 假定将使用编译时的加速器类型。注意：要启用 backend == Backend.MPI，需要在支持 MPI 的系统上从源码构建 PyTorch。注意：多后端支持为实验性。目前未指定后端时，将同时创建 gloo 和 nccl 后端。gloo 后端用于 CPU 张量的集合操作，nccl 后端用于 CUDA 张量的集合操作。可通过传入格式为 "&lt;device_type>:&lt;backend_name>,&lt;device_type>:&lt;backend_name>" 的字符串指定自定义后端，例如 "cpu:gloo,cuda:custom_backend"。torch.distributed.device_mesh.init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None, backend_override=None)[source]# 根据 device_type、mesh_shape 和 mesh_dim_names 参数初始化 DeviceMesh。这将创建一个具有 n 维数组布局的 DeviceMesh，其中 n 为 mesh_shape 的长度。如果提供了 mesh_dim_names，每个维度将被标记为 mesh_dim_names[i]。注意：init_device_mesh 遵循 SPMD 编程模型，即相同的 PyTorch Python 程序在集群中所有进程/rank 上运行。确保 mesh_shape（描述设备布局的 nD 数组的维度）在所有 rank 上完全相同。不一致的 mesh_shape 可能导致挂起。注意：如果未找到进程组，init_device_mesh 将在后台初始化分布式通信所需的分布式进程组。参数：device_type (str) – mesh 的设备类型。目前支持："cpu"、"cuda/cuda-like"、"xpu"。不允许传入带 GPU 索引的设备类型，如 "cuda:0"。mesh_shape (Tuple[int]) – 定义描述设备布局的多维数组维度的元组。mesh_dim_names (Tuple[str]，可选) – 分配给描述设备布局的多维数组各维度的 mesh 维度名称元组。其长度必须与 mesh_shape 的长度匹配。mesh_dim_names 中的每个字符串必须唯一。backend_override (Dict[int | str, tuple[str, Options] | str | Options]，可选) – 对将为每个 mesh 维度创建的部分或全部 ProcessGroup 的覆盖。每个键可以是维度的索引或其名称（如果提供了 mesh_dim_names）。每个值可以是包含后端名称及其选项的元组，或仅其中一个组件（另一个将设为默认值）。返回：表示设备布局的 DeviceMesh 对象。返回类型：DeviceMesh 示例：>>> from torch.distributed.device_mesh import init_device_mesh >>> >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,)) >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp")) torch.distributed.is_initialized()[source]# 检查默认进程组是否已初始化。返回类型：bool torch.distributed.is_mpi_available()[source]# 检查 MPI 后端是否可用。返回类型：bool torch.distributed.is_nccl_available()[source]# 检查 NCCL 后端是否可用。返回类型：bool torch.distributed.is_gloo_available()[source]# 检查 Gloo 后端是否可用。返回类型：bool torch.distributed.distributed_c10d.is_xccl_available()[source]# 检查 XCCL 后端是否可用。返回类型：bool torch.distributed.is_torchelastic_launched()[source]# 检查此进程是否通过 torch.distributed.elastic（即 torchelastic）启动。使用 TORCHELASTIC_RUN_ID 环境变量的存在作为代理，判断当前进程是否通过 torchelastic 启动。这是合理的代理，因为 TORCHELASTIC_RUN_ID 映射到 rendezvous id，该 id 始终是非空值，用于对等发现的作业 id。返回类型：bool torch.distributed.get_default_backend_for_device(device)[source]# 返回给定设备的默认后端。参数：device (Union[str, torch.device]) – 要获取默认后端的设备。返回：给定设备的默认后端（小写字符串）。返回类型：str 目前支持三种初始化方法：TCP 初始化# 使用 TCP 初始化有两种方式，均需要所有进程可访问的网络地址和所需的 world_size。第一种方式需要指定属于 rank 0 进程的地址。此初始化方法要求所有进程手动指定 rank。注意，最新的分布式包不再支持多播地址，group_name 也已弃用。import torch.distributed as dist # 使用其中一台机器的地址 dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4) 共享文件系统初始化# 另一种初始化方法使用组中所有机器均可见的共享文件系统，以及所需的 world_size。URL 应以 file:// 开头，并包含共享文件系统上不存在的文件路径（在已存在的目录中）。文件系统初始化将在文件不存在时自动创建，但不会删除该文件。因此，用户有责任确保在下次以相同文件路径/名称调用 init_process_group() 之前清理该文件。注意，最新的分布式包不再支持自动 rank 分配，group_name 也已弃用。警告：此方法假定文件系统支持使用 fcntl 加锁 - 大多数本地系统和 NFS 支持此功能。警告：此方法将始终创建文件，并在程序结束时尽力清理和删除该文件。换言之，每次使用文件初始化方法都需要一个全新的空文件才能成功初始化。如果再次使用之前初始化留下的同一文件（恰好未被清理），这是意外行为，通常会导致死锁和失败。因此，即使此方法会尽力清理文件，如果自动删除失败，用户有责任确保在训练结束时删除该文件，以防止下次被重复使用。如果计划多次以相同文件名调用 init_process_group()，这一点尤为重要。换言之，如果文件未被删除/清理，再次以该文件调用 init_process_group() 将预期失败。经验法则：确保每次调用 init_process_group() 时文件不存在或为空。import torch.distributed as dist # rank 应始终指定 dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile', world_size=4, rank=args.rank) 环境变量初始化# 此方法从环境变量读取配置，允许完全自定义获取信息的方式。需要设置的变量：MASTER_PORT - 必填；必须是 rank 0 机器上的空闲端口 MASTER_ADDR - 必填（rank 0 除外）；rank 0 节点的地址 WORLD_SIZE - 必填；可在此处设置，也可在调用 init 函数时设置 RANK - 必填；可在此处设置，也可在调用 init 函数时设置 rank 0 的机器将用于建立所有连接。这是默认方法，意味着不必指定 init_method（或可设为 env://）。改善初始化时间# TORCH_GLOO_LAZY_INIT - 按需建立连接，而非使用全网格，可大幅改善非 all2all 操作的初始化时间。

```
torch.distributed
```

**模式 3：** 初始化# 在调用任何其他方法之前，需要使用 torch.distributed.init_process_group() 或 torch.distributed.device_mesh.init_device_mesh() 函数初始化该包。两者均会阻塞直到所有进程加入。警告：初始化不是线程安全的。进程组创建应在单个线程中执行，以防止跨 rank 的 UUID 分配不一致，以及防止初始化期间可能导致挂起的竞争条件。torch.distributed.is_available()[source]# 如果分布式包可用则返回 True。否则，torch.distributed 不会暴露任何其他 API。目前，torch.distributed 在 Linux、MacOS 和 Windows 上可用。从源码构建 PyTorch 时设置 USE_DISTRIBUTED=1 以启用。目前默认值：Linux 和 Windows 为 USE_DISTRIBUTED=1，MacOS 为 USE_DISTRIBUTED=0。返回类型：bool torch.distributed.init_process_group(backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name='', pg_options=None, device_id=None)[source]# 初始化默认分布式进程组，同时也会初始化分布式包。初始化进程组有两种主要方式：显式指定 store、rank 和 world_size。指定 init_method（URL 字符串），指示在何处/如何发现对等节点。可选择性地指定 rank 和 world_size，或将所有必需参数编码到 URL 中并省略它们。如果两者均未指定，则假定 init_method 为 "env://"。参数：backend (str 或 Backend，可选) – 要使用的后端。根据构建时配置，有效值包括 mpi、gloo、nccl、ucc、xccl 或由第三方插件注册的后端。自 2.6 起，如果未提供 backend，c10d 将使用为 device_id 关键字参数（如果提供）所指示的设备类型注册的后端。目前已知的默认注册：cuda 对应 nccl，cpu 对应 gloo，xpu 对应 xccl。如果既未提供 backend 也未提供 device_id，c10d 将检测运行时机器上的加速器，并使用为该检测到的加速器（或 cpu）注册的后端。此字段可以小写字符串形式给出（例如 "gloo"），也可通过 Backend 属性访问（例如 Backend.GLOO）。如果在 nccl 后端下每台机器使用多个进程，每个进程必须对其使用的每个 GPU 拥有独占访问权，因为进程间共享 GPU 可能导致死锁或 NCCL 无效使用。ucc 后端为实验性。可通过 get_default_backend_for_device() 查询设备的默认后端。init_method (str，可选) – 指定如何初始化进程组的 URL。如果未指定 init_method 或 store，默认为 "env://"。与 store 互斥。world_size (int，可选) – 参与作业的进程数。指定 store 时必填。rank (int，可选) – 当前进程的 rank（应为 0 到 world_size-1 之间的数字）。指定 store 时必填。store (Store，可选) – 所有 worker 均可访问的键值存储，用于交换连接/地址信息。与 init_method 互斥。timeout (timedelta，可选) – 针对进程组执行的操作的超时时间。NCCL 默认值为 10 分钟，其他后端为 30 分钟。超过此时间后，集合操作将被异步中止，进程将崩溃。这是因为 CUDA 执行是异步的，继续执行用户代码不再安全，因为失败的异步 NCCL 操作可能导致后续 CUDA 操作在损坏的数据上运行。设置 TORCH_NCCL_BLOCKING_WAIT 时，进程将阻塞并等待此超时。group_name (str，可选，已弃用) – 组名。此参数被忽略。pg_options (ProcessGroupOptions，可选) – 进程组选项，指定在构建特定进程组时需要传入的额外选项。目前仅支持 nccl 后端的 ProcessGroupNCCL.Options，可指定 is_high_priority_stream 以便 nccl 后端在有计算内核等待时选择高优先级 cuda 流。其他可用的 nccl 配置选项，请参见 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t device_id (torch.device | int，可选) – 此进程将使用的单个特定设备，允许进行后端特定的优化。目前仅在 NCCL 下有两个效果：通信器立即形成（立即调用 ncclCommInit* 而非正常的延迟调用），子组将在可能时使用 ncclCommSplit 以避免不必要的组创建开销。如果想尽早了解 NCCL 初始化错误，也可使用此字段。如果提供 int，API 假定将使用编译时的加速器类型。注意：要启用 backend == Backend.MPI，需要在支持 MPI 的系统上从源码构建 PyTorch。注意：多后端支持为实验性。目前未指定后端时，将同时创建 gloo 和 nccl 后端。gloo 后端用于 CPU 张量的集合操作，nccl 后端用于 CUDA 张量的集合操作。可通过传入格式为 "&lt;device_type>:&lt;backend_name>,&lt;device_type>:&lt;backend_name>" 的字符串指定自定义后端，例如 "cpu:gloo,cuda:custom_backend"。torch.distributed.device_mesh.init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None, backend_override=None)[source]# 根据 device_type、mesh_shape 和 mesh_dim_names 参数初始化 DeviceMesh。这将创建一个具有 n 维数组布局的 DeviceMesh，其中 n 为 mesh_shape 的长度。如果提供了 mesh_dim_names，每个维度将被标记为 mesh_dim_names[i]。注意：init_device_mesh 遵循 SPMD 编程模型，即相同的 PyTorch Python 程序在集群中所有进程/rank 上运行。确保 mesh_shape（描述设备布局的 nD 数组的维度）在所有 rank 上完全相同。不一致的 mesh_shape 可能导致挂起。注意：如果未找到进程组，init_device_mesh 将在后台初始化分布式通信所需的分布式进程组。参数：device_type (str) – mesh 的设备类型。目前支持："cpu"、"cuda/cuda-like"、"xpu"。不允许传入带 GPU 索引的设备类型，如 "cuda:0"。mesh_shape (Tuple[int]) – 定义描述设备布局的多维数组维度的元组。mesh_dim_names (Tuple[str]，可选) – 分配给描述设备布局的多维数组各维度的 mesh 维度名称元组。其长度必须与 mesh_shape 的长度匹配。mesh_dim_names 中的每个字符串必须唯一。backend_override (Dict[int | str, tuple[str, Options] | str | Options]，可选) – 对将为每个 mesh 维度创建的部分或全部 ProcessGroup 的覆盖。每个键可以是维度的索引或其名称（如果提供了 mesh_dim_names）。每个值可以是包含后端名称及其选项的元组，或仅其中一个组件（另一个将设为默认值）。返回：表示设备布局的 DeviceMesh 对象。返回类型：DeviceMesh 示例：>>> from torch.distributed.device_mesh import init_device_mesh >>> >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,)) >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp")) torch.distributed.is_initialized()[source]# 检查默认进程组是否已初始化。返回类型：bool torch.distributed.is_mpi_available()[source]# 检查 MPI 后端是否可用。返回类型：bool torch.distributed.is_nccl_available()[source]# 检查 NCCL 后端是否可用。返回类型：bool torch.distributed.is_gloo_available()[source]# 检查 Gloo 后端是否可用。返回类型：bool torch.distributed.distributed_c10d.is_xccl_available()[source]# 检查 XCCL 后端是否可用。返回类型：bool torch.distributed.is_torchelastic_launched()[source]# 检查此进程是否通过 torch.distributed.elastic（即 torchelastic）启动。使用 TORCHELASTIC_RUN_ID 环境变量的存在作为代理，判断当前进程是否通过 torchelastic 启动。这是合理的代理，因为 TORCHELASTIC_RUN_ID 映射到 rendezvous id，该 id 始终是非空值，用于对等发现的作业 id。返回类型：bool torch.distributed.get_default_backend_for_device(device)[source]# 返回给定设备的默认后端。参数：device (Union[str, torch.device]) – 要获取默认后端的设备。返回：给定设备的默认后端（小写字符串）。返回类型：str 目前支持三种初始化方法：TCP 初始化# 使用 TCP 初始化有两种方式，均需要所有进程可访问的网络地址和所需的 world_size。第一种方式需要指定属于 rank 0 进程的地址。此初始化方法要求所有进程手动指定 rank。注意，最新的分布式包不再支持多播地址，group_name 也已弃用。import torch.distributed as dist # 使用其中一台机器的地址 dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4) 共享文件系统初始化# 另一种初始化方法使用组中所有机器均可见的共享文件系统，以及所需的 world_size。URL 应以 file:// 开头，并包含共享文件系统上不存在的文件路径（在已存在的目录中）。文件系统初始化将在文件不存在时自动创建，但不会删除该文件。因此，用户有责任确保在下次以相同文件路径/名称调用 init_process_group() 之前清理该文件。注意，最新的分布式包不再支持自动 rank 分配，group_name 也已弃用。警告：此方法假定文件系统支持使用 fcntl 加锁 - 大多数本地系统和 NFS 支持此功能。警告：此方法将始终创建文件，并在程序结束时尽力清理和删除该文件。换言之，每次使用文件初始化方法都需要一个全新的空文件才能成功初始化。如果再次使用之前初始化留下的同一文件（恰好未被清理），这是意外行为，通常会导致死锁和失败。因此，即使此方法会尽力清理文件，如果自动删除失败，用户有责任确保在训练结束时删除该文件，以防止下次被重复使用。如果计划多次以相同文件名调用 init_process_group()，这一点尤为重要。换言之，如果文件未被删除/清理，再次以该文件调用 init_process_group() 将预期失败。经验法则：确保每次调用 init_process_group() 时文件不存在或为空。import torch.distributed as dist # rank 应始终指定 dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile', world_size=4, rank=args.rank) 环境变量初始化# 此方法从环境变量读取配置，允许完全自定义获取信息的方式。需要设置的变量：MASTER_PORT - 必填；必须是 rank 0 机器上的空闲端口 MASTER_ADDR - 必填（rank 0 除外）；rank 0 节点的地址 WORLD_SIZE - 必填；可在此处设置，也可在调用 init 函数时设置 RANK - 必填；可在此处设置，也可在调用 init 函数时设置 rank 0 的机器将用于建立所有连接。这是默认方法，意味着不必指定 init_method（或可设为 env://）。改善初始化时间# TORCH_GLOO_LAZY_INIT - 按需建立连接，而非使用全网格，可大幅改善非 all2all 操作的初始化时间。

```
torch.distributed.init_process_group()
```

**模式 4：** 示例：

```
>>> from torch.distributed.device_mesh import init_device_mesh
>>>
>>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
>>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))
```

**模式 5：** 组（Groups）# 默认情况下，集合操作在默认组（也称为 world）上运行，要求所有进程进入分布式函数调用。但某些工作负载可从更细粒度的通信中受益，这就是分布式组的用武之地。可使用 new_group() 函数创建新组，包含所有进程的任意子集。它返回一个不透明的组句柄，可作为 group 参数传给所有集合操作（集合操作是以某些众所周知的编程模式交换信息的分布式函数）。torch.distributed.new_group(ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False, group_desc=None, device_id=None)[source]# 创建新的分布式组。此函数要求主组中的所有进程（即分布式作业中的所有进程）都进入此函数，即使它们不会成为该组的成员。此外，组应在所有进程中以相同顺序创建。警告：安全并发使用：使用 NCCL 后端的多个进程组时，用户必须确保跨 rank 的集合操作全局执行顺序一致。如果进程内的多个线程发出集合操作，需要显式同步以确保一致的顺序。使用 torch.distributed 通信 API 的异步变体时，将返回一个 work 对象，通信内核被排入单独的 CUDA 流，允许通信与计算重叠。在一个进程组上发出一个或多个异步操作后，必须在使用另一个进程组之前通过调用 work.wait() 与其他 cuda 流同步。详情请参见《同时使用多个 NCCL 通信器》。参数：ranks (list[int]) – 组成员的 rank 列表。如果为 None，将设为所有 rank。默认为 None。timeout (timedelta，可选) – 详情和默认值请参见 init_process_group。backend (str 或 Backend，可选) – 要使用的后端。根据构建时配置，有效值为 gloo 和 nccl。默认使用与全局组相同的后端。此字段应以小写字符串形式给出（例如 "gloo"），也可通过 Backend 属性访问（例如 Backend.GLOO）。如果传入 None，将使用默认进程组对应的后端。默认为 None。pg_options (ProcessGroupOptions，可选) – 进程组选项，指定在构建特定进程组时需要传入的额外选项。即对于 nccl 后端，可指定 is_high_priority_stream 以便进程组选择高优先级 cuda 流。其他可用的 nccl 配置选项，请参见 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t use_local_synchronization (bool，可选)：在进程组创建结束时执行组本地 barrier。与全局 barrier 不同，非成员 rank 无需调用 API 也不会加入 barrier。group_desc (str，可选) – 描述进程组的字符串。device_id (torch.device，可选) – 将此进程"绑定"到的单个特定设备，如果提供此字段，new_group 调用将立即尝试为该设备初始化通信后端。返回：可传给集合调用的分布式组句柄，如果 rank 不在 ranks 中则返回 GroupMember.NON_GROUP_MEMBER。注意：use_local_synchronization 不适用于 MPI。注意：虽然 use_local_synchronization=True 在较大集群和小进程组中可能显著更快，但需注意它会改变集群行为，因为非成员 rank 不会加入组 barrier()。注意：use_local_synchronization=True 在每个 rank 创建多个重叠进程组时可能导致死锁。为避免这种情况，确保所有 rank 遵循相同的全局创建顺序。torch.distributed.get_group_rank(group, global_rank)[source]# 将全局 rank 转换为组 rank。global_rank 必须是 group 的成员，否则将引发 RuntimeError。参数：group (ProcessGroup) – 用于查找相对 rank 的 ProcessGroup。global_rank (int) – 要查询的全局 rank。返回：global_rank 相对于 group 的组 rank。返回类型：int 注意：在默认进程组上调用此函数返回恒等映射。torch.distributed.get_global_rank(group, group_rank)[source]# 将组 rank 转换为全局 rank。group_rank 必须是 group 的成员，否则将引发 RuntimeError。参数：group (ProcessGroup) – 用于查找全局 rank 的 ProcessGroup。group_rank (int) – 要查询的组 rank。返回：group_rank 相对于 group 的全局 rank。返回类型：int 注意：在默认进程组上调用此函数返回恒等映射。torch.distributed.get_process_group_ranks(group)[source]# 获取与 group 关联的所有 rank。参数：group (Optional[ProcessGroup]) – 要获取所有 rank 的 ProcessGroup。如果为 None，将使用默认进程组。返回：按组 rank 排序的全局 rank 列表。返回类型：list[int]

```
new_group()
```

**模式 6：** 警告：安全并发使用：使用 NCCL 后端的多个进程组时，用户必须确保跨 rank 的集合操作全局执行顺序一致。如果进程内的多个线程发出集合操作，需要显式同步以确保一致的顺序。使用 torch.distributed 通信 API 的异步变体时，将返回一个 work 对象，通信内核被排入单独的 CUDA 流，允许通信与计算重叠。在一个进程组上发出一个或多个异步操作后，必须在使用另一个进程组之前通过调用 work.wait() 与其他 cuda 流同步。详情请参见《同时使用多个 NCCL 通信器》。

```
NCCL
```

**模式 7：** 注意：如果将 DistributedDataParallel 与分布式 RPC 框架结合使用，应始终使用 torch.distributed.autograd.backward() 计算梯度，并使用 torch.distributed.optim.DistributedOptimizer 优化参数。示例：>>> import torch.distributed.autograd as dist_autograd >>> from torch.nn.parallel import DistributedDataParallel as DDP >>> import torch >>> from torch import optim >>> from torch.distributed.optim import DistributedOptimizer >>> import torch.distributed.rpc as rpc >>> from torch.distributed.rpc import RRef >>> >>> t1 = torch.rand((3, 3), requires_grad=True) >>> t2 = torch.rand((3, 3), requires_grad=True) >>> rref = rpc.remote("worker1", torch.add, args=(t1, t2)) >>> ddp_model = DDP(my_model) >>> >>> # 设置优化器 >>> optimizer_params = [rref] >>> for param in ddp_model.parameters(): >>> optimizer_params.append(RRef(param)) >>> >>> dist_optim = DistributedOptimizer( >>> optim.SGD, >>> optimizer_params, >>> lr=0.05, >>> ) >>> >>> with dist_autograd.context() as context_id: >>> pred = ddp_model(rref.to_here()) >>> loss = loss_func(pred, target) >>> dist_autograd.backward(context_id, [loss]) >>> dist_optim.step(context_id)

```
torch.distributed.autograd.backward()
```

**模式 8：** static_graph (bool) – 设为 True 时，DDP 知道训练图是静态的。静态图意味着：1）在整个训练循环中，已使用和未使用参数的集合不会改变；在这种情况下，用户是否设置 find_unused_parameters = True 无关紧要。2）图的训练方式在整个训练循环中不会改变（即不存在依赖迭代次数的控制流）。当 static_graph 设为 True 时，DDP 将支持以前无法支持的情况：1）可重入反向传播。2）多次激活检查点。3）模型有未使用参数时的激活检查点。4）存在前向函数之外的模型参数。5）当存在未使用参数时可能提升性能，因为 static_graph 设为 True 时 DDP 不会在每次迭代中搜索图以检测未使用参数。要检查是否可以将 static_graph 设为 True，一种方法是在之前的模型训练结束时检查 ddp 日志数据，如果 ddp_logging_data.get("can_set_static_graph") == True，大多数情况下也可以设置 static_graph = True。示例：>>> model_DDP = torch.nn.parallel.DistributedDataParallel(model) >>> # 训练循环 >>> ... >>> ddp_logging_data = model_DDP._get_ddp_logging_data() >>> static_graph = ddp_logging_data.get("can_set_static_graph")

```
True
```

## 参考文件

此 skill 在 `references/` 中包含完整文档：

- **other.md** - 其他文档

需要详细信息时，使用 `view` 读取特定参考文件。

## 使用此 Skill

### 初学者
从 getting_started 或 tutorials 参考文件开始，了解基础概念。

### 特定功能
使用相应类别的参考文件（api、guides 等）获取详细信息。

### 代码示例
上方快速参考部分包含从官方文档中提取的常用模式。

## 资源

### references/
从官方来源提取的有组织文档，包含：
- 详细说明
- 带语言注释的代码示例
- 原始文档链接
- 快速导航目录

### scripts/
在此添加常见自动化任务的辅助脚本。

### assets/
在此添加模板、样板代码或示例项目。

## 说明

- 此 skill 由官方文档自动生成
- 参考文件保留了源文档的结构和示例
- 代码示例包含语言检测以提供更好的语法高亮
- 快速参考模式从文档中的常见用法示例中提取

## 更新

要使用最新文档刷新此 skill：
1. 使用相同配置重新运行爬虫
2. skill 将使用最新信息重新构建