import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *

class ModelRunner:
    """
    模型运行器 - 负责管理模型的加载、执行和多GPU协调

    主要职责：
    1. 模型初始化和权重加载
    2. KV缓存的分配和管理（实现PagedAttention的关键）
    3. 多GPU通信和同步（通过NCCL和共享内存）
    4. CUDA图捕获和重放（加速decode阶段）
    5. 准备输入数据和执行前向传播

    工作模式：
    - Rank 0: 主进程，负责调度和采样
    - Rank 1+: 工作进程，通过共享内存接收指令并执行
    """
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        """
        初始化模型运行器

        参数:
            config: 模型配置字典，包含模型参数、内存设置等
            rank: 当前GPU的rank编号（0表示主进程）
            event: 用于多进程同步的Event对象
                  - rank 0: Event列表，用于通知其他rank
                  - rank 1+: 单个Event，用于等待rank 0的指令
        """
        self.config = config
        self.event = event

        # 设置分布式配置
        self.block_size = config['block_size']  # KV缓存块大小（如16个token）
        self.world_size = config['world_size']  # GPU数量
        self.enforce_eager = config.get('enforce_eager', False)  # 是否禁用CUDA图优化

        # 初始化分布式进程组（使用NCCL后端进行GPU间通信）
        self.rank = rank
        # init_process_group 是一个集体操作（collective operation），也是一个会和点，后面一个是dist.barrier()
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # 根据模型名称创建对应的模型实例
        # 支持的模型: Qwen3-0.6B, Llama-3.2-1B-Instruct
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # 将模型移动到对应的GPU上（在加载权重之前）
        self.model = self.model.cuda(rank)

        # 从checkpoint加载预训练权重
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # 注意：另一种方式是先在CPU加载权重再移动到GPU（已注释）
        # 当前方式：先移动模型到GPU，再加载权重（更节省内存）
        # self.model = self.model.cuda(rank)

        # 初始化采样器（用于从logits中采样下一个token）
        self.sampler = SamplerLayer()

        # 保存默认数据类型（在allocate_kv_cache中需要用到）
        self.default_dtype = torch.get_default_dtype()

        # Debug标志：用于标记第一次decode步骤
        self._first_decode = False

        # 预热模型以测量峰值内存使用量
        # 这一步很关键：通过运行一次最大batch来确定实际内存需求
        self.warmup_model()

        # 分配KV缓存（PagedAttention的核心）
        # 根据可用显存计算能分配多少个KV缓存块
        self.allocate_kv_cache()

        # 捕获CUDA图以加速decode阶段
        # CUDA图可以将整个计算图录制下来，后续直接重放，避免Python开销
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 设置默认设备和数据类型
        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        # 重要：在所有模型初始化完成后再设置共享内存和同步
        # 确保所有rank完成预热和内存分配后，rank 1才进入事件循环
        if self.world_size > 1:
            # 在设置共享内存前进行同步
            dist.barrier()
            if self.rank == 0:
                # 先尝试清理已存在的共享内存
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # 不存在也没关系

                # 创建共享内存（1MB大小，用于rank间通信）
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier确保rank 1等待共享内存创建完成
                dist.barrier()
            else:
                # Rank 1+等待rank 0创建共享内存
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')
                # 不在__init__中调用self.loop()，让外部代码处理
                # 否则会在初始化时陷入无限循环

    def read_shm(self):
        """
        从共享内存读取方法名和参数（仅rank 1+使用）

        工作流程：
        1. 等待event信号（由rank 0设置）
        2. 从共享内存读取数据长度（前4字节）
        3. 读取并反序列化方法名和参数
        4. 清除event信号

        返回:
            (method_name, args): 方法名和参数元组
        """
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()  # 阻塞等待rank 0的信号
        n = int.from_bytes(self.shm.buf[:4], 'little')  # 读取数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化
        self.event.clear()  # 清除信号，准备下次接收，等待Rank 0下次set()
        return method_name, args

    def write_shm(self, method_name: str, args: tuple):
        """
        将方法名和参数写入共享内存（仅rank 0使用）

        工作流程：
        1. 序列化方法名和参数
        2. 将数据长度写入前4字节
        3. 将序列化数据写入共享内存
        4. 设置所有event信号，通知其他rank

        参数:
            method_name: 要调用的方法名
            args: 方法参数元组
        """
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # 序列化：(method_name, args) 其中args是元组 -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')  # 写入长度
        self.shm.buf[4:n+4] = data  # 写入数据
        # 通知所有worker rank
        for event in self.event:
            event.set()

    def exit(self):
        """
        清理资源：关闭共享内存、销毁进程组、删除CUDA图

        在引擎关闭时调用，确保资源正确释放
        """
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()  # 只有rank 0负责删除共享内存
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # 检查进程组是否存在再销毁
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        """
        Worker进程的主循环（仅rank 1+使用）

        工作流程：
        1. 从共享内存读取方法名和参数
        2. 执行对应的方法
        3. 如果是exit命令，清理资源并退出循环

        这个循环使得worker rank可以持续响应主进程的指令
        """
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)  # 解包参数并调用方法
            if method_name == 'exit':
                self.exit()
                break

    def call(self, method_name: str, *args: dict):
        """
        统一的方法调用入口

        参数:
            method_name: 要调用的方法名（如'run', 'exit'等）
            args: 方法参数

        返回:
            方法执行结果

        工作原理：
        - Rank 0调用时：将指令写入共享内存，然后本地执行
        - Rank 1+调用时：直接本地执行（已从共享内存读取）
        """
        if self.world_size > 1 and self.rank == 0:  # 主进程需要通知worker
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    def warmup_model(self):
        """
        预热模型以测量峰值内存使用量

        目的：
        通过运行一次最大batch的推理来确定：
        1. 模型运行时的峰值内存使用
        2. 为后续KV缓存分配提供准确的内存基准

        工作流程：
        1. 清理缓存并重置内存统计
        2. 计算最大batch size（基于max_tokens和max_model_length）
        3. 创建虚拟序列并运行prefill
        4. 清理缓存，但保留内存统计信息
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 计算最大batch size
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length

        # 创建虚拟序列（填充0）进行预热
        seqs = [
            Sequence(
                token_ids=[0]*max_model_length, 
                block_size=self.config['block_size']
            ) for _ in range(batch_size)
        ]
        self.run(seqs, is_prefill=True) # 运行一次prefill
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配KV缓存内存 - PagedAttention的核心实现

        核心思想：
        不为每个序列单独分配KV缓存，而是创建一个统一的KV缓存池，
        分成固定大小的块（block），由BlockManager动态分配给序列。

        优势：
        1. 内存利用率高：避免碎片化，支持更大的batch size
        2. 灵活调度：块可以在序列间共享和重用
        3. 支持prefix caching：相同前缀的序列可以共享KV块

        工作流程：
        1. 计算可用GPU内存
        2. 计算每个KV块需要的字节数
        3. 确定能分配多少个块（跨rank同步取最小值）
        4. 分配统一的KV缓存张量并注入到模型的attention层
        """
        # 获取可用内存
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # 预留峰值内存差值，避免运行时OOM
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)

        # 获取KV缓存相关参数
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size  # 多GPU时KV heads被分片
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # 计算每个KV块的字节数
        # block_size: 每块包含的token数
        # 2: K和V两个缓存
        # num_layers: 每层都需要KV缓存
        # num_kv_heads * head_dim: 每个token的KV维度
        # dtype.itemsize: 每个元素的字节数（如float16=2字节）
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'

        # 跨rank同步max_cached_blocks，取最小值
        # 原因：不同rank的可用内存可能不同（rank 0通常内存开销更大）
        # 调度器运行在rank 0，需要确保分配的块数不超过任何rank的容量
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: 所有rank都获得最保守的限制
            # 即最内存受限的rank能服务的块数
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # 单GPU：直接使用本地值
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # 分配统一的KV缓存池
        # 形状: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # 2: K和V
        # num_blocks: 总块数
        # 重要：使用zeros()而不是empty()，避免垃圾值导致的数值问题
        allocated_kv_cache = torch.zeros(
            2,
            self.config['num_layers'],
            self.config['max_cached_blocks'],
            self.block_size,
            num_kv_heads,
            head_dim,
            device=f'cuda:{self.rank}'
        )

        # 将KV缓存注入到模型的每个attention层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]  # K缓存
                module.v_cache = allocated_kv_cache[1, layer_id]  # V缓存
                layer_id += 1

    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """
        为prefill阶段准备输入数据

        Prefill阶段：处理输入prompt的初始tokens，生成它们的KV缓存

        考虑prefix caching优化：
        - 如果序列有已缓存的prefix，只处理新的tokens
        - 已缓存的tokens可以直接从KV缓存读取，不需要重新计算

        返回的数据结构：
        - input_ids: 需要处理的token IDs（跳过已缓存的）
        - slot_mapping: 新KV值应该写入的位置（在KV缓存池中的索引）
        - cu_seqlens_q/k: cumulative sequence lengths，用于varlen attention
        - block_tables: 每个序列的块表（用于读取已缓存的KV）

        cu_seqlens的含义示例：
        cu_seqlens_q = [0, 3, 5, 9]
                        │  │  │  │
                        │  │  │  └─ seq3结束位置（第9个位置）
                        │  │  └──── seq2结束位置（第5个位置）
                        │  └─────── seq1结束位置（第3个位置）
                        └────────── 起始位置（第0个位置）

        参数:
            seqs: 要处理的序列列表

        返回:
            input_ids: 拼接后的token IDs张量
        """
        # 存储所有序列跳过prefix后的token IDs
        input_ids = []
        # 存储slot映射（KV缓存写入位置），这次计算的新KV应该写入哪些具体的slot
        slot_mappings = []
        # 每个序列的查询长度（新tokens数量），因为旧的Q对应的KV已经缓存了
        seqlens_q = []
        # 每个序列的总长度（包括已缓存的），因为新的Q需要读取之前的KV缓存
        seqlens_k = []
        # cumulative query lengths（用于varlen attention）
        cu_seqlens_q = [0]
        # cumulative key lengths
        cu_seqlens_k = [0]
        # 块表列表（需要padding到相同长度）
        block_tables = []

        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens

            # 只处理未缓存的tokens
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))

            # 累积序列长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])

            # 构建slot映射：确定每个新token的KV应该写入哪个位置
            if seq.block_table:
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    # 如果不是最后一个块，填满整个块
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        # 从block_id * block_size到(block_id+1) * block_size的范围
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        # 最后一个块可能没填满，从block_id * block_size到block_id * block_size + last_block_num_tokens的范围
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))

        # 如果存在prefix caching（查询长度 < 总长度），需要padding块表
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # 这个条件的真正含义：
            # "是否有序列需要从KV缓存中读取数据"
            #
            # 如果没有prefix cache：
            # - 所有KV都是新计算的
            # - 不需要读取缓存
            # - 不需要block_tables参数
            # - 因此不需要padding
            #
            # 如果有prefix cache：
            # - 需要读取缓存的KV
            # - 必须提供block_tables
            # - Flash Attention要求2D tensor对齐
            # - 因此需要padding
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                # padding到最大块数（用-1填充）
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)

        # 转换为CUDA张量（使用pin_memory + non_blocking加速数据传输）
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        # 设置全局上下文（供attention层使用）
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids


    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """
        为decode阶段准备输入数据

        Decode阶段：每次只生成一个新token，所有序列并行处理

        特点：
        - 每个序列只处理1个token（最后生成的token）
        - 需要读取之前所有tokens的KV缓存（通过block_tables）
        - 适合使用CUDA图优化（固定的计算模式）

        返回的数据结构：
        - input_ids: 每个序列的最后一个token
        - slot_mapping: 新KV值的写入位置（每个序列1个位置）
        - context_lens: 每个序列已处理的token数
        - block_tables: 每个序列的块表（padding到相同长度）

        参数:
            seqs: 要处理的序列列表

        返回:
            input_ids: 形状为[batch_size]的张量
        """
        input_ids = []
        context_lens = []   # 每个序列的上下文长度
        slot_mappings = []  # KV写入位置
        block_tables = []   # 块表

        for seq in seqs:
            # Decode时只处理最后一个token
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))  # 已处理的token总数
            # 计算slot位置：最后一个块的起始位置 + 块内偏移
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # Padding块表到相同长度（CUDA图需要固定形状）
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)

        # 转换为CUDA张量
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        # 设置全局上下文（供attention层使用）
        set_context(
            is_prefill=False,  # Decode阶段
            cu_seqlens_q=None,  # Decode不需要varlen attention
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids

    def prepare_sample(self, seqs: list[Sequence]) -> None:
        """
        准备采样参数（temperature）

        参数:
            seqs: 序列列表

        返回:
            temperature张量，形状为[batch_size]
        """
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """
        执行模型前向传播

        两种执行模式：
        1. Prefill模式 或 强制eager模式：
           - 直接运行模型，支持可变长度输入
           - 使用flash_attn_varlen_func处理batched varlen attention

        2. Decode模式（启用CUDA图）：
           - 使用预先捕获的CUDA图重放
           - 大幅减少Python开销，加速decode
           - 根据batch size选择合适的图

        CUDA图优化原理：
        - 第一次运行时录制整个计算图（kernel序列）
        - 后续执行只需拷贝输入数据到图变量，然后重放
        - 避免每次都从Python调用CUDA kernels的开销

        参数:
            input_ids: 输入token IDs
                      - Prefill: 1D张量（所有序列拼接）
                      - Decode: 1D张量（每个序列1个token）
            is_prefill: 是否为prefill阶段

        返回:
            logits: 输出logits，形状为[num_tokens, vocab_size]
        """
        if is_prefill or self.enforce_eager:
            # Prefill阶段：直接运行模型
            # 注意：input_ids保持1D（拼接的tokens），flash_attn_varlen_func需要1D输入配合cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            # Decode阶段：使用CUDA图加速
            bs = input_ids.size(0)
            context = get_context()

            # 找到能容纳当前batch size的最小图
            # 例如：bs=5会使用为bs=8捕获的图
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars

            # 将输入数据拷贝到图变量中
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)  # 先填充-1
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放CUDA图（执行录制的kernel序列）
            graph.replay()

            # 从图变量中读取输出
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        运行一次推理步骤（prefill或decode）

        这是ModelRunner的主要入口函数，完整流程：
        1. 准备输入数据（根据prefill/decode选择对应的prepare函数）
        2. 运行模型前向传播
        3. 采样下一个token（仅rank 0执行）
        4. 重置上下文

        多GPU协调：
        - 所有rank执行模型前向传播（通过NCCL同步）
        - 只有rank 0执行采样（避免重复计算）

        参数:
            seqs: 要处理的序列列表
            is_prefill: True为prefill阶段，False为decode阶段

        返回:
            token_ids: 采样得到的token IDs（仅rank 0返回，其他rank返回None）
        """
        # 步骤1: 准备输入数据
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)

        # 步骤2: 运行模型
        logits = self.run_model(input_ids, is_prefill)

        # 步骤3: 采样（仅rank 0）
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))

        # 步骤4: 重置全局上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        """
        捕获CUDA图以加速decode阶段

        CUDA图原理：
        将整个模型的前向传播录制为一个计算图（CUDA kernel序列）
        后续执行时只需更新输入数据，然后重放图，避免Python调用开销

        优化效果：
        - 减少CPU-GPU同步开销
        - 消除Python解释器开销
        - 对于decode这种固定pattern的操作，加速可达2-3倍

        捕获策略：
        1. 预分配最大尺寸的张量（一次分配，所有图共享）
        2. 为不同batch size捕获多个图：[1, 2, 4, 8, 16, 32, ...]
        3. 运行时根据实际batch size选择最小的合适图

        为什么需要多个batch size：
        - CUDA图要求输入形状固定
        - 不同batch size的计算图不同
        - 使用稍大的图可以容纳小batch（浪费少量内存但避免重新捕获）

        Graph pool的作用：
        - 第一个图创建pool，后续图共享这个pool
        - Pool复用内存，减少显存占用
        """
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)

        # 预分配最大尺寸的张量（所有图共享这些buffer）
        # Decode时input形状为(batch_size,) - 每个序列1个token
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # PagedAttention相关张量
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')  # KV写入位置
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')  # 上下文长度
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')  # 块表
        # 输出logits
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # 为不同batch size捕获图
        # [1, 2, 4, 8] + [16, 32, 48, ...] 覆盖常见的batch size
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        # 从大到小捕获（先捕获大的，pool复用更高效）
        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()

            # 设置上下文（为这个batch size）
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )

            # 预热：运行一次让PyTorch分配所有需要的内存
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            # 捕获CUDA图
            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    # 第一个图创建pool，后续图共享
                    graph_pool = graph.pool()

            # 保存捕获的图
            self.graphs[batch_size] = graph

            # 确保捕获完成后再重置上下文，准备下一次捕获
            torch.cuda.synchronize()
            reset_context()

        # 保存图变量（重放时会更新这些变量的值）
        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )