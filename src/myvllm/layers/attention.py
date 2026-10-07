# 高性能注意力机制实现
# 使用 Triton 编译器编写自定义 GPU 内核，实现了三个关键功能：
# 1. 分页 KV Cache 存储 - 将 key/value 存储到分页内存中以支持动态批处理
# 2. Flash Attention - 用于预填充（prefill）阶段的高效注意力计算
# 3. Paged Attention - 用于解码（decode）阶段的高效注意力计算

import triton
import triton.language as tl
from myvllm.utils import get_context, set_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr,          # 指向要存储的 key 张量的指针
    value_ptr,        # 指向要存储的 value 张量的指针
    k_cache_ptr,      # 指向 key cache 的指针（目标存储位置）
    v_cache_ptr,      # 指向 value cache 的指针（目标存储位置）
    slot_mapping_ptr, # 指向槽位映射的指针，将每个 token 映射到 cache 中的位置
    num_kv_heads: tl.constexpr,  # KV 头的数量（编译时常量）
    head_dim: tl.constexpr,       # 每个注意力头的维度（编译时常量）
    block_size: tl.constexpr      # 每个内存块可以存储的 token 数量（编译时常量）
):
    """
    将 key 和 value 存储到分页 KV cache 中的 Triton 内核

    分页机制：为了支持动态批处理和长序列，cache 被组织成固定大小的块（blocks）
    每个 token 通过 slot_mapping 映射到 cache 中的一个槽位（slot）

    网格布局：(num_tokens, num_kv_heads)
        - 每个 GPU 线程处理一个 (token, head) 对
        - program_id(0) = token 索引
        - program_id(1) = head 索引

    Cache 布局：(num_blocks, block_size, num_kv_heads, head_dim)
        - num_blocks: cache 中的块数量
        - block_size: 每块存储的 token 数量
        - num_kv_heads: KV 头的数量（支持 GQA/MQA）
        - head_dim: 每个头的特征维度
    """
    # 获取当前线程处理的 token 索引（第 0 维的 program ID）
    token_idx = tl.program_id(0)  # 每个 GPU 线程处理一个 token

    # 从 slot_mapping 中加载这个 token 应该存储到 cache 的哪个槽位
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    # 如果 slot_idx 为 -1，表示这个 token 不需要存储（例如填充 token）
    if slot_idx == -1:
        return

    # 计算这个槽位在哪个块中，以及在块内的偏移位置
    # 例如：slot_idx=37, block_size=16 -> block_idx=2, block_offset=5
    block_idx = slot_idx // block_size      # 块索引（整除）
    block_offset = slot_idx % block_size    # 块内偏移（取余）

    # 获取当前线程处理的头索引（第 1 维的 program ID）
    # program_id(0) = 处理哪个 token
    # program_id(1) = 处理哪个 head
    head_idx = tl.program_id(1)

    # 创建一个向量 [0, 1, 2, ..., head_dim-1]，用于访问头维度的所有元素
    head_offsets = tl.arange(0, head_dim)

    # 计算输入张量中的偏移量
    # 输入形状：(num_tokens, num_kv_heads, head_dim)
    # 例如：token_idx=5, num_kv_heads=8, head_dim=128, head_idx=3
    #       input_offset = 5 * (8 * 128) + 3 * 128 + [0, 1, 2, ..., 127]
    #                    = 5120 + 384 + [0, 1, 2, ..., 127]
    #                    = [5504, 5505, 5506, ..., 5631]
    input_offset = (token_idx * num_kv_heads * head_dim +  # 跳过之前的 tokens
                    head_idx * head_dim +                   # 跳过之前的 heads
                    head_offsets)                           # 当前 head 内的位置

    # 计算 cache 中的偏移量
    # Cache 形状：(num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim +  # 跳过之前的块
                   block_offset * num_kv_heads * head_dim +              # 跳过块内之前的位置
                   head_idx * head_dim +                                 # 跳过之前的 heads
                   head_offsets)                                         # 当前 head 内的位置

    # 从输入张量中加载 key 和 value（从全局内存读取到寄存器）
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)

    # 将 key 和 value 存储到 cache 中（从寄存器写入到全局内存）
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor,        # 输入的 key 张量
    value: torch.Tensor,      # 输入的 value 张量
    k_cache: torch.Tensor,    # key cache 张量
    v_cache: torch.Tensor,    # value cache 张量
    slot_mapping: torch.Tensor,  # 槽位映射张量
    block_size: int           # 块大小
):
    """
    将 key-value 对存储到分页 cache 中的 Python 封装函数

    这个函数是 Triton 内核的高层封装，负责：
    1. 验证输入张量的形状和连续性
    2. 设置内核启动的网格维度
    3. 调用 Triton 内核执行实际的存储操作

    参数说明：
        key: (num_tokens, num_kv_heads, head_dim) - 要存储的 key 张量
        value: (num_tokens, num_kv_heads, head_dim) - 要存储的 value 张量
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim) - key 的 cache
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim) - value 的 cache
        slot_mapping: (num_tokens,) - 将每个 token 映射到 cache 槽位的索引数组
        block_size: 每个块能存储的 token 数量
    """
    num_tokens, num_kv_heads, head_dim = key.shape

    # 确保张量在内存中是连续的，这对于 Triton 内核的高效访问很重要
    # 非连续张量会导致内存访问模式不规则，降低性能
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    # 验证输入的合法性
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"

    # 设置网格维度：启动 num_tokens × num_kv_heads 个 GPU 线程
    # 每个线程处理一个 (token, head) 对
    grid = (num_tokens, num_kv_heads)

    # 调用 Triton 内核
    # Triton 会自动将 PyTorch 张量转换为指针传递给内核
    store_kvcache_kernel[grid](
        key,           # 张量会自动转换为指针
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,  # 编译时常量
        head_dim=head_dim,           # 编译时常量
        block_size=block_size        # 编译时常量
    )


# Triton实现的 Flash Attention 内核，支持变长序列和 GQA（Grouped Query Attention）
@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,              # Query, Key, Value 和 Output 张量的指针
    cu_seqlens_q_ptr,        # 累积序列长度的指针（用于变长序列）例如：cu_seqlens_q[i] 表示前 i 个序列的总 token 数
    scale,                   # 注意力缩放因子
    num_heads: tl.constexpr,       # Query 头的数量
    num_kv_heads: tl.constexpr,    # Key/Value 头的数量（支持 GQA）
    head_dim: tl.constexpr,        # 头维度
    BLOCK_M: tl.constexpr,         # Query 块大小（M 维度）
    BLOCK_N: tl.constexpr,         # Key/Value 块大小（N 维度）
):
    """
    Flash Attention 内核，用于处理变长序列的高效注意力计算

    Flash Attention 的核心思想：
    1. 分块计算：将 Q、K、V 分成小块，逐块计算注意力
    2. 在线 softmax：使用在线算法增量更新 softmax 的最大值和归一化因子
    3. 融合计算：将 softmax 和矩阵乘法融合在一个内核中，减少内存访问

    变长序列支持：
    - 通过 cu_seqlens（cumulative sequence lengths）支持批次中不同长度的序列
    - 每个序列独立处理，避免填充浪费

    网格布局：(num_blocks_per_seq, num_heads, num_seqs)
    - program_id(0): 当前处理的 query 块索引
    - program_id(1): 当前处理的头索引
    - program_id(2): 当前处理的序列索引

    支持 Grouped Query Attention (GQA)：
    - 多个 query 头可以共享同一个 key/value 头
    - 例如：32 个 query 头可能只有 8 个 kv 头（每 4 个 query 头共享 1 个 kv 头）
    """
    # 获取当前线程的程序 ID
    start_m = tl.program_id(0)   # query 块索引
    off_h = tl.program_id(1)     # 头索引
    seq_idx = tl.program_id(2)   # 序列索引

    # 计算当前 query 头应该使用哪个 KV 头（用于 GQA）
    # 例如：如果 num_heads=32, num_kv_heads=8，那么每 4 个 query 头共享 1 个 kv 头
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # 从累积序列长度数组中加载当前序列的起始和结束位置
    # cu_seqlens[i] 表示前 i 个序列的总 token 数
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    # 提前退出：如果当前块超出了序列长度，无需计算
    if start_m * BLOCK_M >= seq_len:
        return

    # 计算当前块处理的 query token 偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [start_m*BLOCK_M, ..., (start_m+1)*BLOCK_M-1]
    offs_d = tl.arange(0, head_dim)                     # [0, 1, ..., head_dim-1]

    # 计算 Query 张量的指针
    # Q 形状：(total_tokens, num_heads, head_dim)
    # 使用广播：offs_m[:, None] 形状 (BLOCK_M, 1), offs_d[None, :] 形状 (1, head_dim)
    # 结果形状：(BLOCK_M, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]

    # 加载 Q 块，使用掩码处理边界情况（避免访问越界）
    mask_m = offs_m < seq_len  # 标记哪些位置是有效的
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)  # 无效位置填充 0

    # 初始化输出累加器（用于在线 softmax 算法）
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)              # 归一化因子（softmax 分母）
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10       # 最大值（初始化为负无穷）
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)    # 输出累加器

    # 计算需要处理的 K/V 块数量（向上取整）
    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    # 循环处理所有 K、V 块（Flash Attention 的核心循环）
    for block_n in range(num_blocks):
        # 当前块的起始位置
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [start_n, start_n+1, ..., start_n+BLOCK_N-1]

        # 标记哪些位置在序列范围内
        mask_n = offs_n < seq_len

        # 计算 K 张量的指针
        # K 形状：(total_tokens, num_kv_heads, head_dim)
        # 注意：这里使用 kv_head_idx 而不是 off_h（支持 GQA）
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]

        # 加载 K 块 - 形状 (head_dim, BLOCK_N)
        # 注意：K 是转置的，方便后续的矩阵乘法 Q @ K^T
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # 计算注意力分数：QK^T
        # q 形状：(BLOCK_M, head_dim)
        # k 形状：(head_dim, BLOCK_N)
        # qk 形状：(BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale  # 应用缩放因子（通常是 1/sqrt(head_dim)）

        # 应用因果掩码（causal mask）：只能注意到当前位置及之前的位置
        # 这对于自回归生成至关重要，防止模型"看到未来"
        # mask_causal[i, j] = True 表示 query i 可以注意到 key j
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)  # 无效位置设为负无穷

        # === 在线 softmax 更新算法（Flash Attention 的关键） ===
        # 传统 softmax 需要两次遍历：1) 找最大值 2) 计算 exp 和归一化
        # 在线算法只需一次遍历，增量更新最大值和归一化因子

        # 1. 计算当前块的最大分数，沿着第 1 维（列方向）操作，得到每一行的最大值
        m_ij = tl.max(qk, axis=1)  # 形状：(BLOCK_M,)

        # 2. 更新全局最大值
        m_i_new = tl.maximum(m_i, m_ij)

        # 3. 计算重缩放因子（用于更新之前的累加值）
        alpha = tl.exp(m_i - m_i_new)

        # 4. 计算当前块的 softmax 概率（相对于新的最大值）
        p = tl.exp(qk - m_i_new[:, None])

        # 5. 重缩放之前的累加器（因为最大值可能变了）
        acc = acc * alpha[:, None]

        # 6. 加载 V 块
        # V 形状：(total_tokens, num_kv_heads, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # 形状：(BLOCK_N, head_dim)

        # 7. 累加加权的 value：output += softmax(QK^T) @ V
        # p 形状：(BLOCK_M, BLOCK_N)
        # v 形状：(BLOCK_N, head_dim)
        # 结果形状：(BLOCK_M, head_dim)
        acc = acc + tl.dot(p.to(v.dtype), v)

        # 8. 更新归一化因子（softmax 分母）
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # 9. 更新最大值
        m_i = m_i_new

    # 最终归一化：除以累积的 softmax 分母
    acc = acc / l_i[:, None]

    # 存储输出
    # O 形状：(total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])



def flash_attention_prefill(
    q: torch.Tensor,           # Query 张量
    k: torch.Tensor,           # Key 张量
    v: torch.Tensor,           # Value 张量
    cu_seqlens: torch.Tensor,  # 累积序列长度
    scale: float,              # 注意力缩放因子
    num_heads: int,            # Query 头数量
    num_kv_heads: int,         # Key/Value 头数量
    head_dim: int,             # 头维度
) -> torch.Tensor:
    """
    预填充阶段的优化 Flash Attention 实现（支持变长序列）

    预填充（Prefill）vs 解码（Decode）：
    - 预填充：处理输入提示（prompt）的所有 token，生成初始的 KV cache
    - 解码：逐个生成新 token，每次只处理一个 token

    Flash Attention 的优势：
    1. 内存高效：O(N) 内存复杂度（vs 标准注意力的 O(N²)）
    2. 计算高效：通过分块和融合操作减少 HBM 访问
    3. 支持长序列：可以处理标准注意力无法放入内存的长序列

    参数说明：
        q: (total_tokens, num_heads, head_dim) - Query 张量
        k: (total_tokens, num_kv_heads, head_dim) - Key 张量
        v: (total_tokens, num_kv_heads, head_dim) - Value 张量
        cu_seqlens: (num_seqs + 1,) - 累积序列长度，cu_seqlens[i] 表示前 i 个序列的总 token 数
        scale: 注意力缩放因子，通常是 1.0（会在内核中除以 sqrt(head_dim)）

    返回：
        output: (total_tokens, num_heads, head_dim) - 注意力输出
    """
    # 确保张量在内存中连续，提高访问效率
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # 分配输出张量
    output = torch.empty_like(q)

    # === 选择块大小（关键的性能调优参数） ===
    # 块大小的选择需要平衡：
    # 1. 共享内存使用：块越大，需要的共享内存越多
    # 2. 计算效率：块越大，计算吞吐量越高
    # 3. 占用率：块大小影响 GPU 的占用率

    # 共享内存使用估算：
    # - 注意力分数矩阵：BLOCK_M × BLOCK_N × 4 字节（float32）
    # - Q 块：BLOCK_M × head_dim × 4 字节
    # - K、V 块：BLOCK_N × head_dim × 4 字节
    # 大多数 GPU 的共享内存限制在 48KB-64KB

    if head_dim <= 64:
        BLOCK_M = 64   # Query 块大小
        BLOCK_N = 64   # Key/Value 块大小
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    # 序列数量
    num_seqs = cu_seqlens.shape[0] - 1

    # 找到最长序列的长度，用于确定网格大小
    # 注意：这里需要将 cu_seqlens 移到 CPU 上进行计算
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    # 计算网格维度 - 一次启动所有需要的内核
    # grid = (每个序列的 query 块数, 头数量, 序列数量)
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

    # 调用 Triton 内核
    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,           # 输出张量指针
    query_ptr,            # Query 张量指针
    k_cache_ptr,          # Key cache 指针
    v_cache_ptr,          # Value cache 指针
    block_tables_ptr,     # 块表指针（映射逻辑块到物理块）
    context_lens_ptr,     # 上下文长度指针
    scale: tl.constexpr,          # 注意力缩放因子
    num_heads: tl.constexpr,      # Query 头数量
    num_kv_heads: tl.constexpr,   # KV 头数量
    head_dim: tl.constexpr,       # 头维度
    block_size: tl.constexpr,     # 块大小
    max_num_blocks: tl.constexpr, # 每个序列的最大块数
    BLOCK_N: tl.constexpr,        # 处理的块大小（chunk size）
):
    """
    解码阶段的分页注意力内核（优化版本）

    解码阶段的特点：
    - 每次只处理一个新 token 的 query
    - 需要注意到之前所有 token 的 key 和 value（存储在 KV cache 中）
    - KV cache 使用分页机制管理，提高内存利用率

    分页机制（Paged Attention）：
    - 将 KV cache 组织成固定大小的块（类似操作系统的分页）
    - 使用块表（block_tables）将逻辑块索引映射到物理块索引
    - 优势：避免内存碎片，支持高效的内存共享和动态批处理

    块表工作原理：
    - block_tables[batch_idx, logical_block_idx] = physical_block_idx
    - 每个 token 的位置：logical_block = token_idx // block_size
    - 块内偏移：block_offset = token_idx % block_size

    处理策略：
    - 将所有历史 token 分成 chunks（每个 chunk 包含 BLOCK_N 个 token）
    - 每个 chunk 可以跨越多个物理块（块边界不与 chunk 边界对齐）
    - 每个 lane（线程）独立解析自己的 token 位置

    网格布局：(batch_size, num_heads)
    - 每个 GPU 线程处理一个 (batch, head) 对
    """
    # 获取当前线程处理的批次和头索引
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # 计算这个 query 头对应的 KV 头（支持 GQA）（当前每两个q头对应一个KV头）
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # 加载这个序列的上下文长度（即有多少个历史 token）
    context_len = tl.load(context_lens_ptr + batch_idx)

    # 加载 query
    # Query 形状：(batch_size, num_heads, head_dim)
    offs_d = tl.arange(0, head_dim)  # [0, 1, ..., head_dim-1]
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    # q形状为 (head_dim,)
    q = tl.load(query_ptr + q_offset)

    # 初始化累加器（用于在线 softmax）
    acc = tl.zeros([head_dim], dtype=tl.float32)  # 输出累加器
    l_i = 0.0                                       # 归一化因子
    m_i = -1e10                                     # 最大值（初始化为负无穷）

    # 计算需要处理的总 chunk 数
    # 注意：这里使用 max_num_blocks * block_size 作为上界，而不是 context_len
    # 因为我们需要为所有可能的 token 位置分配 chunk
    # 比如有两个序列，一个长52，一个长20，前一个存储占用了4个块，后一个存储占用了2个块，
    # 那么 max_num_blocks=4，block_size=16，足够覆盖所有历史 token
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    # 遍历所有 chunks，处理所有历史 token
    for chunk_idx in range(max_chunks):
        # 当前 chunk 的全局 token 起始索引
        token_start = chunk_idx * BLOCK_N

        # 只处理在有效范围内的 chunk，起始位置超过序列长度的 chunk 可以跳过
        if token_start < context_len:
            # 确定当前 chunk 中哪些 token 是有效的，广播成 [BLOCK_N] 的布尔掩码
            offs_n = token_start + tl.arange(0, BLOCK_N)  # [token_start, ..., token_start+BLOCK_N-1]

            # 计算每个 token 对应的逻辑块和块内偏移
            logical_block = offs_n // block_size
            # physical_block * block_size 已经将基础指针移到了块的起始位置
            # 所以这里加的是块内偏移，而不是全局 token 索引
            offs_in_block = offs_n % block_size

            # 前者标记哪些 token 在有效范围内，后者标记哪些chunk在块表中有对应的物理块
            in_range = (offs_n < context_len) & (logical_block < max_num_blocks)

            # 从块表中加载物理块索引，向量化计算：一次性得到所有 token 的物理块索引
            # 一个 chunk 可以跨越多个块，这些块在 cache 中不需要相邻
            # 虽然变量名叫 physical_block，但它的值仍然是一个索引号（整数 ID），而不是真正的物理内存地址
            physical_block = tl.load(
                block_tables_ptr + batch_idx * max_num_blocks + logical_block,
                mask=in_range, other=-1)

            # 标记有效的 token（在范围内且物理块存在）
            valid = in_range & (physical_block != -1)

            # 被掩码掉的 lane 仍然参与地址计算，所以给它们块 0
            # 以保持每个计算的偏移都在 cache 范围内
            physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

            # 计算 KV cache 中的偏移量
            # Cache 形状：(num_blocks, block_size, num_kv_heads, head_dim)
            # kv_offset形状: (head_dim, BLOCK_N)
            kv_offset = (physical_block[None, :] * (block_size * num_kv_heads * head_dim)
                         + offs_in_block[None, :] * (num_kv_heads * head_dim)
                         + kv_head_idx * head_dim
                         + offs_d[:, None])

            # 加载 K 并计算注意力分数。一次性加载整个 chunk 的 K：形状 (head_dim, BLOCK_N)
            k = tl.load(k_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            k = tl.cast(k, tl.float32)
            # 计算 Q·K（点积）
            # q 形状：(head_dim,)
            # k 形状：(head_dim, BLOCK_N)
            # score 形状：(BLOCK_N,)
            score = tl.sum(q[:, None] * k, axis=0) * scale
            qk = tl.where(valid, score, -1e10)  # 无效位置设为负无穷

            # === 在线 softmax 更新 ===
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            # 重缩放累加器
            acc = acc * alpha
            l_i = l_i * alpha

            # 加载 V 并累加加权的 value
            v = tl.load(v_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            v = tl.cast(v, tl.float32)
            weight = tl.where(valid, p, 0.0)
            acc = acc + tl.sum(weight[None, :] * v, axis=1)
            l_i = l_i + tl.sum(weight)

            m_i = m_i_new

    # 最终归一化
    output = acc / l_i

    # 存储输出
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)



def paged_attention_decode(
    query: torch.Tensor,        # 当前 token 的 query
    k_cache: torch.Tensor,      # Key cache
    v_cache: torch.Tensor,      # Value cache
    block_tables: torch.Tensor, # 块表（逻辑块到物理块的映射）
    context_lens: torch.Tensor, # 每个序列的上下文长度
    scale: float,               # 注意力缩放因子
    num_heads: int,             # Query 头数量
    num_kv_heads: int,          # KV 头数量
    head_dim: int,              # 头维度
    block_size: int             # 块大小，序列数量
) -> torch.Tensor:
    """
    使用分页 KV cache 的解码模式注意力计算

    解码阶段特点：
    - 批次中的每个序列只生成一个新 token
    - 需要注意到该序列之前所有的 token（存储在 KV cache 中）
    - 使用分页机制高效管理和访问 KV cache

    参数说明：
        query: (batch_size, num_heads, head_dim) - 当前步的 query（每个序列一个 token）
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim) - 分页的 key cache
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim) - 分页的 value cache
        block_tables: (batch_size, max_num_blocks) - 块表，映射每个序列的逻辑块到物理块
        context_lens: (batch_size,) - 每个序列已有的 token 数量
        scale: 注意力缩放因子

    返回：
        output: (batch_size, num_heads, head_dim) - 注意力输出
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1] # 最大block数量

    # 确保 query 在内存中连续
    query = query.contiguous()

    # 分配输出张量
    output = torch.empty_like(query)

    # 设置处理 KV token 的 chunk 大小
    # 较小的 head_dim 可以使用更大的 chunk
    BLOCK_N = 64 if head_dim <= 128 else 32

    # 网格维度：(batch_size, num_heads)
    # 每个线程处理一个 (batch, head) 对
    grid = (batch_size, num_heads)

    # 调用 Triton 内核
    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )

    return output


class Attention(nn.Module):
    """
    统一的注意力模块，支持预填充和解码两种模式

    这个模块整合了：
    1. Flash Attention - 用于预填充阶段的高效注意力计算
    2. Paged Attention - 用于解码阶段使用 KV cache 的高效计算
    3. KV Cache 管理 - 分页式的 key-value cache 存储

    架构特点：
    - 支持 Multi-Head Attention (MHA)
    - 支持 Grouped Query Attention (GQA) - 多个 query 头共享 KV 头
    - 支持 Multi-Query Attention (MQA) - 所有 query 头共享一个 KV 头
    - 使用分页机制管理 KV cache，提高内存利用率

    工作模式：
    - 预填充（Prefill）：处理输入 prompt，生成初始 KV cache
    - 解码（Decode）：逐个生成新 token，使用 KV cache 加速
    """

    def __init__(
        self,
        num_heads: int,           # Query 头的数量
        head_dim: int,            # 每个头的维度
        scale: float = 1.0,       # 基础缩放因子（会额外除以 sqrt(head_dim)）
        num_kv_heads: int = None, # KV 头的数量（None 表示与 num_heads 相同，即 MHA）
        block_size: int = 16,     # 每个 cache 块的大小（token 数量）
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        # 如果未指定 num_kv_heads，默认等于 num_heads（标准多头注意力）
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size

        # 初始化空的 KV cache（稍后会被分配实际的内存）
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        前向传播：根据上下文自动选择预填充或解码模式 prefill/decode

        参数：
            q: Query 张量
               - 预填充模式：(total_tokens, num_heads, head_dim)  
               - 解码模式：(batch_size, num_heads, head_dim)
            k: Key 张量（当前步的 key）
               - 预填充模式：(total_tokens, num_kv_heads, head_dim)
               - 解码模式：(batch_size, num_kv_heads, head_dim)
            v: Value 张量（当前步的 value）
               - 预填充模式：(total_tokens, num_kv_heads, head_dim)
               - 解码模式：(batch_size, num_kv_heads, head_dim)

        返回：
            output: 注意力输出
               - 预填充模式：(total_tokens, num_heads * head_dim)
               - 解码模式：(batch_size, num_heads * head_dim)
        """
        # 从全局上下文获取当前批次的元数据
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # === 第一步：将当前的 k, v 存入 cache（如果 cache 已分配） ===
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # 确保 k, v 的形状正确：(num_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:
                # 如果是 4 维：(B, N, num_kv_heads, head_dim)
                # 需要 reshape 成 (B*N, num_kv_heads, head_dim) 3维向量
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # 已经是正确的形状 (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()

            # 调用存储函数，将 k, v 存入分页 cache，把所有 token 看成一个扁平的序列，不关心批次结构。
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        # 计算注意力缩放因子：base_scale / sqrt(head_dim)
        # 这是 scaled dot-product attention 的标准缩放
        scale = self.scale / (self.head_dim ** 0.5)

        # === 第二步：根据模式选择不同的注意力计算方法 ===
        if context.is_prefill:
            # ========== 预填充模式 ==========
            # 使用 Flash Attention 处理完整序列
            # 支持变长序列（通过 cu_seqlens）cu_seqlens = Cumulative Sequence Lengths（累积序列长度）

            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")

            # 调用 Flash Attention
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale,
                                        self.num_heads, self.num_kv_heads, self.head_dim)

            # 输出形状：(total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            # ========== 解码模式 ==========
            # 使用 Paged Attention 从 cache 中读取历史 KV
            # 只处理当前步的 query

            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,    # 块表：逻辑块到物理块的映射
                context.context_lens,    # 每个序列的上下文长度
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )

            # 输出形状：(batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    """
    性能测试示例代码

    这个测试脚本用于：
    1. 验证注意力模块的正确性
    2. 测量推理延迟
    3. 预热 GPU（首次运行会编译 Triton 内核）
    """
    # 创建注意力层，移到 GPU
    layer = Attention(num_heads=8, head_dim=64).cuda()

    # 测试参数
    B, N = 4, 1024  # 批次大小=4, 序列长度=1024
    num_heads = layer.num_heads  # 8
    num_kv_heads = layer.num_kv_heads  # 8
    head_dim = layer.head_dim  # 64
    total_tokens = B * N  # 4096

    # 在 prefill 模式下，输入形状为 (total_tokens, num_heads, head_dim)
    q = torch.randn(total_tokens, num_heads, head_dim).cuda()
    k = torch.randn(total_tokens, num_kv_heads, head_dim).cuda()
    v = torch.randn(total_tokens, num_kv_heads, head_dim).cuda()

    # 初始化 KV cache（实际使用中会由推理引擎分配）
    # cache 的形状取决于分页策略，这里简单分配足够的空间
    max_num_blocks = 128
    block_size = layer.block_size
    layer.k_cache = torch.zeros(max_num_blocks, block_size, num_kv_heads, head_dim).cuda()
    layer.v_cache = torch.zeros(max_num_blocks, block_size, num_kv_heads, head_dim).cuda()

    # 设置全局 context，使用 prefill 模式
    # 在 prefill 模式下，处理完整序列，不需要 block_tables
    set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, N, 2*N, 3*N, 4*N], dtype=torch.int32).cuda(),  # 累积序列长度
        max_seqlen_q=N,
        max_seqlen_k=N
    )

    # 预热迭代：首次运行会触发 Triton 内核编译，需要较长时间
    print("Warming up...")
    for _ in range(10):
        _ = layer(q, k, v)

    # 性能测试：测量 100 次推理的平均延迟
    import time
    times = []
    print("Running performance test...")
    for _ in range(100):
        torch.cuda.synchronize()  # 等待之前的操作完成
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()  # 等待当前操作完成
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
    print(f"Output shape: {output_tensor.shape}")
