# KV Cache 流程详解

## 目录
- [1. 核心概念](#1-核心概念)
- [2. PagedAttention 原理](#2-pagedattention-原理)
- [3. 数据结构](#3-数据结构)
- [4. 完整流程](#4-完整流程)
- [5. Prefix Caching](#5-prefix-caching)
- [6. 实际案例](#6-实际案例)

---

## 1. 核心概念

### 1.1 什么是 KV Cache？

在 Transformer 的自注意力机制中，每个 token 需要：
- **Query (Q)**: 当前 token 的查询向量
- **Key (K)**: 所有历史 token 的键向量
- **Value (V)**: 所有历史 token 的值向量

为了避免重复计算，我们将已计算的 K 和 V 缓存起来，这就是 **KV Cache**。

```python
# 不使用 KV Cache（每次都重新计算所有token）
for step in range(100):
    # 重新计算所有历史token的K和V
    K = compute_keys(tokens[0:step+1])    # 计算量随step增长
    V = compute_values(tokens[0:step+1])  # 计算量随step增长
    output = attention(Q, K, V)

# 使用 KV Cache（只计算新token）
kv_cache = []
for step in range(100):
    # 只计算新token的K和V
    K_new = compute_key(tokens[step])     # 固定计算量
    V_new = compute_value(tokens[step])   # 固定计算量
    kv_cache.append((K_new, V_new))
    output = attention(Q, kv_cache)
```

### 1.2 传统方法的问题

**方法1：每个序列独立分配**
```python
# 为每个序列分配固定大小的KV缓存
for seq in sequences:
    seq.kv_cache = torch.zeros(max_length, hidden_size)
```

**问题：**
- 内存碎片化严重
- 必须预分配最大长度的空间
- 内存利用率低（很多序列远短于max_length）
- 无法共享相同的prefix

---

## 2. PagedAttention 原理

### 2.1 核心思想

**不为每个序列单独分配KV缓存，而是创建一个统一的KV缓存池，分成固定大小的块（block），动态分配给序列。**

```
传统方法：
┌─────────────────┐
│ Seq1: [------] │  预分配max_length，浪费空间
│ Seq2: [---]    │  预分配max_length，浪费更多
│ Seq3: [--------]│  预分配max_length
└─────────────────┘

PagedAttention：
┌─────────────────────────────────────┐
│ KV缓存池（统一管理）：               │
│ [Block_0][Block_1][Block_2]...[Block_N] │
│    ↑        ↑                ↑      │
│    │        │                │      │
│   Seq1    Seq2             Seq3    │
│  使用2块   使用1块          使用3块  │
└─────────────────────────────────────┘
```

### 2.2 优势

1. **高内存利用率** - 按需分配，没有预留空间
2. **支持动态长度** - 序列可以无限增长（直到缓存池满）
3. **支持 Prefix Caching** - 相同prefix的序列可以共享块
4. **灵活调度** - 块可以在序列间复用

---

## 3. 数据结构

### 3.1 KV 缓存池

```python
# 全局统一的KV缓存池
kv_cache = torch.zeros(
    2,                      # K 和 V
    num_layers,             # 模型层数（如24层）
    max_cached_blocks,      # 总块数（如128个块）
    block_size,             # 每块的slot数（如16个token）
    num_kv_heads,           # KV头数（如8个头）
    head_dim                # 每个头的维度（如128）
)

# 示例：形状为 [2, 24, 128, 16, 8, 128]
# 总共可以缓存：128块 × 16个token = 2048个token的KV
```

### 3.2 序列的关键属性

```python
class Sequence:
    token_ids: list[int]              # Token序列 [101, 102, 103, ...]
    block_table: list[int]            # 分配的块ID [2, 5, 7, 9]
    
    # 缓存状态
    num_cached_tokens: int            # 已缓存的token数
    num_cached_blocks: int            # 已缓存的块数
    
    # 块状态
    num_blocks: int                   # 总块数
    last_block_num_tokens: int        # 最后一个块的token数
    
    # 采样参数
    temperature: float                # 采样温度
    last_token: int                   # 最后一个token
```

### 3.3 Block Table（块表）

**块表是序列KV数据的"地址簿"，记录这个序列的KV存储在哪些块中。**

```python
# 序列1有35个token，block_size=16
seq.token_ids = [1, 2, 3, ..., 35]
seq.block_table = [2, 5, 7]

# 映射关系：
# tokens[0:16]  → Block_2（slots 32-47）
# tokens[16:32] → Block_5（slots 80-95）
# tokens[32:35] → Block_7（slots 112-114，部分填充）
```

### 3.4 Slot Mapping（槽位映射）

**槽位映射是临时生成的，指定新计算的KV应该写入哪些具体的slot。**

```python
# 每次prefill/decode时生成
slot_mappings = [112, 113, 114]  # 新KV写入这些位置

# 计算公式：
slot_idx = block_id * block_size + offset_in_block
```

---

## 4. 完整流程

### 4.1 初始化阶段

```python
def __init__(self, config, rank):
    # 1. 初始化模型
    self.model = Qwen3ForCausalLM(...)
    self.model = self.model.cuda(rank)
    
    # 2. 预热模型（测量峰值内存）
    self.warmup_model()
    
    # 3. 分配KV缓存
    self.allocate_kv_cache()
    
    # 4. 捕获CUDA图（加速decode）
    self.capture_cudagraph()
```

#### 详细：预热模型

```python
def warmup_model(self):
    """
    目的：运行一次最大负载，测量峰值内存使用量
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 创建最坏情况：最大batch × 最长序列
    max_tokens = 2048  # 系统最多同时处理2048个token
    max_model_length = 512  # 每个序列最长512个token
    batch_size = max_tokens // max_model_length  # 4个序列
    
    # 创建虚拟序列（全0占位）
    seqs = [
        Sequence(token_ids=[0]*512, block_size=16),
        Sequence(token_ids=[0]*512, block_size=16),
        Sequence(token_ids=[0]*512, block_size=16),
        Sequence(token_ids=[0]*512, block_size=16),
    ]
    
    # 运行一次prefill
    self.run(seqs, is_prefill=True)
    
    # 记录峰值内存
    peak_mem = torch.cuda.memory_stats()['allocated_bytes.all.peak']
```

#### 详细：分配KV缓存

```python
def allocate_kv_cache(self):
    """
    根据可用显存计算能分配多少KV缓存块
    """
    # 1. 计算可用内存
    free_mem, total_mem = torch.cuda.mem_get_info()
    total_free_mem = free_mem * 0.9  # 使用90%的空闲内存
    peak_mem = torch.cuda.memory_stats()['allocated_bytes.all.peak']
    current_mem = torch.cuda.memory_stats()['allocated_bytes.all.current']
    available_mem = total_free_mem - (peak_mem - current_mem)
    
    # 2. 计算每个块需要的字节数
    # block_bytes = block_size × 2(K和V) × num_layers × num_kv_heads × head_dim × dtype_size
    block_bytes = 16 × 2 × 24 × 8 × 128 × 2  # 假设float16
    # = 1,572,864 bytes ≈ 1.5 MB per block
    
    # 3. 计算能分配多少块
    num_blocks = int(available_mem // block_bytes)
    # 例如：10GB可用 → 约6500个块 → 可缓存104,000个token
    
    # 4. 分配统一的KV缓存池
    kv_cache = torch.zeros(
        2, num_layers, num_blocks, block_size,
        num_kv_heads, head_dim
    )
    
    # 5. 注入到模型的每个attention层
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, 'k_cache'):
            module.k_cache = kv_cache[0, layer_id]
            module.v_cache = kv_cache[1, layer_id]
            layer_id += 1
```

### 4.2 Prefill 阶段（处理输入prompt）

```python
def prepare_prefill(self, seqs: list[Sequence]):
    """
    准备prefill阶段的输入数据
    """
    input_ids = []
    slot_mappings = []
    seqlens_q = []
    seqlens_k = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    block_tables = []
    
    for seq in seqs:
        # 1. 确定需要处理的tokens（跳过已缓存的prefix）
        token_ids = seq.token_ids
        num_cached = seq.num_cached_tokens
        new_tokens = token_ids[num_cached:]
        
        # 2. 记录序列长度
        seqlens_q.append(len(new_tokens))
        seqlens_k.append(len(token_ids))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
        
        # 3. 构建slot映射（确定KV写入位置）
        for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
            if seq.num_cached_blocks + i != seq.num_blocks - 1:
                # 非最后块：填满整个块
                start = block_id * self.block_size
                end = (block_id + 1) * self.block_size
                slot_mappings.extend(range(start, end))
            else:
                # 最后块：部分填充
                start = block_id * self.block_size
                end = start + seq.last_block_num_tokens
                slot_mappings.extend(range(start, end))
        
        input_ids.extend(new_tokens)
    
    # 4. 如果有prefix cache，需要padding block_tables
    if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
        max_num_blocks = max(len(seq.block_table) for seq in seqs)
        for seq in seqs:
            padded = seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
            block_tables.append(padded)
    
    # 5. 设置全局上下文
    set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor(cu_seqlens_q),
        cu_seqlens_k=torch.tensor(cu_seqlens_k),
        max_seqlen_q=max(seqlens_q),
        max_seqlen_k=max(seqlens_k),
        slot_mapping=torch.tensor(slot_mappings),
        block_tables=torch.tensor(block_tables) if block_tables else None,
    )
    
    return torch.tensor(input_ids)
```

### 4.3 Decode 阶段（生成新token）

```python
def prepare_decode(self, seqs: list[Sequence]):
    """
    准备decode阶段的输入数据
    """
    input_ids = []
    context_lens = []
    slot_mappings = []
    block_tables = []
    
    for seq in seqs:
        # 1. 每个序列只处理最后一个token
        input_ids.append(seq.last_token)
        context_lens.append(len(seq))
        
        # 2. 计算新KV的写入位置（最后一个块的下一个slot）
        last_block_id = seq.block_table[-1]
        offset = seq.last_block_num_tokens - 1
        slot_idx = last_block_id * self.block_size + offset
        slot_mappings.append(slot_idx)
    
    # 3. Padding block_tables到相同长度
    max_num_blocks = max(len(seq.block_table) for seq in seqs)
    for seq in seqs:
        padded = seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
        block_tables.append(padded)
    
    # 4. 设置全局上下文
    set_context(
        is_prefill=False,
        slot_mapping=torch.tensor(slot_mappings),
        context_lens=torch.tensor(context_lens),
        block_tables=torch.tensor(block_tables),
    )
    
    return torch.tensor(input_ids)
```

### 4.4 运行模型

```python
def run(self, seqs: list[Sequence], is_prefill: bool):
    """
    执行一次推理步骤
    """
    # 1. 准备输入
    if is_prefill:
        input_ids = self.prepare_prefill(seqs)
    else:
        input_ids = self.prepare_decode(seqs)
    
    # 2. 运行模型
    logits = self.run_model(input_ids, is_prefill)
    
    # 3. 采样（仅rank 0）
    if self.rank == 0:
        token_ids = self.sampler(logits, temperatures)
    
    # 4. 重置上下文
    reset_context()
    
    return token_ids
```

---

## 5. Prefix Caching

### 5.1 什么是 Prefix Caching？

当多个请求有相同的前缀时，可以复用前缀的KV缓存，避免重复计算。

```python
# 请求1："请帮我写一首诗"
# 请求2："请帮我写一首诗，关于春天的"
#         [----相同前缀----] [新增部分]

# 请求2可以复用请求1的前缀KV缓存
```

### 5.2 实现机制

#### Query长度 vs Key长度

```python
# 没有prefix cache（首次处理）
seq.token_ids = [1, 2, 3, 4, 5]
seq.num_cached_tokens = 0

query_tokens = [1, 2, 3, 4, 5]  # 需要计算的
key_tokens = [1, 2, 3, 4, 5]    # 用于attention的
seqlen_q = 5
seqlen_k = 5
# Query == Key → 不需要读取缓存

# 有prefix cache（部分已缓存）
seq.token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
seq.num_cached_tokens = 5

query_tokens = [6, 7, 8]              # 只计算新的
key_tokens = [1, 2, 3, 4, 5, 6, 7, 8] # 需要全部
seqlen_q = 3
seqlen_k = 8
# Query < Key → 需要读取缓存
```

#### 为什么需要 padding block_tables？

```python
# Batch中有2个序列

# 序列1：有prefix cache
seq1.seqlen_q = 3, seq1.seqlen_k = 8
seq1.block_table = [0, 1, 2]  # 3个块

# 序列2：没有prefix cache
seq2.seqlen_q = 5, seq2.seqlen_k = 5
seq2.block_table = [3, 4]  # 2个块

# 判断条件
cu_seqlens_q[-1] = 8   # 3 + 5
cu_seqlens_k[-1] = 13  # 8 + 5
# 8 < 13 → 需要padding

# Flash Attention需要2D tensor（batch中所有block_table必须等长）
block_tables = [
    [0, 1, 2],  # 序列1
    [3, 4]      # 序列2 ← 长度不同！
]

# Padding后
block_tables = [
    [0, 1, 2],    # 序列1
    [3, 4, -1]    # 序列2，padding一个-1
]
```

### 5.3 为什么没有prefix cache就不需要padding？

```python
# 关键：Query == Key 时不需要读取缓存

# 场景：两个序列都没有prefix cache
seq1.seqlen_q = 6, seq1.seqlen_k = 6
seq2.seqlen_q = 3, seq2.seqlen_k = 3

cu_seqlens_q[-1] = 9
cu_seqlens_k[-1] = 9
# 9 == 9 → 不进入if分支

# 结果：block_tables 不会传给 Flash Attention
set_context(
    block_tables=None,  # 不需要！
)

# 原因：所有KV都是新计算的，不需要从缓存读取
```

---

## 6. 实际案例

### 案例1：首次处理（无缓存）

#### 输入

```python
# 3个用户同时请求
prompts = [
    "你好",                              # 用户1，3个token
    "请帮我写一首诗",                     # 用户2，7个token
    "今天天气怎么样",                     # 用户3，6个token
]

# Tokenize
seqs = [
    Sequence(token_ids=[101, 102, 103]),
    Sequence(token_ids=[201, 202, 203, 204, 205, 206, 207]),
    Sequence(token_ids=[301, 302, 303, 304, 305, 306]),
]

# BlockManager分配块（block_size=16）
seqs[0].block_table = [0]      # 需要1个块
seqs[1].block_table = [1]      # 需要1个块
seqs[2].block_table = [2]      # 需要1个块
```

#### Prefill处理

```python
# 1. 准备输入
input_ids = [101, 102, 103,  # seq1
             201, 202, 203, 204, 205, 206, 207,  # seq2
             301, 302, 303, 304, 305, 306]  # seq3

seqlens_q = [3, 7, 6]
seqlens_k = [3, 7, 6]

cu_seqlens_q = [0, 3, 10, 16]
cu_seqlens_k = [0, 3, 10, 16]

# 2. 构建slot映射
slot_mappings = [
    0, 1, 2,          # seq1 → Block_0的slots[0:3]
    16, 17, 18, 19, 20, 21, 22,  # seq2 → Block_1的slots[16:23]
    32, 33, 34, 35, 36, 37   # seq3 → Block_2的slots[32:38]
]

# 3. 检查是否需要block_tables
cu_seqlens_q[-1] == cu_seqlens_k[-1]  # 16 == 16
# 不需要padding，不需要block_tables

# 4. 设置上下文
set_context(
    is_prefill=True,
    cu_seqlens_q=[0, 3, 10, 16],
    cu_seqlens_k=[0, 3, 10, 16],
    slot_mapping=[0, 1, 2, 16, 17, ..., 37],
    block_tables=None,  # 不需要
)

# 5. 运行模型
hidden_states = model(input_ids)

# 6. Attention层写入KV
for i, slot_idx in enumerate(slot_mappings):
    k_cache[slot_idx] = computed_K[i]
    v_cache[slot_idx] = computed_V[i]

# 结果：
# Block_0: [K₁₀₁, K₁₀₂, K₁₀₃, _, _, ...]
# Block_1: [K₂₀₁, K₂₀₂, K₂₀₃, K₂₀₄, K₂₀₅, K₂₀₆, K₂₀₇, _, ...]
# Block_2: [K₃₀₁, K₃₀₂, K₃₀₃, K₃₀₄, K₃₀₅, K₃₀₆, _, ...]
```

#### Decode处理

```python
# 每个序列生成1个新token

# 1. 准备输入
input_ids = [103, 207, 306]  # 每个序列的最后一个token

context_lens = [3, 7, 6]  # 已处理的token数

slot_mappings = [
    0 * 16 + 3,   # seq1: Block_0的第4个slot（index 3）
    1 * 16 + 7,   # seq2: Block_1的第8个slot（index 23）
    2 * 16 + 6,   # seq3: Block_2的第7个slot（index 38）
] = [3, 23, 38]

# 2. Padding block_tables
block_tables = [
    [0],
    [1],
    [2],
]  # 长度都是1，不需要padding

# 3. 设置上下文
set_context(
    is_prefill=False,
    slot_mapping=[3, 23, 38],
    context_lens=[3, 7, 6],
    block_tables=[[0], [1], [2]],
)

# 4. 运行模型（使用CUDA图）
logits = graph.replay()

# 5. 采样
next_tokens = sampler(logits)  # [104, 208, 307]

# 6. 写入新KV
k_cache[3] = K₁₀₄
k_cache[23] = K₂₀₈
k_cache[38] = K₃₀₇
```

### 案例2：Prefix Caching

#### 场景

```python
# 第一次请求
prompt1 = "请帮我写一首诗"
tokens1 = [1, 2, 3, 4, 5]  # 5个token

# 第二次请求（有相同前缀）
prompt2 = "请帮我写一首诗，关于春天的"
tokens2 = [1, 2, 3, 4, 5, 6, 7, 8]  # 8个token
#          [---已缓存---] [新增]
```

#### 第一次请求

```python
seq1 = Sequence(token_ids=[1, 2, 3, 4, 5])
seq1.block_table = [10]  # BlockManager分配Block_10
seq1.num_cached_tokens = 0

# Prefill
input_ids = [1, 2, 3, 4, 5]
slot_mappings = [160, 161, 162, 163, 164]  # Block_10的slots[0:5]

# 运行后
seq1.num_cached_tokens = 5
seq1.num_cached_blocks = 0  # 块没填满，不算完全缓存

# KV缓存状态
# Block_10: [K₁, K₂, K₃, K₄, K₅, _, _, ...]
```

#### 第二次请求（复用prefix）

```python
seq2 = Sequence(token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
seq2.block_table = [10]  # 复用Block_10！
seq2.num_cached_tokens = 5  # 前5个token已缓存
seq2.num_cached_blocks = 0

# Prefill（只处理新token）
input_ids = [6, 7, 8]  # 只有3个新token

seqlen_q = 3  # 新计算的
seqlen_k = 8  # 用于attention的

cu_seqlens_q = [0, 3]
cu_seqlens_k = [0, 8]
# 3 < 8 → 需要block_tables！

slot_mappings = [165, 166, 167]  # Block_10的slots[5:8]

block_tables = [[10]]  # 告诉attention从Block_10读取缓存的K₁-K₅

# 设置上下文
set_context(
    is_prefill=True,
    cu_seqlens_q=[0, 3],
    cu_seqlens_k=[0, 8],
    slot_mapping=[165, 166, 167],
    block_tables=[[10]],  # 必须提供！
)

# Attention计算
# Q₆, Q₇, Q₈需要attend到K₁-K₈
# 其中K₁-K₅从缓存读取（通过block_tables定位）
# K₆-K₈是新计算的
```

### 案例3：多块序列

#### 场景

```python
# 一个很长的序列（35个token）
seq = Sequence(token_ids=[1, 2, 3, ..., 35])

# block_size = 16
# 需要3个块：
# Block_0: tokens[0:16]   (16个token，满)
# Block_1: tokens[16:32]  (16个token，满)
# Block_2: tokens[32:35]  (3个token，部分)

seq.block_table = [5, 8, 12]  # BlockManager分配的块ID
seq.num_cached_tokens = 0
```

#### Prefill处理

```python
# 1. 准备输入
input_ids = [1, 2, 3, ..., 35]  # 所有35个token

# 2. 构建slot映射
slot_mappings = []

# 处理Block_5（第1个块，满）
block_id = 5
slot_mappings.extend(range(5*16, 6*16))  # [80, 81, ..., 95]

# 处理Block_8（第2个块，满）
block_id = 8
slot_mappings.extend(range(8*16, 9*16))  # [128, 129, ..., 143]

# 处理Block_12（第3个块，部分）
block_id = 12
last_block_num_tokens = 3
slot_mappings.extend(range(12*16, 12*16+3))  # [192, 193, 194]

# 最终：
slot_mappings = [80, 81, ..., 95,      # Block_5的16个slot
                 128, 129, ..., 143,    # Block_8的16个slot
                 192, 193, 194]         # Block_12的3个slot
# 总共35个slot，对应35个token

# 3. 运行模型
hidden_states = model(input_ids)

# 4. 写入KV
for i in range(35):
    slot_idx = slot_mappings[i]
    k_cache[slot_idx] = computed_K[i]
    v_cache[slot_idx] = computed_V[i]

# KV缓存状态
# Block_5:  [K₁, K₂, ..., K₁₆]           (满)
# Block_8:  [K₁₇, K₁₈, ..., K₃₂]        (满)
# Block_12: [K₃₃, K₃₄, K₃₅, _, _, ...]  (部分)
```

#### 继续生成（Decode）

```python
# 生成第36个token

# 1. 准备输入
input_ids = [35]  # 最后一个token

context_lens = [35]  # 已处理35个token

# 2. 计算slot位置
last_block_id = 12
offset = 3  # Block_12已有3个token
slot_idx = 12 * 16 + 3 = 195

slot_mappings = [195]

# 3. Block_12还有空间，不需要新块
seq.block_table = [5, 8, 12]  # 不变

# 4. 运行模型
logits = model(input_ids)

# 5. 采样
next_token = sampler(logits)  # 假设是36

# 6. 写入新KV
k_cache[195] = K₃₆

# 更新序列状态
seq.token_ids.append(36)
seq.last_block_num_tokens = 4  # Block_12现在有4个token
```

#### Block满了需要新块

```python
# 继续生成到第48个token
# Block_12已满（16个token）

# BlockManager分配新块
seq.block_table = [5, 8, 12, 15]  # 新增Block_15

# 生成第49个token
last_block_id = 15
offset = 0  # 新块，从0开始
slot_idx = 15 * 16 + 0 = 240

# 写入
k_cache[240] = K₄₉
```

---

## 7. 总结

### 核心流程图

```
┌─────────────┐
│ 1. 初始化   │
│  - 预热模型 │
│  - 分配KV池 │
│  - 捕获图   │
└──────┬──────┘
       │
       v
┌─────────────────┐
│ 2. Prefill      │
│  - 处理prompt   │
│  - 写入KV缓存   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ 3. Decode       │  ◄───┐
│  - 生成1个token │      │
│  - 写入新KV     │      │
└──────┬──────────┘      │
       │                 │
       └─────────────────┘
       (循环直到结束)
```

### 关键数据结构

| 名称 | 作用 | 类型 | 生命周期 |
|------|------|------|----------|
| `kv_cache` | 统一的KV缓存池 | GPU tensor | 全局，持久 |
| `block_table` | 序列的块地址簿 | list[int] | 序列级，持久 |
| `slot_mapping` | 临时写入位置映射 | tensor | 每次run()生成 |
| `cu_seqlens_q/k` | 累积序列长度 | tensor | 每次run()生成 |

### 关键优化

1. **PagedAttention** - 块级管理，高内存利用率
2. **Prefix Caching** - 复用相同前缀，减少计算
3. **CUDA Graph** - 捕获计算图，加速decode
4. **多GPU协调** - NCCL通信，跨GPU同步

---

## 参考资料

- [PagedAttention论文](https://arxiv.org/abs/2309.06180)
- [vLLM官方文档](https://docs.vllm.ai/)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
