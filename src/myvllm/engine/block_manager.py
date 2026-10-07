import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence


class Block:
    """
    表示 GPU 内存中的一个物理块，用于存储 KV cache。

    每个 Block 对应 GPU 显存中的一段连续空间，可以存储固定数量（block_size）的 token 的 KV cache。
    通过引用计数和哈希值支持多个序列共享同一个物理块（prefix caching）。
    """
    def __init__(self, block_id):
        self.block_id = block_id      # 块的唯一标识符
        self.hash = -1                # 块内容的哈希值，-1 表示未计算或块未满
        self.ref_count = 0            # 引用计数，记录有多少个序列正在使用这个块
        self.token_ids = []           # 块中存储的 token ID 列表

    def update(self, h: int, token_ids: list[int]):
        """更新块的哈希值和 token 内容"""
        self.hash = h
        self.token_ids = token_ids

    def reset(self):
        """重置块状态，准备分配给新序列"""
        self.hash = -1
        # reset() is only reached via BlockManager._allocate_block, which takes the
        # block off the free list on behalf of exactly one sequence. Allocation is
        # therefore the first reference: leaving this at 0 makes the matching
        # deallocate() drive ref_count to -1, so the block is never freed.
        self.ref_count = 1
        self.token_ids = []

class BlockManager:
    """
    管理 GPU 内存中的物理块分配，实现 PagedAttention 的核心逻辑。

    BlockManager 负责：
    1. 物理块的分配与释放
    2. 基于哈希的 prefix caching（前缀缓存共享）
    3. 维护空闲块和已使用块的状态
    4. 为序列分配和管理 block_table（逻辑块到物理块的映射）

    工作原理：
    - 将 GPU 显存划分为固定大小的物理块
    - 每个序列维护一个 block_table，记录其使用的物理块 ID
    - 通过引用计数支持多个序列共享相同的物理块（prefix caching）
    - 使用哈希值快速查找是否存在相同内容的块
    """
    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化 BlockManager

        Args:
            num_blocks: GPU 显存中可用的物理块总数
            block_size: 每个块可以存储的 token 数量
        """
        # 每个块包含的 token 数量
        self.block_size: int = block_size
        # 所有物理块的列表
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希值到块 ID 的映射，用于 prefix caching
        self.hash_to_block_id: dict[int, int] = {}
        # 空闲块 ID 队列，deque — 双端队列
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用块 ID 集合
        self.used_block_ids: set[int] = set()

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        """
        计算 token 序列的哈希值，支持增量哈希计算

        为了支持 prefix caching，哈希计算是上下文相关的：
        - 相同的 token_ids 在不同的前缀上下文中会产生不同的哈希值
        - 这确保了只有完整前缀匹配的块才能被共享

        Args:
            token_ids: 当前块的 token ID 列表
            prefix_hash_value: 前一个块的哈希值，-1 表示这是第一个块

        Returns:
            计算得到的哈希值
        """
        # xxHash 是非加密哈希，速度极快（比 MD5/SHA 快一个数量级）
        h = xxhash.xxh64()
        # 如果有前缀哈希值，将其纳入计算，确保上下文敏感性
        if prefix_hash_value != -1:
            # 把前驱哈希 8 字节小端 塞进哈希流
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配一个物理块给序列使用

        将块从空闲列表移动到已使用列表，并重置其状态。
        这是内部方法，由 allocate() 和 append() 调用。

        Args:
            block_id: 要分配的块 ID

        Returns:
            分配后的 Block 对象
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()  # 设置 ref_count = 1
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """
        释放一个物理块，将其归还到空闲列表

        只有当块的引用计数降为 0 时才会被释放。

        Args:
            block_id: 要释放的块 ID
        """
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        # Clearing token_ids deliberately keeps the prefix cache scoped to blocks that
        # are still referenced: a freed block can no longer match in allocate(), so a
        # cache hit never spans a finished sequence. Reuse across sequences is not
        # enabled yet because the prefill path cannot consume it -- the Triton kernel
        # in layers/attention.py attends only over the K/V computed in that pass and
        # ignores context.block_tables, and qwen3 derives RoPE positions from
        # cu_seqlens_q, so both would be wrong by num_cached_tokens. Enabling reuse
        # means a paged prefill kernel (cu_seqlens_q != cu_seqlens_k) plus a position
        # offset; until then this line is what keeps the engine correct.
        block.token_ids = []  # 清空 token_ids，禁用跨序列的缓存重用
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲块来为序列分配内存

        在实际分配之前调用，用于调度决策。

        Args:
            seq: 要检查的序列

        Returns:
            如果有足够的空闲块返回 True，否则返回 False
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """
        为序列分配物理块，支持 prefix caching

        这是初始分配方法，通常在 prefill 阶段调用。对序列的每个逻辑块：
        1. 计算块内容的哈希值（仅对已满的块）
        2. 检查是否存在相同内容的块（cache hit）
        3. 如果命中缓存，增加该块的引用计数并共享
        4. 如果未命中，分配新的物理块

        通过 prefix caching，多个序列可以共享相同前缀的 KV cache，
        显著减少内存使用和计算开销。

        Args:
            seq: 要分配内存的序列
        """
        h = -1  # 前缀哈希值，初始为 -1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # 只为已满的块计算哈希值，部分填充的块哈希值保持为 -1
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            # 检查是否缓存未命中或发生哈希碰撞
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # 缓存命中：找到了相同内容的块
                seq.num_cached_tokens += self.block_size  # 等于 len(token_ids)
                # 处理边界情况：块有哈希值但尚未被分配
                if block_id not in self.used_block_ids:
                    # Unreachable while _deallocate_block clears token_ids: a freed
                    # block has token_ids == [] and fails the match above. Kept for
                    # when cross-sequence reuse is enabled.
                    block = self._allocate_block(block_id)
                else:
                    # 块已被分配，增加引用计数以共享
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # 缓存未命中：需要分配新块
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                # 将新块的哈希值注册到映射表（仅对已满的块，h != -1）
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            # 将物理块 ID 添加到序列的 block_table
            seq.block_table.append(block.block_id)

    def deallocate(self, seq: Sequence) -> None:
        """
        释放序列占用的所有物理块

        遍历序列的 block_table，递减每个块的引用计数。
        当块的引用计数降为 0 时，将其归还到空闲列表。

        Args:
            seq: 要释放内存的序列
        """
        # 遍历序列的所有物理块
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            # 如果引用计数降为 0，释放该块
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 清空序列的 block_table 和缓存信息
        seq.block_table = []
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以为序列追加新 token（可能需要分配新块）

        在 decode 阶段，每生成一个新 token 都会调用此方法检查是否有足够的内存。
        只有当新 token 会导致需要分配新块时（即 num_tokens % block_size == 1），
        才需要检查是否有空闲块。

        注意：此方法在新 token 已经计入 seq.num_tokens 之后调用，因此：
        - num_tokens % block_size == 1：新 token 是新块的第一个 token，需要分配新块
        - num_tokens % block_size == 0：新 token 刚好填满上一个块，无需分配新块
        - 其他情况：新 token 在当前块中间，无需分配新块

        Args:
            seq: 要检查的序列

        Returns:
            如果可以追加返回 True，否则返回 False
        """
        # Called after the new token is already counted in seq.num_tokens, so the
        # condition must match append()'s allocation branch: a fresh block is only
        # needed when that token is the first of a new block (num_tokens % size == 1).
        # At num_tokens % size == 0 the token still fits in the block the sequence
        # already holds, and append() merely finalizes its hash.
        # 新 token 刚好填满上一个块，无需分配新块
        if seq.num_tokens % self.block_size == 1:
            # 需要分配新块，检查是否有空闲块
            return len(self.free_block_ids) > 0
        return True

    def append(self, seq: Sequence) -> None:
        """
        为序列追加新 token，并在必要时分配新块
        token 的追加由 seq.append_token() 完成，
        BlockManager.append() 只负责物理块的状态管理，而部分填充的块不需要状态更新
        
        在 decode 阶段，每生成一个新 token 后调用此方法。根据当前块的填充状态：
        1. 如果最后一个块刚好填满（num_tokens % block_size == 0）：
           - 计算该块的哈希值并注册到 hash_to_block_id
           - 为后续的 prefix caching 做准备
        2. 如果需要新块（num_tokens % block_size == 1）：
           - 分配一个新的物理块
           - 将其添加到序列的 block_table
        3. 其他情况（块部分填充）：
           - 不做任何操作，token 已经在 seq.token_ids 中

        注意：此方法假设新 token 已经添加到 seq.token_ids 中，
        但对应的物理块尚未分配（如果需要的话）。

        Args:
            seq: 要追加 token 的序列
        """
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # 情况 1：最后一个块刚好填满，计算并注册其哈希值
        if seq.num_tokens % self.block_size == 0:
            # 计算前缀哈希值：如果只有一个块则为 -1，否则使用倒数第二个块的哈希
            prefix_hash = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash
            h = self.compute_hash(token_ids=seq.block(seq.num_blocks - 1), prefix_hash_value=prefix_hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # 情况 2：需要分配新块（新 token 是新块的第一个 token）
        elif seq.num_tokens % self.block_size == 1:
            # 前一个块应该已经完成哈希计算
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # 情况 3：块部分填充，无需操作
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
