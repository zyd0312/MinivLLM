from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    """
    序列调度器，负责管理 LLM 推理请求的调度和执行

    主要功能：
    1. 维护等待队列（waiting）和运行队列（running）
    2. 根据资源限制（KV cache、batch size）调度序列执行
    3. 支持两种执行模式：
       - Prefill: 处理输入 prompt，一次性处理所有 prompt tokens
       - Decode: 生成输出 token，每次生成一个 token
    4. 当资源不足时，抢占（preempt）运行中的序列以释放 KV cache
    """
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        """
        初始化调度器

        Args:
            max_num_sequences: 单个 batch 中允许的最大序列数（batch size 限制）
            max_num_batched_tokens: 单个 batch 中允许的最大 token 数
                - Prefill 阶段：所有序列的 prompt tokens 总和
                - Decode 阶段：每个序列贡献 1 个 token
            max_cached_blocks: KV cache 中的最大 block 数量（显存限制）
            block_size: 每个 block 可以存储的 token 数量
            eos: 结束标记 token id
        """
        # KV cache 块管理器，负责分配和释放 KV cache 块
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        # 单个 batch 的 token 数上限（控制计算量）
        self.max_num_batched_tokens = max_num_batched_tokens
        # 单个 batch 的序列数上限（控制并发度）
        self.max_num_sequences = max_num_sequences
        # 等待队列：等待 prefill 的新序列
        self.waiting: deque[Sequence] = deque()
        # 运行队列：正在进行 decode 的序列
        self.running: deque[Sequence] = deque()
        # 结束标记
        self.eos = eos


    def is_finished(self):
        """
        检查是否所有序列都已完成

        Returns:
            bool: 当等待队列和运行队列都为空时返回 True
        """
        return len(self.waiting) == 0 and len(self.running) == 0

    def add_sequence(self, sequence: Sequence):
        """
        添加新序列到等待队列

        在加入队列前会进行容量检查：
        - 如果序列需要的 block 数超过 KV cache 总容量，直接拒绝
        - 这可以避免序列永远等待而无法被调度的情况

        Args:
            sequence: 要添加的序列

        Raises:
            ValueError: 当序列所需的 block 数超过 KV cache 总容量时
        """
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.
        capacity = len(self.block_manager.blocks)
        if sequence.num_blocks > capacity:
            raise ValueError(
                f"Sequence {sequence.seq_id} needs {sequence.num_blocks} blocks "
                f"({len(sequence)} tokens at block_size={self.block_manager.block_size}) "
                f"but the KV cache only holds {capacity}. "
                f"Raise max_cached_blocks or block_size, or shorten the prompt."
            )
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度函数，决定哪些序列在本轮执行

        调度策略（按优先级）：
        1. **Prefill 阶段**（优先）：从等待队列调度新序列
           - 为序列分配 KV cache blocks
           - 将序列加入运行队列
           - 一次性处理整个 prompt

        2. **Decode 阶段**：从运行队列调度正在生成的序列
           - 为每个序列追加一个新 token 的 KV cache 空间
           - 每个序列只生成一个 token

        资源限制：
        - max_num_sequences: batch 中的序列数上限
        - max_num_batched_tokens: batch 中的 token 数上限
        - KV cache 容量: 由 block_manager 管理

        抢占机制：
        - 当 KV cache 不足时，会抢占运行队列尾部的序列
        - 被抢占的序列释放 KV cache，重新加入等待队列

        Returns:
            tuple[list[Sequence], bool]:
                - 本轮调度的序列列表
                - 是否为 prefill 阶段（True）或 decode 阶段（False）

        Raises:
            RuntimeError: 当调度器无法取得进展时（死锁检测）
        """
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # An empty schedule is only legitimate when this call freed blocks by
        # preempting, so the next call can make progress. See the guard below.
        preempted = False

        # ============================================================
        # 阶段 1: 尝试从等待队列调度 prefill 任务
        # ============================================================
        # try schedule for prefilling from waiting queue if not exceeding limits
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            # 检查是否有足够的 KV cache blocks 和 token 预算
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                # 为序列分配 KV cache blocks
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                # prefill 阶段：token 数等于 prompt 长度
                current_scheduled_tokens += len(seq)
            else:
                # 无法满足资源要求，停止调度更多 prefill 任务
                break

        # 如果成功调度了 prefill 任务，直接返回（prefill 优先）
        if scheduled_sequences:
            return scheduled_sequences, True

        # ============================================================
        # 阶段 2: 从运行队列调度 decode 任务
        # ============================================================
        # try schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            # 检查是否有足够的 KV cache 空间追加一个新 token
            if not self.block_manager.can_append(seq):
                # KV cache 不足，需要抢占序列
                preempted = True
                if self.running:
                    # 还有其他序列在运行，抢占队列尾部的序列（FIFO 策略）
                    self.running.appendleft(seq)  # 当前序列放回队首
                    self.preempt(self.running.pop())  # 抢占队尾序列
                else:
                    # 只剩当前序列，抢占它并停止调度
                    self.preempt(seq)
                    break
            else:
                # KV Cache 充足，检查是否超过 batch 限制
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    # 已达到 batch 上限，停止调度
                    self.running.appendleft(seq)
                    break
                # append one token
                # 为新 token 分配 KV cache 空间
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                # decode 阶段：每个序列只贡献 1 个 token
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        # 将已调度的序列按原顺序放回运行队列头部
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
        elif not preempted and (self.waiting or self.running):
            # Nothing was scheduled and nothing was preempted, so no engine state
            # changed: every later schedule() would take the same decisions and
            # LLMEngine.generate() would spin forever. Fail loudly instead.
            # 死锁检测：如果有任务但无法调度且未抢占任何序列，说明出现了死锁
            # 可能原因：
            # 1. 某个序列需要的 blocks 超过 KV cache 总容量
            # 2. block 引用计数泄漏，导致 blocks 无法释放
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} of "
                f"{len(self.block_manager.blocks)} blocks free. "
                "This means either a sequence that cannot fit in the KV cache, or "
                "blocks leaked because their ref_count never returned to 0."
            )

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        """
        抢占（中断）一个正在运行的序列

        抢占操作包括：
        1. 释放序列占用的所有 KV cache blocks
        2. 将序列状态改为 WAITING
        3. 将序列重新加入等待队列头部（优先重新调度）

        抢占通常发生在 KV cache 不足时，需要为新序列或其他序列腾出空间

        Args:
            seq: 要抢占的序列
        """
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """
        生成后处理：更新序列状态并检查是否完成

        在每轮 decode 生成新 token 后调用，负责：
        1. 将生成的 token 追加到序列
        2. 检查序列是否满足停止条件
        3. 清理已完成序列的资源

        停止条件（满足任一即停止）：
        - 生成了 EOS（结束标记）token
        - 达到 max_tokens 限制（completion tokens 数量）
        - 达到 max_model_length 限制（总序列长度，包括 prompt）

        Args:
            seqs: 本轮生成的序列列表
            token_ids: 对应生成的 token id 列表
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                # 序列完成，释放 KV cache 并从运行队列移除
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)