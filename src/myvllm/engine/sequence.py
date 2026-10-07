from enum import Enum, auto
import math
from itertools import count
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    """序列的生命周期状态枚举

    WAITING: 序列等待调度执行
    RUNNING: 序列正在生成中
    FINISHED: 序列已完成生成
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """表示一个文本生成序列

    这个类管理一个生成请求的完整生命周期，包括：
    - token管理（prompt和生成的tokens）
    - 内存块管理（使用PagedAttention的block_table）
    - 采样参数配置
    - 序列状态跟踪

    主要用于vLLM引擎的请求调度和KV cache管理。
    """
    # 全局计数器，用于为每个序列分配唯一ID
    counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        """初始化一个新序列

        Args:
            token_ids: 初始的prompt token列表
            block_size: PagedAttention中每个物理块包含的token数量
            sampling_params: 生成时的采样参数配置
        """
        # PagedAttention块大小，每个物理块存储的token数量
        self.block_size = block_size

        # 分配唯一的序列ID
        self.seq_id = next(Sequence.counter)

        # 序列当前状态，初始为WAITING
        self.status = SequenceStatus.WAITING

        # 复制token列表，避免外部修改影响内部状态
        self.token_ids = copy(token_ids)

        # 记录最后一个token，用于快速访问
        self.last_token = self.token_ids[-1] if self.token_ids else None

        # 当前序列的总token数
        self.num_tokens = len(self.token_ids)
        # prompt部分的token数（不会改变）
        self.num_prompt_tokens = len(self.token_ids)

        # 已缓存的token数量，用于prefix caching优化
        self.num_cached_tokens = 0

        # 物理块表，记录分配给该序列的物理块ID列表
        # 用于PagedAttention的KV cache管理
        self.block_table = []

        # 从sampling_params提取生成相关的参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    def __len__(self):
        """返回序列当前的总token数"""
        return self.num_tokens

    def __getitem__(self, idx):
        """支持通过索引访问token_ids"""
        return self.token_ids[idx]

    @property
    def is_finished(self):
        """检查序列是否已完成生成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """返回已生成的completion token数量（不包括prompt）"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回prompt部分的token列表"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回生成的completion部分的token列表"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """返回已缓存的完整块数量

        用于prefix caching，计算有多少完整块可以复用
        """
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        """返回当前序列需要的总块数

        根据token数量和block_size计算需要的物理块数量
        """
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        """返回最后一个块中实际包含的token数量

        最后一个块通常不满，这个属性返回其实际token数
        """
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    def block(self, i):
        """获取第i个块的token列表

        Args:
            i: 块索引，从0开始

        Returns:
            该块包含的token列表（最后一个块可能不满）

        用于PagedAttention中访问特定块的tokens
        """
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            # 最后一个块，返回剩余的所有tokens
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            # 中间块，返回完整的block_size个tokens
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    def append_token(self, token_id):
        """在序列末尾添加一个新生成的token

        Args:
            token_id: 新生成的token ID

        用于decode阶段每次生成一个token后更新序列
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 

    def __getstate__(self):
        """序列化序列状态（用于pickle或传输）

        为了优化内存和传输效率：
        - 在prefill阶段（num_completion_tokens == 0）：保存完整的token_ids
        - 在decode阶段：只保存最后一个token，节省内存

        Returns:
            包含序列核心状态的元组
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            # 根据阶段决定保存完整token_ids还是只保存last_token
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state):
        """反序列化序列状态（用于pickle或传输）

        Args:
            state: 从__getstate__返回的状态元组

        根据是否为prefill阶段恢复完整token_ids或只恢复last_token
        """
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            last_token_or_ids
        ) = state

        # 判断当前处于prefill还是decode阶段
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens
        if num_completion_tokens == 0:
            # Prefill阶段：last_token_or_ids是完整的token_ids列表
            self.token_ids = last_token_or_ids
        else:
            # Decode阶段：last_token_or_ids只是最后一个token
            # 在decode阶段，完整的token_ids由KV cache重建，这里只保留最后一个
            self.token_ids = [last_token_or_ids]

        # 恢复last_token属性
        self.last_token = self.token_ids[-1] if self.token_ids else None
