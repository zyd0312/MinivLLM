import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from myvllm.utils import get_context


# VocabParallelEmbedding: 词汇表并行的嵌入层
# 在词汇表维度上进行分片（而非嵌入维度），将大词汇表分布到多个 GPU 上
# 例如：50000 个词汇，4 个 GPU，每个 GPU 负责约 12500 个词汇的嵌入
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # 获取张量并行（Tensor Parallelism）的总 GPU 数量和当前 GPU 的编号
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # 保存原始的词汇表大小
        self.num_embeddings = num_embeddings
        # 将词汇表大小向上取整，使其能被 tp_size 整除
        # 例如：50000 个词汇，4 个 GPU -> padding 到 50000 -> 每个 GPU 12500
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        # 当前 GPU 上分配的词汇数量
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim

        # 创建当前分片的权重参数：[当前分片词汇数, 嵌入维度]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 绑定自定义的权重加载器，用于从检查点加载分片权重
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        自定义权重加载器：从完整的权重张量中加载当前 GPU 负责的分片

        Args:
            param: 当前 GPU 的参数（需要被填充）
            loaded_weights: 从检查点加载的完整权重张量 [num_embeddings, embedding_dim]
        """
        param_data = param.data

        # 计算当前分片在完整词汇表中的起始偏移
        offset = self.tp_rank * self.num_embeddings_per_partition
        shard_size = self.num_embeddings_per_partition

        # 计算当前分片实际对应的原始词汇表范围
        # 需要处理 padding 情况：最后一个 GPU 可能包含超出原始词汇表的 padding 部分
        actual_start = min(offset, self.num_embeddings)
        actual_end = min(offset + shard_size, self.num_embeddings)
        actual_size = max(0, actual_end - actual_start)

        if actual_size > 0:
            # 从完整权重中提取当前分片对应的部分
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)
            param_data[:actual_size].copy_(sharded_weights)

        # 如果有 padding 部分（超出原始词汇表），用零填充
        if actual_size < shard_size:
            param_data[actual_size:].zero_()

    # 从 token IDs → 嵌入向量（输入层）
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：并行嵌入查找

        核心思路：
        1. 每个 GPU 只处理属于自己分片范围内的 token IDs
        2. 通过掩码标识哪些 token 属于当前 GPU
        3. 使用 all_reduce 汇总所有 GPU 的结果

        Args:
            x: token IDs 张量 [batch_size, seq_len]

        Returns:
            嵌入向量 [batch_size, seq_len, embedding_dim]
        """
        # 创建掩码：标识哪些 token 属于当前 GPU 的分片范围
        # 条件1: token_id >= 当前分片起始位置
        # 条件2: token_id < 当前分片结束位置
        # 条件3: token_id < 原始词汇表大小（排除 padding 部分）
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
               (x < self.num_embeddings)
        # 将全局 token ID 转换为当前分片内的局部 ID
        # 例如：GPU 1 (rank=1)，分片大小=12500，token_id=15000 -> 局部 ID=2500
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        # 使用局部 ID 进行嵌入查找
        output = F.embedding(x, self.weight)

        if dist.get_world_size() > 1:
            # 再次应用掩码：将不属于当前 GPU 的 token 对应的嵌入向量置零
            # 这样可以避免越界 ID 被映射到 ID 0 的嵌入向量
            output = mask.unsqueeze(1) * output
            # all_reduce 求和：汇总所有 GPU 的结果
            # 由于每个 token 只在一个 GPU 上有非零值，求和后得到完整的嵌入结果
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

# ParallelLMHead: 并行语言模型头
# 继承自 VocabParallelEmbedding，实现权重共享（weight tying）
# 用于将隐藏状态映射到词汇表 logits，通常与输入嵌入层共享权重
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    # x: [batch_size, seq_len, hidden_size]
    # weight: [vocab_size_per_partition, hidden_size]
    # 任务：从隐藏状态 → logits（输出层），给定隐藏状态，计算每个词汇的得分（logits），用于预测下一个 token
    # 隐藏维度通常命名为 hidden_size、 embedding_dim、d_model 等，表示 Transformer 模型的特征维度
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算语言模型 logits

        处理流程：
        1. 在 prefill 阶段，只保留每个序列的最后一个 token（用于生成下一个 token）
        2. 计算 logits = x @ weight.T（线性变换）
        3. 通过 gather 操作将所有 GPU 的分片 logits 收集到 GPU 0

        Args:
            x: 隐藏状态 [batch_size, seq_len, hidden_size] 或 [batch_size, hidden_size]

        Returns:
            logits: [batch_size, vocab_size]（仅在 GPU 0 上返回完整结果）
        """
        context = get_context()
        if context.is_prefill:
            # prefill 阶段：提取每个序列的最后一个 token
            # cu_seqlens_q 是累积序列长度，例如 [0, 5, 8, 12] 表示 3 个序列，长度分别为 5, 3, 4
            # last_indices = [5, 8, 12] - 1 = [4, 7, 11] 是每个序列最后一个 token 的索引
            last_token = context.cu_seqlens_q[1:] - 1  # 排除第一个元素（总是 0）
            x = x[last_token].contiguous()

        # 计算 logits: [batch_size, vocab_size_per_partition]
        # F.linear 会自动转置权重矩阵：x @ weight.T
        logits = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            # 准备接收所有 GPU 的 logits（仅在 GPU 0 上分配内存）
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # gather 操作：将所有 GPU 的 logits 收集到 GPU 0
            # 每个 GPU 贡献一部分词汇表的 logits
            # dst=0 表示收集到 GPU 0 上，其他 GPU 不需要分配内存
            dist.gather(logits, gather_list=all_logits, dst=0)
            # 在 GPU 0 上拼接所有分片的 logits
            if self.tp_rank == 0:
                # 拼接后得到 [batch_size, padded_vocab_size]
                logits = torch.cat(all_logits, dim=-1)
                # 裁剪到原始词汇表大小（去除 padding 部分）
                logits = logits[..., :self.num_embeddings]

        return logits