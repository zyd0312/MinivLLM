import torch.nn as nn
import torch
import torch.distributed as dist
import os

class LinearBase(nn.Module):
    """
    张量并行线性层的基类

    定义了支持张量并行的线性层的基本接口，包括：
    - 自定义权重加载机制（weight_loader），用于从完整检查点加载切分后的权重
    - 张量并行的基本属性（tp_dim, tp_rank, tp_size）
    """

    def __init__(
        self,
        input_size: int,       # 输入特征维度
        output_size: int,      # 输出特征维度（可能是切分后的维度）
        bias: bool = True,     # 是否使用偏置
        tp_dim: int | None = None  # 张量并行的切分维度：0表示列切分，1表示行切分，None表示不切分
    ):
        super().__init__()
        # 设置张量并行相关属性
        self.tp_dim = tp_dim                  # 切分维度
        self.tp_rank = dist.get_rank()        # 当前 GPU 在并行组中的排名
        self.tp_size = dist.get_world_size()  # 并行组的总 GPU 数量

        # 创建权重参数，并附加自定义加载器
        # weight_loader 允许从完整模型权重中提取当前 GPU 对应的切片
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # 创建偏置参数
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        自定义权重加载器，子类必须实现

        Args:
            param: 当前 GPU 上的参数（已切分）
            loaded_weights: 从检查点加载的完整权重
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，子类必须实现"""
        raise NotImplementedError("Subclasses should implement this method.")

"""
权重加载机制说明：

使用场景：
1. 首先使用张量并行/流水线并行方法在 GPU 上部署一个可能随机初始化的模型
2. 然后从保存的检查点加载完整模型权重到这个分布式模型

加载流程：
for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # 完整模型参数，例如 (4096, 4096)

        # 检查参数是否有自定义的 weight_loader
        if hasattr(param, 'weight_loader'):
            # 调用自定义 weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader 会自动：
            # 1. 提取当前 GPU 对应的切片
            # 2. 将切片复制到 param.data
        else:
            # 默认行为：直接复制（适用于非并行参数）
            param.data.copy_(loaded_weight)
"""

# 最简单的线性层：ReplicatedLinear
# 所有 GPU 都保存完整的权重副本，不进行切分
class ReplicatedLinear(LinearBase):
    """
    复制型线性层（无张量并行）

    每个 GPU 上都保存完整的权重副本，不进行切分。
    适用于：
    - 参数量较小的层
    - 不需要并行化的层
    - 用作对比基准
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """直接复制完整权重，无需切分"""
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准线性层前向传播"""
        return nn.functional.linear(x, self.weight, self.bias)

# 列切分线性层：ColumnParallelLinear
# 沿输出维度切分权重矩阵，每个 GPU 负责计算部分输出
class ColumnParallelLinear(LinearBase):
    """
    列并行线性层（沿输出维度切分）

    原理：
    - 完整权重矩阵：W[output_size, input_size]
    - 切分后每个 GPU：W_i[output_size/tp_size, input_size]
    - 前向传播：Y_i = X @ W_i^T  （每个 GPU 计算部分输出）

    示例（假设 tp_size=4, output_size=4096）：
    - GPU 0: 负责输出的第 0~1023 维
    - GPU 1: 负责输出的第 1024~2047 维
    - GPU 2: 负责输出的第 2048~3071 维
    - GPU 3: 负责输出的第 3072~4095 维

    适用场景：
    - Transformer 的 FFN 第一层（hidden -> intermediate）
    - Attention 的输出投影之前的 QKV 投影
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        # 每个 GPU 只存储 output_size/tp_size 的输出维度
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        从完整权重中提取当前 GPU 对应的列切片

        Args:
            param: 当前 GPU 的参数 [output_size/tp_size, input_size]
            loaded_weights: 完整权重 [output_size, input_size]
        """
        param_data = param.data
        # 完整权重的输出维度大小
        full_data_output_size = loaded_weights.size(0)
        # 切分后每个 GPU 的输出维度大小
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # 计算当前 GPU 的起始索引
        start_index = self.tp_rank * shard_size
        # 提取对应的切片：loaded_weights[start_index:start_index+shard_size, :]
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：每个 GPU 计算部分输出

        输入：x [batch, seq_len, input_size]
        输出：y [batch, seq_len, output_size/tp_size]
        """
        return nn.functional.linear(x, self.weight, self.bias)

# 合并列并行线性层：将多个矩阵合并后再进行列切分
class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层

    功能：
    - 将多个独立的线性层（如 Q、K、V 投影）合并成一个大矩阵
    - 然后对合并后的矩阵进行列切分
    - 优势：一次矩阵乘法完成多个投影，提高计算效率

    示例（QKV 投影，tp_size=4）：
    - 原始：Q[4096, 4096], K[4096, 4096], V[4096, 4096]
    - 合并：QKV[12288, 4096]
    - 切分后每个 GPU：QKV_i[3072, 4096]

    权重布局（每个 GPU）：
    [Q_shard | K_shard | V_shard]
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],  # 例如 [q_size, k_size, v_size]
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        # 合并后的总输出维度
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        加载合并矩阵中的某个子矩阵

        Args:
            param: 当前 GPU 的合并参数 [sum(output_sizes)/tp_size, input_size]
            loaded_weights: 要加载的子矩阵完整权重（如 Q 的完整权重）
            loaded_weight_id: 子矩阵索引（0=Q, 1=K, 2=V）

        示例：
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        加载到：
        merged_layer = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[4096, 4096, 4096],  # Q, K, V
        )
        每个 GPU 存储 [3072, 4096]，布局为 [Q_1024 | K_1024 | V_1024]
        """
        param_data = param.data
        # 计算当前子矩阵在合并参数中的起始偏移（切分后）
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        # 计算当前子矩阵的切分大小
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # 定位到合并参数中对应的切片
        param_data = param_data.narrow(0, offset, shard_size)
        # 从完整权重中提取当前 GPU 对应的切片
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    """
    专门用于 Attention 的 QKV 投影的列并行线性层

    特点：
    - 支持 Multi-Head Attention (MHA) 和 Grouped-Query Attention (GQA)
    - Q 头数可能大于 KV 头数（GQA 中多个 Q 头共享一组 KV）
    - 自动计算每个 GPU 的头数分配

    权重布局（每个 GPU）：
    [Q_heads_local | K_heads_local | V_heads_local]

    参数计算示例（tp_size=4）：
    - num_heads=32, num_kv_heads=8, head_size=128
    - 完整输出：32*128 + 8*128 + 8*128 = 5120
    - 每个 GPU：8*128 + 2*128 + 2*128 = 1536
    """

    def __init__(
        self,
        input_size: int,
        head_size: int,           # 每个注意力头的维度
        num_heads: int,           # Query 头的总数（必须能被 tp_size 整除）
        num_kv_heads: int | None = None,  # Key/Value 头的总数（默认等于 num_heads，即 MHA）
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads  # 默认 MHA
        self.head_size = head_size
        self.num_heads = num_heads // self.tp_size        # 每个 GPU 的 Q 头数
        self.num_kv_heads = num_kv_heads // self.tp_size  # 每个 GPU 的 KV 头数
        # 计算每个 GPU 的输出维度
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # 传递总输出维度给父类（父类会自动除以 tp_size）
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        """
        加载 Q、K 或 V 的权重

        Args:
            param: 当前 GPU 的合并参数 [output_size_local, input_size]
            loaded_weights: Q/K/V 的完整权重
            load_weight_id: 'q', 'k' 或 'v'

        权重布局说明：
        - param 的维度：[head_size * (num_heads_local + 2*num_kv_heads_local), input_size]
        - 布局：[Q部分 | K部分 | V部分]
        """
        param_data = param.data
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"

        # 计算在合并参数中的偏移和大小
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        # 定位到合并参数中的对应位置
        param_data = param_data.narrow(0, offset, shard_size)
        # 从完整权重中提取当前 GPU 对应的切片
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):
    """
    行并行线性层（沿输入维度切分）

    原理：
    - 完整权重矩阵：W[output_size, input_size]
    - 切分后每个 GPU：W_i[output_size, input_size/tp_size]
    - 前向传播：
      1. 每个 GPU 计算部分乘积：Y_i = X_i @ W_i^T
      2. All-Reduce 汇总所有 GPU 的结果：Y = sum(Y_i)

    示例（假设 tp_size=4, input_size=4096）：
    - GPU 0: W_0[:, 0:1024], 处理输入的第 0~1023 维
    - GPU 1: W_1[:, 1024:2048], 处理输入的第 1024~2047 维
    - GPU 2: W_2[:, 2048:3072], 处理输入的第 2048~3071 维
    - GPU 3: W_3[:, 3072:4096], 处理输入的第 3072~4095 维

    典型用法：
    - 通常与 ColumnParallelLinear 配对使用
    - Transformer FFN：hidden -> intermediate (Column) -> hidden (Row)
    - Attention：hidden -> QKV (Column) -> hidden (Row)

    关键特性：
    - 需要 All-Reduce 通信来汇总各 GPU 的部分结果
    - 输入 x 通常已经被前面的 ColumnParallelLinear 切分过
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        # 每个 GPU 只存储 input_size/tp_size 的输入维度
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        从完整权重中提取当前 GPU 对应的行切片

        Args:
            param: 当前 GPU 的参数 [output_size, input_size/tp_size]
            loaded_weights: 完整权重 [output_size, input_size]
        """
        param_data = param.data
        # 完整权重的输入维度大小
        full_data_input_size = loaded_weights.size(1)
        # 切分后每个 GPU 的输入维度大小
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        # 计算当前 GPU 的起始索引
        start_index = self.tp_rank * shard_size
        # 提取对应的切片：loaded_weights[:, start_index:start_index+shard_size]
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算部分乘积，然后 All-Reduce 汇总

        输入：x [batch, seq_len, input_size/tp_size]（通常来自前面的 ColumnParallel）
        输出：y [batch, seq_len, output_size]（完整输出）

        计算流程：
        1. 本地计算：y_local = x @ W_local^T
        2. All-Reduce：y = sum(y_local_0, y_local_1, ..., y_local_N)
        """
        result = nn.functional.linear(x, self.weight, self.bias)
        # 如果使用了张量并行，需要 All-Reduce 汇总各 GPU 的结果
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result




if __name__ == "__main__":
    """
    测试程序：验证各种张量并行线性层的正确性

    运行方法：
    1. cd src/myvllm/layers
    2. CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 linear.py

    测试内容：
    - ColumnParallelLinear：列并行线性层
    - MergedColumnParallelLinear：合并列并行线性层
    - QKVColumnParallelLinear：QKV 投影层
    - RowParallelLinear：行并行线性层

    验证方法：
    对比张量并行版本和单 GPU 版本的输出是否一致（allclose 检查）
    """

    def _init_dist():
        """初始化分布式环境（如果环境变量不存在则使用单进程模式）"""
        # 检查是否在分布式环境中
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            # 单进程模式
            rank = 0
            world_size = 1
            local_rank = 0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("警告：未检测到分布式环境变量，使用单进程模式")
            return rank, world_size, local_rank, device

        # 分布式模式
        rank = int(os.environ["RANK"])               # 当前进程的全局排名
        world_size = int(os.environ["WORLD_SIZE"])   # 总进程数
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 本地排名（单机内）
        backend = "nccl" if torch.cuda.is_available() else "gloo"  # GPU 用 nccl，CPU 用 gloo

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                device_id=local_rank,
                )
        else:
            device = torch.device("cpu")
            dist.init_process_group( backend=backend, init_method="env://", )
        return rank, world_size, local_rank, device

    # 测试 1：列并行线性层
    @torch.no_grad()
    def test_column_parallel(device):
        """
        测试 ColumnParallelLinear 的正确性

        验证策略：
        1. 单 GPU 基准：使用完整权重计算 y_single = x @ W^T
        2. 多 GPU 并行：每个 GPU 计算部分输出 y_i = x @ W_i^T
        3. All-Gather 收集所有 GPU 的输出并拼接
        4. 对比 y_full 和 y_single 是否一致
        """
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        # 维度设置（自动适配 tp_size 以确保可整除）
        in_features = 1024 * tp_size
        out_features = 1024 * tp_size
        batch = 4

        # 使用固定随机种子确保所有进程生成相同的输入和权重
        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_full = torch.randn(out_features, in_features, generator=g)
        b_full = torch.randn(out_features, generator=g)

        x_full = x_full.to(device)
        w_full = w_full.to(device)
        b_full = b_full.to(device)

        # 基准：单 GPU 完整计算
        single_layer = ReplicatedLinear(in_features, out_features, bias=True).to(device)
        single_layer.weight.weight_loader(single_layer.weight, w_full)
        single_layer.bias.weight_loader(single_layer.bias, b_full)
        y_single = single_layer(x_full)

        # 张量并行版本：每个 GPU 存储 out_features/tp_size 的输出维度
        col_tp_layer = ColumnParallelLinear(in_features, out_features, bias=True).to(device)
        col_tp_layer.weight.weight_loader(col_tp_layer.weight, w_full)
        col_tp_layer.bias.weight_loader(col_tp_layer.bias, b_full)

        # 前向传播：每个 GPU 计算部分输出
        y_col_tp = col_tp_layer(x_full)  # [batch, out_features/tp_size]

        # 恢复完整输出：All-Gather + 拼接
        y_parts = [torch.empty_like(y_col_tp) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_col_tp)
        y_full = torch.cat(y_parts, dim=-1)  # [batch, out_features]

        # 对齐检查（仅在 rank0 打印）
        max_err = (y_full - y_single).abs().max().item()
        ok = torch.allclose(y_full, y_single, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[ColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # 测试 2：合并列并行线性层
    @torch.no_grad()
    def test_merged_column_parallel(device):
        """
        测试 MergedColumnParallelLinear 的正确性

        场景：模拟 QKV 投影的合并
        - 原始：3 个独立的线性层（Q、K、V）
        - 合并：1 个大线性层，输出维度为 Q+K+V
        - 验证：合并版本的输出与独立计算后拼接的结果一致
        """
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        # 维度自动适配 tp_size
        in_features = 1024 * tp_size
        out_each = 512 * tp_size
        out_sizes = [out_each, out_each, out_each]  # Q、K、V 三个矩阵
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_q = torch.randn(out_sizes[0], in_features, generator=g)
        w_k = torch.randn(out_sizes[1], in_features, generator=g)
        w_v = torch.randn(out_sizes[2], in_features, generator=g)

        x_full = x_full.to(device)
        w_q = w_q.to(device)
        w_k = w_k.to(device)
        w_v = w_v.to(device)

        # 基准：单独计算 Q、K、V 后拼接
        y_ref = torch.cat(
            [
                nn.functional.linear(x_full, w_q, None),
                nn.functional.linear(x_full, w_k, None),
                nn.functional.linear(x_full, w_v, None),
            ],
            dim=-1,
        )

        # 张量并行版本：合并后切分
        # 注意：MergedColumnParallelLinear 的 weight_loader 签名与基类不同，因此 bias=False
        merged = MergedColumnParallelLinear(in_features, out_sizes, bias=False).to(device)
        merged.weight_loader(merged.weight, w_q, 0)
        merged.weight_loader(merged.weight, w_k, 1)
        merged.weight_loader(merged.weight, w_v, 2)

        y_local = merged(x_full)  # [batch, sum(out_sizes)/tp]，布局：[q_local, k_local, v_local]

        # All-Gather 后重新打包成 [Q_all | K_all | V_all]
        y_parts = [torch.empty_like(y_local) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_local)

        # 计算每个 GPU 的局部输出大小
        ql = out_sizes[0] // tp_size
        kl = out_sizes[1] // tp_size
        vl = out_sizes[2] // tp_size

        # 从各 GPU 的局部输出中提取并拼接完整的 Q、K、V
        q_full = torch.cat([p[:, :ql] for p in y_parts], dim=-1)
        k_full = torch.cat([p[:, ql : ql + kl] for p in y_parts], dim=-1)
        v_full = torch.cat([p[:, ql + kl : ql + kl + vl] for p in y_parts], dim=-1)
        y_full = torch.cat([q_full, k_full, v_full], dim=-1)

        max_err = (y_full - y_ref).abs().max().item()
        ok = torch.allclose(y_full, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[MergedColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # 测试 3：QKV 列并行线性层
    @torch.no_grad()
    def test_qkv_column_parallel(device):
        """
        测试 QKVColumnParallelLinear 的正确性

        特点：
        - 支持 Grouped-Query Attention (GQA)
        - Q 头数可能大于 KV 头数
        - 自动处理头数的分配和权重切分
        """
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        # 维度设置（确保可被 tp_size 整除）
        input_size = 1024 * tp_size
        head_size = 16
        num_heads = 4 * tp_size       # Q 头数
        num_kv_heads = 2 * tp_size    # KV 头数（GQA：少于 Q 头数）
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, input_size, generator=g)
        w_q = torch.randn(head_size * num_heads, input_size, generator=g)
        w_k = torch.randn(head_size * num_kv_heads, input_size, generator=g)
        w_v = torch.randn(head_size * num_kv_heads, input_size, generator=g)

        x_full = x_full.to(device)
        w_q = w_q.to(device)
        w_k = w_k.to(device)
        w_v = w_v.to(device)

        # 基准：完整的 Q|K|V 拼接
        y_ref = torch.cat(
            [
                nn.functional.linear(x_full, w_q, None),
                nn.functional.linear(x_full, w_k, None),
                nn.functional.linear(x_full, w_v, None),
            ],
            dim=-1,
        )

        # 张量并行版本：每个 GPU 的输出布局为 [q_local | k_local | v_local]
        qkv = QKVColumnParallelLinear(
            input_size=input_size,
            head_size=head_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=False,
        ).to(device)

        qkv.weight_loader(qkv.weight, w_q, "q")
        qkv.weight_loader(qkv.weight, w_k, "k")
        qkv.weight_loader(qkv.weight, w_v, "v")

        y_local = qkv(x_full)  # [batch, head_size*(local_h + 2*local_kv)]

        # All-Gather 后重新打包成 [Q_all | K_all | V_all]
        y_parts = [torch.empty_like(y_local) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_local)

        # 计算每个 GPU 的局部头维度
        ql = head_size * (num_heads // tp_size)
        kl = head_size * (num_kv_heads // tp_size)
        vl = head_size * (num_kv_heads // tp_size)

        # 从各 GPU 的局部输出中提取并拼接完整的 Q、K、V
        q_full = torch.cat([p[:, :ql] for p in y_parts], dim=-1)
        k_full = torch.cat([p[:, ql : ql + kl] for p in y_parts], dim=-1)
        v_full = torch.cat([p[:, ql + kl : ql + kl + vl] for p in y_parts], dim=-1)
        y_full = torch.cat([q_full, k_full, v_full], dim=-1)

        max_err = (y_full - y_ref).abs().max().item()
        ok = torch.allclose(y_full, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[QKVColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # 测试 4：行并行线性层
    @torch.no_grad()
    def test_row_parallel(device):
        """
        测试 RowParallelLinear 的正确性

        验证策略：
        1. 单 GPU 基准：使用完整权重和完整输入计算
        2. 多 GPU 并行：
           - 每个 GPU 存储部分权重（按输入维度切分）
           - 每个 GPU 接收部分输入（与权重切分对应）
           - 每个 GPU 计算部分乘积
           - All-Reduce 汇总所有 GPU 的结果
        3. 对比结果是否一致

        注意：
        - 输入 x_part 是手动切分的，模拟前面 ColumnParallel 的输出
        - 偏置需要在所有 GPU 间均分（除以 tp_size），避免重复累加
        """
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        in_features = 128 * tp_size
        out_features = 256
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_full = torch.randn(out_features, in_features, generator=g)
        b_full = torch.randn(out_features, generator=g)

        x_full = x_full.to(device)
        w_full = w_full.to(device)
        b_full = b_full.to(device)

        # 基准：单 GPU 完整计算
        single = ReplicatedLinear(in_features, out_features, bias=True).to(device)
        single.weight.weight_loader(single.weight, w_full)
        single.bias.weight_loader(single.bias, b_full)
        y_ref = single(x_full)

        # 张量并行版本：行切分
        row_tp = RowParallelLinear(in_features, out_features, bias=True).to(device)
        row_tp.weight.weight_loader(row_tp.weight, w_full)

        # 偏置需要均分，因为 All-Reduce 会累加所有 GPU 的结果
        if row_tp.bias is not None:
            row_tp.bias.data.copy_(b_full / tp_size)

        # 手动切分输入（模拟前面 ColumnParallel 的输出）
        shard = in_features // tp_size
        start = tp_rank * shard
        x_part = x_full.narrow(-1, start, shard)

        # 前向传播：计算部分乘积 + All-Reduce
        y_row = row_tp(x_part)

        max_err = (y_row - y_ref).abs().max().item()
        ok = torch.allclose(y_row, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[RowParallel] allclose={ok}, max_abs_err={max_err:.6f}")

    # 主程序入口
    rank, world_size, local_rank, device = _init_dist()
    if rank == 0:
        print(f"运行张量并行测试：world_size={world_size}, device={device}")
        print("=" * 60)

    # 运行所有测试（输出 'allclose=True' 表示测试通过）
    test_column_parallel(device)
    test_merged_column_parallel(device)
    test_qkv_column_parallel(device)
    test_row_parallel(device)

    # 同步所有进程并清理
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print("=" * 60)
        print("所有测试完成！")