import torch
import time

class LayerNorm(torch.nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization) 实现

    RMSNorm 是一种简化的归一化方法，相比标准 LayerNorm 省略了均值中心化步骤，
    只使用均方根进行归一化，在保持性能的同时显著提升计算效率。

    公式: RMSNorm(x) = (x / RMS(x)) * γ
    其中: RMS(x) = sqrt(mean(x²) + ε)

    Args:
        gamma: 可学习的缩放参数（形状为 [hidden_size]）
        eps: 数值稳定性的小常数，防止除零（默认 1e-5）
    """
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # 将 gamma 转换为 nn.Parameter，使其可学习且能从 checkpoint 加载
        # 步骤 1: 切断计算图，假设 gamma 来自某个预训练模型，可能带有历史计算图
        temp1 = gamma.detach()  # 不再跟踪梯度，但共享数据

        # 步骤 2: 独立拷贝数据
        temp2 = temp1.clone()   # 创建独立副本

        # 步骤 3: 包装为可学习参数
        self.weight = torch.nn.Parameter(temp2)  # requires_grad=True（重新启用梯度）

        self.eps = eps

    @property
    def gamma(self):
        """向后兼容属性：gamma 是 weight 的别名"""
        return self.weight

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm 前向传播（经过 torch.compile 优化）

        Args:
            x: 输入张量，形状 [..., hidden_size]

        Returns:
            归一化后的张量，形状与输入相同
        """
        # 计算均方根：RMS(x) = sqrt(mean(x²) + ε)
        # 在最后一维（hidden_size 维度）上计算
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        sqrt_variance = variance.sqrt()
        # 归一化并应用可学习的缩放参数 γ
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        融合残差连接的 RMSNorm 前向传播

        将残差加法和归一化融合在一起，避免额外的内存写入，提升性能。
        常用于 Transformer 层中：LayerNorm(x + residual)

        Args:
            x: 当前层的输出
            residual: 残差连接的输入

        Returns:
            (归一化结果, 加法后的张量)
            返回两个值以便后续层可以继续使用残差连接
        """
        x = x + residual
        return self.rms_forward(x), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """
        统一的前向传播接口

        Args:
            x: 输入张量
            residual: 可选的残差张量，如果提供则执行融合的残差+归一化

        Returns:
            如果有 residual：返回 (归一化结果, 加法后的张量)
            如果无 residual：返回归一化结果
        """
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)

if __name__ == "__main__":
    """
    性能基准测试

    测试场景：
    - 输入形状：[batch=8, seq_len=4000, hidden_size=8000]
    - 对比两种模式：无残差连接 vs 有残差连接
    - 测试方法：预热10次 + 正式计时100次取平均
    """
    # 创建测试数据
    x = torch.randn(8, 4000, 8000).cuda()  # 模拟大语言模型的中间激活张量
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)  # 初始化缩放参数
    layer = LayerNorm(gamma=gamma).cuda()
    residual = torch.full_like(x, fill_value=1)  # 残差张量

    # 预热阶段：让 torch.compile 完成 JIT 编译，填充 CUDA kernel 缓存
    for _ in range(10):
        _ = layer(x)

    # ============ 测试 1: 无残差连接模式 ============
    times = []
    for _ in range(100):
        torch.cuda.synchronize()  # 等待 GPU 完成之前的所有操作
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()  # 确保 GPU 完成当前操作再记录时间
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    # ============ 测试 2: 有残差连接模式 ============
    times.clear()
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x, residual)  # 测试融合的残差+归一化路径
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

