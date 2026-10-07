"""
旋转位置编码（Rotary Position Embedding, RoPE）实现

RoPE是一种相对位置编码方法，通过旋转变换将位置信息注入到query和key中。
相比传统的绝对位置编码，RoPE能够更好地捕捉相对位置关系，并且支持外推到更长的序列。

核心思想：
1. 将注意力头的维度两两配对，每对维度形成一个二维平面
2. 在每个平面上应用旋转变换，旋转角度由位置索引和频率决定
3. 这样可以使得query和key之间的内积只依赖于它们的相对位置差

数学公式：
对于位置m的向量x，旋转后的结果为：
[x1'] = [cos(mθ)  -sin(mθ)] [x1]
[x2']   [sin(mθ)   cos(mθ)] [x2]

其中θ = base^(-2i/d)，i是维度索引，d是旋转维度，base通常为10000
"""

import torch.nn as nn
import torch


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量

    Args:
        x: 输入张量，形状为 (total_tokens, num_heads, head_dim) 或 (B, seq_len, num_heads, head_dim)
        cos: 余弦值，形状为 (seq_len, head_dim/2)
        sin: 正弦值，形状为 (seq_len, head_dim/2)

    Returns:
        应用旋转编码后的张量，形状与输入x相同

    实现原理：
        将head_dim分成两半[x1, x2]，然后应用旋转矩阵：
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        最后拼接 [out1, out2]
    """
    # 处理两种输入格式：
    # 1. 3D变长模式：(total_tokens, num_heads, head_dim) - 用于变长序列批处理
    #    - 使用场景：推理阶段的高效批处理，特别是在vLLM等推理引擎中
    #    - 优势：将不同长度的序列打包成一维，避免padding浪费
    #    - 示例：3个序列长度分别为[5,3,7]，打包后total_tokens=15
    # 2. 4D批次模式：(B, seq_len, num_heads, head_dim) - 用于标准批处理
    #    - 使用场景：训练阶段或简单推理场景
    #    - 特点：每个序列有固定长度（短序列需padding到统一长度）
    #    - 示例：批次大小B=4，每个序列长度seq_len=128
    if x.dim() == 3:
        # 变长模式：(total_tokens, num_heads, head_dim)
        total_tokens, num_heads, head_dim = x.shape
        # cos, sin 原始形状: (total_tokens, head_dim/2)
        # 扩展为 (total_tokens, 1, head_dim/2) 以便广播到所有注意力头
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 将 x 沿着头维度分成两半：[x1, x2]
        # 每一半的形状: (total_tokens, num_heads, head_dim/2)
        x1, x2 = x.chunk(2, dim=-1)

        # 应用旋转嵌入的核心公式：
        # out1 = x1 * cos(θ) - x2 * sin(θ)  （旋转矩阵的第一行）
        # out2 = x1 * sin(θ) + x2 * cos(θ)  （旋转矩阵的第二行）
        # x1, x2 形状: (total_tokens, num_heads, head_dim/2)
        # cos, sin 形状: (total_tokens, 1, head_dim/2)
        # 广播后相乘，得到旋转后的结果
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # 拼接两部分，恢复原始维度
        return torch.cat([out1, out2], dim=-1)
    else:
        # 批次模式：(B, seq_len, num_heads, head_dim)
        B = x.size(0)
        seq_len = x.size(1)
        num_heads = x.size(2)
        head_dim = x.size(-1)

        # 扩展 cos 和 sin 以匹配批次和注意力头维度
        # cos, sin 原始形状: (seq_len, head_dim/2)
        # 扩展为: (1, seq_len, 1, head_dim/2) 以便广播到所有批次和注意力头
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # 将 x 沿着头维度分成两半
        x1, x2 = x.chunk(2, dim=-1)

        # 应用旋转嵌入，利用广播机制
        # x1, x2 形状: (B, seq_len, num_heads, head_dim/2)
        # cos, sin 形状: (1, seq_len, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # 拼接两部分，恢复原始维度
        return torch.cat([out1, out2], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块

    该模块预计算并缓存所有位置的cos和sin值，在forward时根据位置索引取用。
    支持标准RoPE和Llama3改进版本（带有频率缩放）。

    参数说明：
        base: 频率基数，通常为10000，控制不同维度的旋转频率
        rotary_embedding: 应用旋转编码的维度数（通常是head_dim的一部分或全部）
        max_position: 支持的最大位置，用于预计算缓存
        is_llama3: 是否使用Llama3的改进版RoPE（支持更长上下文）
        llama3_rope_factor: Llama3的缩放因子，用于降低高频部分
        llama3_rope_high_freq_factor: 高频阈值因子
        llama3_rope_low_freq_factor: 低频阈值因子
        llama3_rope_original_max_position_embeddings: Llama3原始训练时的最大位置
    """
    def __init__(
        self,
        base:int,
        rotary_embedding: int,
        max_position: int = 2048,
        is_llama3: bool = False,
        # 以下参数仅用于 Llama3.2
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.base = base
        # 应用旋转编码的维度数
        self.rotary_embedding = rotary_embedding
        # 长上下文可以达到的最大位置
        self.max_position = max_position

        # 计算逆频率：inv_freq[i] = 1 / (base^(2i/d))
        # 其中 i ∈ [0, d/2)，d 是旋转维度
        # 这会生成一系列递减的频率，从高频到低频
        # 例如：base=10000, d=128 时，频率从 1.0 递减到 0.0001
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))

        if is_llama3:
            # Llama3.2 专用的频率调整策略
            # 目的：通过缩放频率来支持更长的上下文，同时保持模型性能
            import math
            inv_freq = self.inv_freq

            # 计算每个频率对应的波长：wave_len = 2π / freq
            # 波长越长，表示该维度捕捉的是更长距离的位置关系
            wave_len = 2 * math.pi / inv_freq

            # 如果低频因子等于高频因子，使用简单的二分策略
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                # 对于波长小于阈值的高频部分，保持原频率不变
                # 对于波长大于阈值的低频部分，缩小频率（除以rope_factor）
                # 这样可以让低频部分适应更长的上下文
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                # 使用平滑过渡策略，避免频率突变
                # 计算平滑系数：从低频到高频线性过渡
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                # 限制在 [0, 1] 范围内
                smooth = torch.clamp(smooth, 0, 1)
                # 计算缩放因子：smooth=0时完全缩放，smooth=1时不缩放
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            self.inv_freq = inv_freq

        # 预计算所有位置的频率值
        # positions: [0, 1, 2, ..., max_position-1]
        positions = torch.arange(self.max_position).float()

        # 使用爱因斯坦求和约定计算：freqs[i,j] = positions[i] * inv_freq[j]
        # 结果形状: (max_position, rotary_embedding/2)
        # 每一行代表一个位置，每一列代表一个频率维度
        # freqs[m,i] = m * θ_i，其中 m 是位置，θ_i 是第 i 个频率
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)

        # 计算余弦和正弦值
        # 这些值将在forward时直接使用，避免重复计算
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # 拼接 cos 和 sin，形状: (max_position, rotary_embedding)
        # 前半部分是cos，后半部分是sin
        cos_sin_cache = torch.cat([cos, sin], dim=-1)

        # 注册为buffer，这样模型保存时会包含这个缓存
        # 但不会被视为模型参数，不参与梯度更新
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    def forward(self, positions, query, key):
        """
        对query和key应用旋转位置编码

        Args:
            positions: 位置索引张量，形状为 (seq_len,) 或 (total_tokens,)
                      指示每个token在序列中的位置
            query: 查询张量，形状为 (total_tokens, num_heads, head_dim) 或 (B, seq_len, num_heads, head_dim)
            key: 键张量，形状与query相同

        Returns:
            (rotated_query, rotated_key): 应用旋转编码后的query和key元组

        工作流程：
            1. 根据positions从预计算的缓存中取出对应的cos和sin值
            2. 将cos和sin分离
            3. 分别对query和key应用旋转变换
            4. 返回旋转后的结果

        注意：使用@torch.compile装饰器进行JIT编译以提升性能
        """
        # 从缓存中根据位置索引取出cos和sin值
        # cos_sin形状: (seq_len, rotary_embedding)
        cos_sin = self.cos_sin_cache[positions]

        # 分离cos和sin，每个形状: (seq_len, rotary_embedding/2)
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 对query和key分别应用旋转位置编码
        # 返回旋转后的(query, key)元组
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


if __name__ == "__main__":
    """
    测试代码：演示RoPE的频率计算过程

    示例参数：
        base = 5: 频率基数（实际应用中通常为10000）
        rotary_dim = 16: 旋转编码的维度
        max_position = 100: 最大位置数

    输出说明：
        1. 维度索引: [0, 2, 4, 6, 8, 10, 12, 14] （偶数维度）
        2. 基数的幂次: base^(2i/d)，随维度增加而增大
        3. 逆频率: 1 / base^(2i/d)，随维度增加而减小
        4. 频率矩阵: (max_position, rotary_dim/2)
        5. 第3个位置的频率值（展示特定位置的频率分布）
    """
    base = 5
    # 应用旋转编码的维度数
    rotary_dim = 16
    # 长上下文可以达到的最大位置
    max_position = 100

    # 打印偶数维度索引: [0, 2, 4, ..., 14]
    print(torch.arange(0, rotary_dim, 2))

    # 打印基数的幂次: [base^0, base^(2/16), base^(4/16), ..., base^(14/16)]
    # 这些值随维度增加而增大
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))

    # 计算逆频率: 1 / base^(2i/d)
    # 这些值随维度增加而减小，意味着不同维度捕捉不同尺度的位置关系
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    print(inv_freq)

    # 生成位置索引: [0, 1, 2, ..., 99]
    t = torch.arange(max_position).float()

    # 计算频率矩阵: freqs[i,j] = position[i] * inv_freq[j]
    # 结果形状: (100, 8)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)

    print(freqs.size())

    # 打印第3个位置（索引2）的频率值
    # 这展示了该位置在不同频率维度上的值
    print(freqs[2])

