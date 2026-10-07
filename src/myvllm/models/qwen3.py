"""
Qwen3 模型实现

该模块实现了 Qwen3 架构的完整模型，包括注意力机制、前馈网络、解码器层和语言模型头。
支持张量并行（Tensor Parallelism）和分组查询注意力（Grouped Query Attention, GQA）。

主要组件：
- Qwen3Attention: 多头注意力机制，支持 GQA 和旋转位置编码（RoPE）
- Qwen3MLP: 前馈神经网络，使用 SwiGLU 激活函数
- Qwen3DecoderLayer: Transformer 解码器层
- Qwen3Model: 完整的 Qwen3 模型
- Qwen3ForCausalLM: 用于因果语言建模的 Qwen3 模型
"""

from myvllm.layers import *
import torch
import torch.nn as nn


class Qwen3Attention(nn.Module):
    """
    Qwen3 多头注意力机制

    实现了以下特性：
    1. 分组查询注意力（GQA）：支持 num_kv_heads < num_heads
    2. 张量并行：跨多个 GPU 分片注意力头
    3. Q/K 归一化：当不使用 QKV bias 时，对 Q 和 K 应用 LayerNorm 以稳定训练
    4. 旋转位置编码（RoPE）：为 Q 和 K 添加位置信息

    数据流：
    输入 -> QKV 投影（列并行）-> 分割 Q/K/V -> Q/K 归一化 -> RoPE -> 注意力 -> 输出投影（行并行）

    Args:
        hidden_size: 隐藏层维度
        num_heads: 总的查询头数量（会被 tensor parallel size 整除）
        head_dim: 每个注意力头的维度
        scale: 注意力缩放因子（默认为 1.0，实际缩放为 1/sqrt(head_dim)）
        num_kv_heads: KV 头数量，用于 GQA（None 表示与 num_heads 相同）
        rms_norm_epsilon: RMS 归一化的 epsilon 值
        qkv_bias: 是否在 QKV 投影中使用 bias
        base: RoPE 的基础频率
        max_position: 最大位置编码长度
        block_size: KV 缓存的块大小
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        block_size: int = 256,
    ):
        super().__init__()
        # 获取张量并行的 GPU 数量
        self.tp_size = dist.get_world_size()

        # 总的注意力头数和每个 GPU 上的注意力头数
        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size  # 每个 GPU 负责的查询头数

        # 总的 KV 头数和每个 GPU 上的 KV 头数（用于 GQA）
        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size  # 每个 GPU 负责的 KV 头数

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = scale

        # QKV 投影：列并行，将注意力头分片到不同的 GPU
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 每个 GPU 上的 Q 和 KV 张量大小
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        # Q 和 K 归一化：Qwen3 特有，用于稳定注意力计算
        # 防止 Q·K^T 产生过大的值导致 softmax 不稳定
        self.q_norm = LayerNorm(torch.ones(head_dim))
        self.k_norm = LayerNorm(torch.ones(head_dim))

        # 旋转位置编码（RoPE）
        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position
        )

        # 注意力计算模块（支持 flash attention 和 paged attention）
        self.attention = Attention(
            self.num_heads,
            head_dim,
            scale,
            self.num_kv_heads,
            block_size
        )

        # 输出投影：行并行，执行 all-reduce 以聚合所有 GPU 的结果
        self.o_proj = RowParallelLinear(
            input_size=head_dim * self.total_num_heads,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (B, N, hidden_size) 或 (total_tokens, hidden_size)
               在所有 GPU 上复制
            positions: 位置索引，用于 RoPE

        Returns:
            输出张量，形状与输入相同，在所有 GPU 上复制（经过 all-reduce）

        张量并行流程：
        1. QKV 投影（列并行）：在这里进行分片，每个 GPU 得到不同的注意力头
        2. 注意力计算：每个 GPU 独立计算自己的注意力头
        3. 输出投影（行并行）：通过 all-reduce 聚合所有 GPU 的结果
        """
        # 输入：x 形状 (B, N, hidden_size)，在所有 GPU 上复制

        # QKV 投影（列并行 - 这里发生分片）
        # 每个 GPU 的输出形状：(B, N, head_dim * (num_heads + 2*num_kv_heads))
        # 其中 num_heads = total_num_heads / tp_size
        #      num_kv_heads = total_num_kv_heads / tp_size
        qkv = self.qkv_projection(x)

        # 分割 QKV
        # q_size = head_dim * num_heads     - 每个 GPU 的大小
        # kv_size = head_dim * num_kv_heads - 每个 GPU 的大小
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 处理批处理（3D）和变长（2D）两种输入格式
        # 变长模式：q 形状 (total_tokens, q_size)，其中 q_size = num_heads * head_dim
        # 批处理模式：q 形状 (B, N, q_size)
        if q.dim() == 2:
            # 变长模式：(total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            # 批处理模式：(B, N, q_size) -> (B, N, num_heads, head_dim)
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # 应用 Q 和 K 归一化（Qwen3 特性）
        # 用于稳定注意力计算，防止 Q·K^T 产生过大的值导致 softmax 不稳定
        # 只在不使用 QKV bias 时应用
        if self.qkv_bias is False:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)

        # 注意力计算
        o = self.attention(q, k, v)
        # o 形状：(B*N, num_heads, head_dim) - 每个 GPU 上的不同注意力头

        # 输出投影（行并行 - 这里通过 all_reduce 进行通信）
        o = self.o_proj(o)
        # 输入：(B*N, num_heads * head_dim)，在 GPU 间分片
        # 输出：(B*N, hidden_size)，在所有 GPU 上复制（all-reduce 后）

        return o


class Qwen3MLP(nn.Module):
    """
    Qwen3 前馈神经网络（MLP）

    使用 SwiGLU 激活函数的前馈网络：
    - gate 分支：用于门控
    - up 分支：用于特征变换
    - 激活函数：SiLU(gate) * up
    - down 投影：将维度映射回 hidden_size

    数据流：
    输入 -> gate_up 投影（列并行）-> SiLU 激活 + 门控乘法 -> down 投影（行并行）

    Args:
        hidden_size: 隐藏层维度
        intermediate_size: 中间层维度（通常是 hidden_size 的 4 倍）
        bias: 是否在线性层中使用 bias
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        # gate 和 up 投影合并为一个矩阵，提高效率
        # 输出被分成两部分：[gate, up]
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        # SwiGLU 激活：SiLU(gate) * up
        self.activation = SiluAndMul()
        # down 投影：将中间维度映射回隐藏维度
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (B, N, hidden_size)

        Returns:
            输出张量，形状为 (B, N, hidden_size)
        """
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 解码器层

    标准的 Transformer 解码器层，包含：
    1. 输入层归一化 + 残差连接
    2. 自注意力机制
    3. 注意力后层归一化 + 残差连接
    4. 前馈神经网络（MLP）

    采用 Pre-LN 架构：LayerNorm 在子层之前应用
    残差连接：使用融合的 LayerNorm + 残差加法以提高效率

    数据流：
    输入 -> [LayerNorm + 残差] -> 自注意力 -> [LayerNorm + 残差] -> MLP -> 输出

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数量
        head_dim: 每个注意力头的维度
        scale: 注意力缩放因子
        num_kv_heads: KV 头数量（用于 GQA）
        rms_norm_epsilon: RMS 归一化的 epsilon 值
        qkv_bias: 是否在 QKV 投影中使用 bias
        base: RoPE 的基础频率
        max_position: 最大位置编码长度
        intermediate_size: MLP 中间层维度
        ffn_bias: 是否在 FFN 中使用 bias
        block_size: KV 缓存的块大小
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        block_size: int = 256,
    ):
        super().__init__()
        # 初始化归一化层的权重
        gamma = torch.ones(hidden_size)
        # 输入层归一化
        self.input_layernorm = LayerNorm(gamma)
        # 自注意力机制
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            block_size=block_size,
        )
        # 注意力后层归一化
        self.post_attention_layernorm = LayerNorm(gamma)
        # 前馈神经网络
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (B, N, hidden_size) 或 (total_tokens, hidden_size)
            residual: 残差张量（可选），用于融合的 LayerNorm + 残差加法

        Returns:
            tuple: (输出张量, 新的残差张量)

        残差连接机制：
        - 如果提供了 residual，则执行融合的 residual + LayerNorm(x)
        - 否则，保存当前 x 作为残差，然后应用 LayerNorm
        - 这种设计允许将残差加法和 LayerNorm 融合为单个 kernel，提高效率
        """
        # 输入层归一化 + 残差加法
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            residual = x  # 在归一化之前保存残差
            x = self.input_layernorm(x)

        # 计算位置索引（根据上下文推断，支持批处理 prefill 的序列边界）
        from myvllm.utils import get_context
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            # 批处理 prefill：为每个序列创建从 0 开始的位置索引
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                seq_len = cu_seqlens[i+1] - cu_seqlens[i]
                positions.extend(range(seq_len))
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        elif context.is_prefill:
            # 单序列 prefill：使用顺序位置索引
            positions = torch.arange(x.size(0), device=x.device)
        else:
            # 解码阶段：使用 context_lens - 1 作为位置（每个序列的当前位置）
            positions = context.context_lens - 1

        # 自注意力
        x = self.self_attn(x, positions=positions)

        # 注意力后层归一化 + 残差加法（总是启用）
        x, residual = self.post_attention_layernorm(x, residual)

        # 前馈神经网络
        x = self.mlp(x)

        return x, residual

class Qwen3Model(nn.Module):
    """
    Qwen3 基础模型

    完整的 Transformer 解码器模型，包含：
    1. 词嵌入层（支持张量并行）
    2. 多层 Transformer 解码器层堆叠
    3. 最终层归一化

    数据流：
    token_ids -> 嵌入 -> 解码器层 x N -> 最终 LayerNorm -> 隐藏状态

    Args:
        vocab_size: 词汇表大小
        hidden_size: 隐藏层维度
        num_heads: 注意力头数量
        head_dim: 每个注意力头的维度
        scale: 注意力缩放因子
        num_kv_heads: KV 头数量（用于 GQA）
        rms_norm_epsilon: RMS 归一化的 epsilon 值
        qkv_bias: 是否在 QKV 投影中使用 bias
        base: RoPE 的基础频率
        max_position: 最大位置编码长度
        intermediate_size: MLP 中间层维度
        ffn_bias: 是否在 FFN 中使用 bias
        num_layers: 解码器层数量
        block_size: KV 缓存的块大小
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        block_size: int = 256,
    ):
        super().__init__()
        # 词嵌入层（支持张量并行，词汇表在 GPU 间分片）
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim = hidden_size
        )
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_epsilon=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
            ) for _ in range(num_layers)
        ])
        # 最终层归一化
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入 token ID 张量，形状为 (B, N) 或 (total_tokens,)

        Returns:
            隐藏状态张量，形状为 (B, N, hidden_size) 或 (total_tokens, hidden_size)
        """
        # 词嵌入
        x = self.embed_tokens(input_ids)
        # 初始化残差为 None，第一层会保存 x 作为残差
        residual = None
        # 通过所有解码器层
        for layer in self.layers:
            # decoder堆叠
            x, residual = layer(x, residual)
        # 最终层归一化 + 残差加法
        x, _ = self.norm(x, residual)
        return x



class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型

    在 Qwen3Model 基础上添加语言模型头，用于生成下一个 token 的概率分布。

    组件：
    1. Qwen3Model：生成隐藏状态
    2. lm_head：将隐藏状态映射到词汇表 logits

    可选功能：
    - 权重共享（tie_word_embeddings）：lm_head 和嵌入层共享权重

    Args:
        vocab_size: 词汇表大小
        hidden_size: 隐藏层维度
        num_heads: 注意力头数量
        head_dim: 每个注意力头的维度
        scale: 注意力缩放因子
        num_kv_heads: KV 头数量（用于 GQA）
        rms_norm_epsilon: RMS 归一化的 epsilon 值
        qkv_bias: 是否在 QKV 投影中使用 bias
        base: RoPE 的基础频率
        max_position: 最大位置编码长度
        intermediate_size: MLP 中间层维度
        ffn_bias: 是否在 FFN 中使用 bias
        num_layers: 解码器层数量
        tie_word_embeddings: 是否共享嵌入层和 lm_head 的权重
        block_size: KV 缓存的块大小

    注意：
    - packed_module_mapping 用于映射合并的权重名称，实际加载逻辑在 loader.py 中通过正则匹配实现
    """
    # 权重名称映射（用于从 HuggingFace 格式加载合并的权重）
    # 实际加载由 loader.py 的正则匹配逻辑处理，此字典在代码中未被直接引用
    packed_module_mapping = {
        "q_proj": ('q_proj', 'q'),
        "k_proj": ('k_proj', 'k'),
        "v_proj": ('v_proj', 'v'),
        "gate_up": ('gate_up_proj', '0'),
        "gate_down": ('gate_down_proj', '1'),
    }
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False,
        block_size: int = 256,
    ):
        super().__init__()
        # 如果未指定 head_dim，则根据 hidden_size 和 num_heads 计算
        head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        # 基础 Transformer 模型
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size,
        )
        # 语言模型头（将隐藏状态映射到词汇表 logits）
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        # 权重共享：lm_head 和嵌入层使用相同的权重矩阵
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播（仅返回隐藏状态）

        Args:
            input_ids: 输入 token ID 张量，形状为 (B, N) 或 (total_tokens,)

        Returns:
            隐藏状态张量，形状为 (B, N, hidden_size) 或 (total_tokens, hidden_size)

        注意：此方法仅返回隐藏状态，不计算 logits。
        使用 compute_logits() 方法来获取词汇表上的概率分布。
        """
        x = self.model(input_ids)
        return x

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        从隐藏状态计算 logits

        Args:
            hidden_states: 隐藏状态张量，形状为 (B, N, hidden_size) 或 (total_tokens, hidden_size)

        Returns:
            logits 张量，形状为 (B, N, vocab_size) 或 (total_tokens, vocab_size)

        此方法将隐藏状态投影到词汇表空间，用于：
        - 生成下一个 token 的概率分布
        - 计算语言建模损失
        """
        logits = self.lm_head(hidden_states)
        return logits

if __name__ == "__main__":
    """
    测试代码

    创建一个小型 Qwen3 模型并进行前向传播测试。

    模型配置：
    - vocab_size: 50257 (GPT-2 词汇表大小)
    - hidden_size: 768
    - num_heads: 12
    - head_dim: 64
    - intermediate_size: 3072 (4 * hidden_size)
    - num_layers: 2 (仅用于测试)

    输入：随机生成的 token IDs，形状为 (2, 16)
    """
    model = Qwen3ForCausalLM(
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        num_layers=2,
    )
    input_ids = torch.randint(0, 50257, (2, 16)).cuda()
    output = model(input_ids)