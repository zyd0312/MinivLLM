import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    SiLU 门控激活层（SwiGLU 变体）

    这是大模型（如 LLaMA、Qwen）中常用的激活函数，结合了：
    1. SiLU 激活：平滑的非线性变换
    2. 门控机制：用一半特征控制另一半特征的通过

    相比简单的 ReLU，这种激活函数表达能力更强，训练更稳定。
    """

    def __init__(self):
        super().__init__()

    @torch.compile  # PyTorch 2.0+ 的 JIT 编译器，自动优化成高效 CUDA 代码
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (..., 2*hidden_dim)
               注意：最后一维的大小必须是偶数

        Returns:
            输出张量，形状 (..., hidden_dim)

        计算过程：
            1. 将输入沿最后一维切分成两半：x 和 y
            2. 对 x 应用 SiLU 激活：silu(x) = x * sigmoid(x)
            3. 用激活后的 x 作为"门"，控制 y 的通过：silu(x) * y

        例子：
            输入 shape: (batch=2, seq=10, dim=4096)
            chunk 后:
                x shape: (2, 10, 2048)  # 前一半
                y shape: (2, 10, 2048)  # 后一半
            输出 shape: (2, 10, 2048)
        """
        # 沿最后一维（-1）切分成两块，每块大小相等
        # 切分是由于之前将多个列并行层合并（例如 gate + up 两个投影）
        x, y = x.chunk(2, -1)

        # SiLU(x) = x * sigmoid(x)，然后用它作为门控信号，逐元素乘以 y
        # 这种门控机制让模型学会"哪些信息该通过，哪些该抑制"
        return F.silu(x) * y

if __name__ == "__main__":
    """
    A/B 对比：@torch.compile 相对 eager 模式快多少

    SiluAndMul 是 memory-bound 算子（每读一个元素只做几次浮点运算），
    compile 的收益来自把 chunk / silu / mul 融合成一个 kernel，减少显存往返，
    而不是减少计算量。所以除了耗时，也要看有效带宽和显存峰值。
    """
    import statistics

    class SiluAndMulEager(nn.Module):
        """对照组：forward 与 SiluAndMul 逐字相同，只是没有 @torch.compile。"""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x, y = x.chunk(2, -1)
            return F.silu(x) * y

    def timeit(layer, x, n_warmup: int = 20, n_iters: int = 100):
        """
        用 CUDA event 计时。比 time.time() + synchronize 更准：
        时间戳由 GPU 自己打，不混入 Python 侧的调度抖动。

        Returns:
            (每次迭代的毫秒数列表, 显存分配峰值 GB)
        """
        # 预热：触发编译、kernel 加载、显存分配器 warm up
        for _ in range(n_warmup):
            layer(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(n_iters):
            start.record()
            layer(x)
            end.record()
            end.synchronize()  # 只等这一次迭代
            times.append(start.elapsed_time(end))

        return times, torch.cuda.max_memory_allocated() / 1024**3

    compiled = SiluAndMul().cuda()
    eager = SiluAndMulEager().cuda()

    # (batch, seq_len, 2*hidden) —— 最后一维必须是偶数
    shape = (8, 4000, 8000)

    with torch.inference_mode():
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(*shape, dtype=dtype, device="cuda")

            # 首次调用包含 dynamo 编译开销。换 dtype 会让 guard 失效，
            # 触发重新编译 —— 所以这里每个 dtype 都会重新计一次。
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            compiled(x)
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - t0) * 1000

            # 数值一致性：融合后的 kernel 结果应与 eager 相符
            max_diff = (compiled(x) - eager(x)).abs().max().item()

            c_times, c_mem = timeit(compiled, x)
            e_times, e_mem = timeit(eager, x)

            c_med, e_med = statistics.median(c_times), statistics.median(e_times)

            # 有效带宽：读入整个 x，写出一半大小的结果
            n_bytes = x.numel() * x.element_size() * 1.5
            to_gbps = lambda ms: n_bytes / (ms / 1000) / 1024**3

            print(f"\n{'='*62}")
            print(f"{tuple(x.shape)}  {dtype}  ({x.numel() * x.element_size() / 1024**3:.2f} GB 输入)")
            print(f"{'='*62}")
            print(f"{'':12} {'中位数':>10} {'均值':>10} {'带宽':>12} {'显存峰值':>11}")
            print(f"{'eager':12} {e_med:9.3f}ms {statistics.mean(e_times):9.3f}ms "
                  f"{to_gbps(e_med):8.0f}GB/s {e_mem:9.2f}GB")
            print(f"{'compiled':12} {c_med:9.3f}ms {statistics.mean(c_times):9.3f}ms "
                  f"{to_gbps(c_med):8.0f}GB/s {c_mem:9.2f}GB")
            print(f"\n加速比    {e_med / c_med:.2f}x"
                  f"    显存省  {(1 - c_mem / e_mem) * 100:.0f}%"
                  f"    首次编译  {compile_ms:.0f}ms"
                  f"    最大误差  {max_diff:.2e}")

            del x
