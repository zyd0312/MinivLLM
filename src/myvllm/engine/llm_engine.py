"""
LLM推理引擎的核心模块

这个模块实现了一个简化版的vLLM推理引擎，主要功能包括：
1. 多GPU分布式推理支持（使用multiprocessing）
2. 批处理调度和连续批处理（continuous batching）
3. PagedAttention KV缓存管理
4. 预填充（prefill）和解码（decode）阶段分离
"""
import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """
    分布式推理的工作进程函数

    在多GPU推理时，每个非主进程（rank > 0）都会运行这个函数。
    工作进程的主要职责是：
    1. 初始化自己的ModelRunner实例
    2. 进入事件循环，等待主进程的指令
    3. 执行模型前向传播（与主进程同步）

    Args:
        config: 引擎配置字典，包含模型路径、批处理参数等
        rank: 当前进程的rank编号（在分布式环境中的序号）
        event: 用于进程间同步的Event对象

    注意：
        - 必须首先设置stdout/stderr的行缓冲，确保日志能及时输出
        - 这个函数会一直运行直到收到"exit"指令
    """
    # 设置行缓冲模式，确保print语句立即输出（对调试分布式代码很重要）
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    # 初始化模型运行器并进入事件循环
    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    """
    LLM推理引擎的主类

    这是整个推理系统的核心协调器，负责：
    1. 初始化分布式环境（如果world_size > 1）
    2. 管理模型运行器（ModelRunner）和调度器（Scheduler）
    3. 协调预填充和解码阶段的执行
    4. 处理用户请求并返回生成结果

    架构设计：
    - 主进程（rank 0）：运行调度器和用户接口，同时也参与模型推理
    - 工作进程（rank 1~N）：纯粹的模型推理worker，等待主进程指令
    - 使用torch.distributed进行GPU间通信

    关键组件：
    - ModelRunner: 负责实际的模型前向传播
    - Scheduler: 负责批处理调度、KV缓存分配、序列生命周期管理
    - Tokenizer: 负责文本和token之间的转换
    """
    def __init__(self, config: dict):
        """
        初始化LLM引擎

        初始化流程：
        1. 启动worker进程（如果是多GPU）
        2. 初始化主进程的ModelRunner（触发分布式初始化barrier）
        3. 初始化Scheduler（必须在barrier之后）
        4. 加载tokenizer
        5. 注册退出清理函数

        Args:
            config: 配置字典，包含以下关键参数：
                - world_size: GPU数量（默认1）
                - model_name_or_path: 模型路径
                - max_num_sequences: 最大并发序列数
                - max_num_batched_tokens: 单批次最大token数
                - max_cached_blocks: KV缓存块总数
                - block_size: 每个缓存块的token数
                - eos: EOS token id

        注意：
            Scheduler必须在ModelRunner之后初始化，因为ModelRunner.__init__
            会调用dist.init_process_group()进行分布式barrier同步。
        """
        self.config = config
        world_size = config.get("world_size", 1)

        """
        Python的 multiprocessing 模块支持三种不同的进程启动方式：

        1. spawn (这里使用的方式)
        工作原理：启动一个全新的Python解释器进程
        特点：
        子进程只继承运行 Process 对象的 run() 方法所必需的资源
        父进程中的资源不会被继承（除了通过参数显式传递的）
        最安全、最干净，但启动速度较慢
        兼容性：Windows、Linux、macOS 都支持
        为什么这里用它：
        PyTorch + CUDA 环境下最安全
        避免CUDA上下文在fork时的问题
        避免子进程意外继承父进程的GPU状态
        2. fork (Linux默认)
        工作原理：复制父进程的内存空间（写时复制）
        问题：
        CUDA上下文不能安全地fork
        可能导致死锁或CUDA错误
        不适合多线程环境

        3. forkserver
        工作原理：启动一个服务器进程，由它来fork新进程
        特点：介于spawn和fork之间
        """
        # 使用spawn上下文创建子进程（spawn是跨平台最安全的方式）
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []

        # 为每个worker进程创建独立的Event对象用于同步
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()

        # 初始化主进程（rank=0）的模型运行器
        # 如果world_size > 1，这里会阻塞直到所有worker进程完成分布式初始化
        self.model_runner = ModelRunner(config, rank=0, event=self.events)

        # 加载tokenizer（只在主进程需要，用于处理用户输入）
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))

        # 初始化调度器（必须在分布式barrier之后）
        # scheduler需要在model_runner之后初始化：当world_size > 1时，
        # ModelRunner.__init__调用dist.init_process_group()进行集体barrier —
        # rank-0在此阻塞直到所有worker rank加入。
        # scheduler应该只在rendezvous（会合）完成后创建。
        # 当world_size == 1时没有barrier，也就没有真正的依赖关系。
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256)
        )

        # 注册退出清理函数，确保进程正常退出
        # atexit 是 Python 标准库中的一个模块，用于注册程序正常退出时要执行的清理函数。
        atexit.register(self.exit)


    def exit(self):
        """
        清理资源并优雅退出

        执行顺序：
        1. 向所有worker进程发送"exit"指令
        2. 删除主进程的ModelRunner（释放GPU显存）
        3. 等待所有worker进程正常退出

        这个函数通过atexit注册，会在程序退出时自动调用。
        """
        # 通知所有worker进程退出
        self.model_runner.call("exit")
        # 删除主进程的模型运行器，释放显存
        del self.model_runner
        # 等待所有worker进程结束
        for process in self.processes:
            process.join()

    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        """
        执行一次推理步骤（一个iteration）

        这是引擎的核心循环函数，实现了continuous batching的关键逻辑：

        工作流程：
        1. 调用scheduler.schedule()决定本次要处理哪些序列
           - 可能返回prefill批次（处理新请求的prompt）
           - 也可能返回decode批次（为正在生成的序列生成下一个token）
        2. 调用model_runner.run()执行模型前向传播
           - 在分布式环境下会同步所有GPU
        3. 调用scheduler.postprocess()处理模型输出
           - 采样生成下一个token
           - 更新序列状态
           - 分配/释放KV缓存块
        4. 收集已完成的序列并返回

        Returns:
            三元组 (completed_sequences, num_processed_tokens, is_prefill):
            - completed_sequences: 已完成序列的列表 [(seq_id, token_ids), ...]
            - num_processed_tokens: 本次处理的token数量（用于吞吐量统计）
            - is_prefill: 本次是否为预填充阶段

        性能指标说明：
        - Prefill阶段：num_processed_tokens = 所有序列的prompt长度之和
        - Decode阶段：num_processed_tokens = 批次中的序列数量（每个序列生成1个token）
        """
        # 第一步：调度 - 决定本次要处理哪些序列
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        num_processed_tokens = 0

        # 如果没有可调度的序列，直接返回
        if not scheduled_sequences:
            return [], num_processed_tokens, is_prefill

        # 第二步：模型推理 - 执行前向传播
        # call()方法会将指令广播给所有worker进程，实现分布式推理
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)

        # 将GPU上的输出移到CPU并转换为Python列表
        if outputs is not None:
            outputs = outputs.cpu().tolist()

        # 第三步：后处理 - 采样、更新序列状态、管理KV缓存
        self.scheduler.postprocess(scheduled_sequences, outputs)

        # 收集本轮完成的序列（is_finished标志由postprocess设置）
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]

        # 计算处理的token数量（用于吞吐量统计）
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        return outputs, num_processed_tokens, is_prefill


    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        """
        添加一个新的生成请求到等待队列

        这是用户请求进入系统的入口。函数会：
        1. 使用tokenizer将prompt文本编码为token ids
        2. 创建Sequence对象（包含token ids、block_size、采样参数等）
        3. 将序列加入scheduler的等待队列

        Args:
            prompt: 用户输入的prompt文本
            sampling_params: 采样参数（temperature、top_p、max_tokens等）

        注意：
            - 此时序列还未分配KV缓存，真正的资源分配发生在schedule()时
            - block_size必须与引擎配置一致，用于KV缓存的分块管理
        """
        self.scheduler.add_sequence(
            Sequence(
                token_ids=self.tokenizer.encode(prompt),
                block_size=self.config['block_size'],
                sampling_params=sampling_params
            )
        )

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        """
        批量生成文本的高层接口

        这是用户使用引擎的主要入口函数。它实现了完整的生成流程：
        1. 将所有prompts加入等待队列
        2. 循环调用step()直到所有序列完成
        3. 收集生成结果并解码为文本

        工作原理（Continuous Batching）：
        - 不同prompt可能在不同时刻完成prefill
        - 短序列可能先完成，长序列继续生成
        - 新请求可以随时加入（虽然此函数一次性加入所有请求）
        - 调度器会动态调整批次，最大化GPU利用率

        Args:
            prompts: prompt文本列表
            sampling_params: 采样参数（所有prompts共享）

        Returns:
            字典，包含两个字段：
            - 'text': 解码后的生成文本列表
            - 'token_ids': 原始token id列表（用于调试或进一步处理）

        性能监控：
        - 打印每个step的吞吐量（tokens/sec）
        - 区分prefill和decode阶段的性能
        - prefill通常吞吐量更高（并行处理多个token）
        - decode吞吐量受限于自回归性质（每次只生成1个token）
        """
        # 将所有prompts加入调度器的等待队列
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)

        # 用于收集完成的序列 {seq_id: token_ids}
        generated_tokens = {}

        # 主循环：持续执行step直到所有序列完成
        while not self.scheduler.is_finished():
            start_t = time.time()

            # 执行一次推理步骤
            outputs, num_processed_tokens, is_prefill = self.step()

            end_t = time.time()
            running_time = end_t - start_t + 1e-10  # 避免除零

            # 打印性能指标
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens',
                      num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens',
                      num_processed_tokens/running_time, "tokens/sec during decoding")

            # 收集本轮完成的序列
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        # 按seq_id排序，确保输出顺序与输入顺序一致
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]

        # 将token ids解码为文本
        output = {
            'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens],
            'token_ids': generated_tokens
        }
        return output
