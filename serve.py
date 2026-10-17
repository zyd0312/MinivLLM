import asyncio
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.config import Config
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams

MODEL_PATH = os.path.expanduser(
    "~/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8"
)

app = FastAPI()
engine = None
engine_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    system: str | None = None
    max_tokens: int = 128
    temperature: float = 0.6
    max_model_length: int = 512


@app.on_event("startup")
def load_model():
    global engine

    config = Config({
        "model_name_or_path": MODEL_PATH,
        "quantization": "gptq",
        "world_size": 1,
        "enforce_eager": True,
        "max_num_sequences": 16,
        "max_num_batched_tokens": 1024,
        "max_cached_blocks": 1024,
        "block_size": 256,
        "max_model_length": 512,
        "gpu_memory_utilization": 0.9,
    })

    engine = LLMEngine(config)


@app.post("/generate")
async def generate(request: GenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is empty")

    params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        max_model_length=request.max_model_length,
    )

    # Qwen3 expects the conversation template and an assistant generation
    # marker. Passing the raw user text would make the model continue that text
    # instead of answering it.
    messages = []
    if request.system and request.system.strip():
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})
    formatted_prompt = engine.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # 当前引擎不是线程安全的，先串行处理请求
    async with engine_lock:
        result = await asyncio.to_thread(
            engine.generate,
            [formatted_prompt],
            params,
        )

    return {
        "text": result["text"][0],
        "token_ids": result["token_ids"][0],
    }
