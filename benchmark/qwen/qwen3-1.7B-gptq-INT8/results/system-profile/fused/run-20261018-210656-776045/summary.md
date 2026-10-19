# Fused 路径整体 GPU Profile

- 模型：`/home/zzzz/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8`
- Backend：`fused`
- Prompt tokens：20
- Profile max completion tokens：8

本报告包含 profiler instrumentation，只用于定位瓶颈；正式吞吐和延迟应以无 profiler 的 performance 报告为准。

⚠️ **当前 trace 未采集到 CUDA kernel activity。设备时间 unavailable；请勿把它解释为 0 ms。**


## 阶段级耗时

| 阶段 | 次数 | Wall P50 | Wall P90 | CUDA stream P50 | CUDA stream P90 | 输入 token |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 148.334 ms | 148.334 ms | 148.210 ms | 148.210 ms | 20 |
| decode | 7 | 115.604 ms | 153.334 ms | 115.562 ms | 153.291 ms | 7 |

CUDA event 时间是 stream 上的经过时间，可能包含 kernel 间空隙，不等同于 GPU kernel busy time；不能用 Wall 减去此值直接推算 CPU 时间。

## GPTQ 投影（CUDA event）

按每个 fused GPTQ 投影调用的 CUDA event 时间分组。该数据不依赖 CUPTI kernel activity trace，但仍可能包含该投影范围内的 stream 空隙。

| 阶段 | 投影 | 输入形状 | 次数 | 平均 stream ms | P50 ms | P90 ms |
|---|---|---:|---:|---:|---:|---:|
| decode | `model.layers.27.mlp.gate_proj` | `[1, 2048]` | 7 | 1.125 | 0.302 | 2.868 |
| decode | `model.layers.1.self_attn.q_proj` | `[1, 2048]` | 7 | 0.972 | 0.166 | 2.457 |
| decode | `model.layers.20.mlp.down_proj` | `[1, 6144]` | 7 | 0.943 | 0.219 | 2.291 |
| decode | `model.layers.7.self_attn.k_proj` | `[1, 2048]` | 7 | 0.828 | 0.171 | 2.416 |
| decode | `model.layers.27.mlp.up_proj` | `[1, 2048]` | 7 | 0.824 | 0.223 | 2.013 |
| decode | `model.layers.27.self_attn.k_proj` | `[1, 2048]` | 7 | 0.800 | 0.155 | 1.993 |
| decode | `model.layers.9.self_attn.v_proj` | `[1, 2048]` | 7 | 0.654 | 0.254 | 1.657 |
| decode | `model.layers.4.self_attn.q_proj` | `[1, 2048]` | 7 | 0.640 | 0.194 | 1.549 |
| decode | `model.layers.26.self_attn.k_proj` | `[1, 2048]` | 7 | 0.627 | 0.172 | 1.667 |
| decode | `model.layers.27.self_attn.q_proj` | `[1, 2048]` | 7 | 0.615 | 0.174 | 1.488 |
| decode | `model.layers.26.self_attn.q_proj` | `[1, 2048]` | 7 | 0.548 | 0.273 | 1.212 |
| decode | `model.layers.14.self_attn.q_proj` | `[1, 2048]` | 7 | 0.545 | 0.258 | 1.216 |
| decode | `model.layers.16.self_attn.o_proj` | `[1, 2048]` | 7 | 0.530 | 0.366 | 1.138 |
| decode | `model.layers.7.self_attn.o_proj` | `[1, 2048]` | 7 | 0.527 | 0.259 | 1.223 |
| decode | `model.layers.7.self_attn.v_proj` | `[1, 2048]` | 7 | 0.524 | 0.248 | 1.280 |
| decode | `model.layers.16.mlp.gate_proj` | `[1, 2048]` | 7 | 0.497 | 0.235 | 1.142 |
| decode | `model.layers.14.self_attn.k_proj` | `[1, 2048]` | 7 | 0.476 | 0.268 | 1.157 |
| decode | `model.layers.24.mlp.gate_proj` | `[1, 2048]` | 7 | 0.450 | 0.170 | 0.983 |
| decode | `model.layers.20.self_attn.v_proj` | `[1, 2048]` | 7 | 0.409 | 0.205 | 0.890 |
| decode | `model.layers.25.mlp.up_proj` | `[1, 2048]` | 7 | 0.405 | 0.235 | 0.894 |
| decode | `model.layers.15.self_attn.v_proj` | `[1, 2048]` | 7 | 0.397 | 0.209 | 0.823 |
| decode | `model.layers.26.self_attn.v_proj` | `[1, 2048]` | 7 | 0.397 | 0.176 | 0.882 |
| decode | `model.layers.6.mlp.down_proj` | `[1, 6144]` | 7 | 0.395 | 0.212 | 0.849 |
| decode | `model.layers.4.self_attn.o_proj` | `[1, 2048]` | 7 | 0.389 | 0.225 | 0.764 |
| decode | `model.layers.7.mlp.gate_proj` | `[1, 2048]` | 7 | 0.382 | 0.162 | 0.814 |
| decode | `model.layers.0.mlp.down_proj` | `[1, 6144]` | 7 | 0.381 | 0.211 | 0.723 |
| decode | `model.layers.6.mlp.gate_proj` | `[1, 2048]` | 7 | 0.377 | 0.210 | 0.754 |
| decode | `model.layers.0.mlp.up_proj` | `[1, 2048]` | 7 | 0.368 | 0.188 | 0.787 |
| decode | `model.layers.19.self_attn.o_proj` | `[1, 2048]` | 7 | 0.363 | 0.185 | 0.788 |
| decode | `model.layers.14.self_attn.v_proj` | `[1, 2048]` | 7 | 0.351 | 0.189 | 0.701 |
| decode | `model.layers.13.mlp.down_proj` | `[1, 6144]` | 7 | 0.351 | 0.259 | 0.586 |
| decode | `model.layers.25.mlp.down_proj` | `[1, 6144]` | 7 | 0.350 | 0.235 | 0.621 |
| decode | `model.layers.13.self_attn.v_proj` | `[1, 2048]` | 7 | 0.348 | 0.189 | 0.702 |
| decode | `model.layers.9.self_attn.o_proj` | `[1, 2048]` | 7 | 0.340 | 0.281 | 0.608 |
| decode | `model.layers.22.self_attn.q_proj` | `[1, 2048]` | 7 | 0.334 | 0.183 | 0.636 |
| decode | `model.layers.19.mlp.up_proj` | `[1, 2048]` | 7 | 0.332 | 0.167 | 0.704 |
| decode | `model.layers.21.mlp.gate_proj` | `[1, 2048]` | 7 | 0.330 | 0.216 | 0.619 |
| decode | `model.layers.26.mlp.gate_proj` | `[1, 2048]` | 7 | 0.329 | 0.272 | 0.551 |
| decode | `model.layers.22.mlp.up_proj` | `[1, 2048]` | 7 | 0.329 | 0.154 | 0.696 |
| decode | `model.layers.19.mlp.down_proj` | `[1, 6144]` | 7 | 0.325 | 0.182 | 0.581 |
| decode | `model.layers.3.mlp.gate_proj` | `[1, 2048]` | 7 | 0.319 | 0.185 | 0.595 |
| decode | `model.layers.7.self_attn.q_proj` | `[1, 2048]` | 7 | 0.318 | 0.151 | 0.659 |
| decode | `model.layers.13.mlp.gate_proj` | `[1, 2048]` | 7 | 0.315 | 0.214 | 0.586 |
| decode | `model.layers.3.self_attn.o_proj` | `[1, 2048]` | 7 | 0.312 | 0.175 | 0.602 |
| decode | `model.layers.23.self_attn.v_proj` | `[1, 2048]` | 7 | 0.310 | 0.143 | 0.715 |
| decode | `model.layers.27.self_attn.o_proj` | `[1, 2048]` | 7 | 0.309 | 0.222 | 0.525 |
| decode | `model.layers.18.self_attn.v_proj` | `[1, 2048]` | 7 | 0.304 | 0.118 | 0.695 |
| decode | `model.layers.6.self_attn.o_proj` | `[1, 2048]` | 7 | 0.299 | 0.166 | 0.563 |
| decode | `model.layers.9.mlp.gate_proj` | `[1, 2048]` | 7 | 0.299 | 0.324 | 0.369 |
| decode | `model.layers.4.mlp.up_proj` | `[1, 2048]` | 7 | 0.298 | 0.173 | 0.587 |
| decode | `model.layers.17.mlp.down_proj` | `[1, 6144]` | 7 | 0.295 | 0.211 | 0.483 |
| decode | `model.layers.11.self_attn.o_proj` | `[1, 2048]` | 7 | 0.295 | 0.274 | 0.502 |
| decode | `model.layers.14.self_attn.o_proj` | `[1, 2048]` | 7 | 0.295 | 0.197 | 0.542 |
| decode | `model.layers.16.mlp.up_proj` | `[1, 2048]` | 7 | 0.294 | 0.198 | 0.545 |
| decode | `model.layers.4.mlp.gate_proj` | `[1, 2048]` | 7 | 0.288 | 0.222 | 0.486 |
| decode | `model.layers.2.mlp.gate_proj` | `[1, 2048]` | 7 | 0.287 | 0.204 | 0.564 |
| decode | `model.layers.3.mlp.down_proj` | `[1, 6144]` | 7 | 0.281 | 0.194 | 0.462 |
| decode | `model.layers.13.mlp.up_proj` | `[1, 2048]` | 7 | 0.279 | 0.285 | 0.443 |
| decode | `model.layers.15.self_attn.k_proj` | `[1, 2048]` | 7 | 0.277 | 0.191 | 0.556 |
| decode | `model.layers.15.mlp.down_proj` | `[1, 6144]` | 7 | 0.277 | 0.272 | 0.380 |
| decode | `model.layers.21.self_attn.q_proj` | `[1, 2048]` | 7 | 0.275 | 0.210 | 0.489 |
| decode | `model.layers.22.self_attn.k_proj` | `[1, 2048]` | 7 | 0.273 | 0.135 | 0.546 |
| decode | `model.layers.18.self_attn.k_proj` | `[1, 2048]` | 7 | 0.270 | 0.191 | 0.512 |
| decode | `model.layers.18.self_attn.o_proj` | `[1, 2048]` | 7 | 0.269 | 0.180 | 0.491 |
| decode | `model.layers.12.mlp.down_proj` | `[1, 6144]` | 7 | 0.269 | 0.268 | 0.354 |
| decode | `model.layers.13.self_attn.q_proj` | `[1, 2048]` | 7 | 0.268 | 0.163 | 0.452 |
| decode | `model.layers.9.mlp.up_proj` | `[1, 2048]` | 7 | 0.264 | 0.271 | 0.342 |
| decode | `model.layers.5.self_attn.k_proj` | `[1, 2048]` | 7 | 0.264 | 0.193 | 0.466 |
| decode | `model.layers.19.mlp.gate_proj` | `[1, 2048]` | 7 | 0.262 | 0.182 | 0.459 |
| decode | `model.layers.4.mlp.down_proj` | `[1, 6144]` | 7 | 0.257 | 0.248 | 0.362 |
| decode | `model.layers.27.mlp.down_proj` | `[1, 6144]` | 7 | 0.255 | 0.256 | 0.326 |
| decode | `model.layers.0.mlp.gate_proj` | `[1, 2048]` | 7 | 0.254 | 0.187 | 0.411 |
| decode | `model.layers.25.self_attn.q_proj` | `[1, 2048]` | 7 | 0.254 | 0.177 | 0.468 |
| decode | `model.layers.27.self_attn.v_proj` | `[1, 2048]` | 7 | 0.253 | 0.200 | 0.393 |
| decode | `model.layers.17.self_attn.q_proj` | `[1, 2048]` | 7 | 0.252 | 0.236 | 0.399 |
| decode | `model.layers.11.mlp.gate_proj` | `[1, 2048]` | 7 | 0.252 | 0.165 | 0.490 |
| decode | `model.layers.21.mlp.up_proj` | `[1, 2048]` | 7 | 0.252 | 0.147 | 0.455 |
| decode | `model.layers.0.self_attn.o_proj` | `[1, 2048]` | 7 | 0.251 | 0.244 | 0.384 |
| decode | `model.layers.5.mlp.down_proj` | `[1, 6144]` | 7 | 0.250 | 0.269 | 0.284 |
| decode | `model.layers.0.self_attn.q_proj` | `[1, 2048]` | 7 | 0.249 | 0.232 | 0.318 |
| decode | `model.layers.6.mlp.up_proj` | `[1, 2048]` | 7 | 0.249 | 0.211 | 0.384 |
| decode | `model.layers.11.mlp.up_proj` | `[1, 2048]` | 7 | 0.249 | 0.153 | 0.453 |
| decode | `model.layers.21.mlp.down_proj` | `[1, 6144]` | 7 | 0.248 | 0.208 | 0.334 |
| decode | `model.layers.9.mlp.down_proj` | `[1, 6144]` | 7 | 0.247 | 0.256 | 0.287 |
| decode | `model.layers.22.mlp.gate_proj` | `[1, 2048]` | 7 | 0.245 | 0.196 | 0.409 |
| decode | `model.layers.11.mlp.down_proj` | `[1, 6144]` | 7 | 0.241 | 0.199 | 0.343 |
| decode | `model.layers.21.self_attn.v_proj` | `[1, 2048]` | 7 | 0.241 | 0.132 | 0.455 |
| decode | `model.layers.4.self_attn.v_proj` | `[1, 2048]` | 7 | 0.240 | 0.152 | 0.426 |
| decode | `model.layers.3.mlp.up_proj` | `[1, 2048]` | 7 | 0.240 | 0.147 | 0.433 |
| decode | `model.layers.8.mlp.down_proj` | `[1, 6144]` | 7 | 0.238 | 0.220 | 0.296 |
| decode | `model.layers.8.self_attn.o_proj` | `[1, 2048]` | 7 | 0.238 | 0.208 | 0.319 |
| decode | `model.layers.16.mlp.down_proj` | `[1, 6144]` | 7 | 0.238 | 0.227 | 0.290 |
| decode | `model.layers.7.mlp.down_proj` | `[1, 6144]` | 7 | 0.236 | 0.220 | 0.293 |
| decode | `model.layers.14.mlp.up_proj` | `[1, 2048]` | 7 | 0.236 | 0.173 | 0.375 |
| decode | `model.layers.12.mlp.up_proj` | `[1, 2048]` | 7 | 0.233 | 0.189 | 0.392 |
| decode | `model.layers.5.self_attn.o_proj` | `[1, 2048]` | 7 | 0.233 | 0.221 | 0.293 |
| decode | `model.layers.9.self_attn.k_proj` | `[1, 2048]` | 7 | 0.233 | 0.224 | 0.353 |
| decode | `model.layers.9.self_attn.q_proj` | `[1, 2048]` | 7 | 0.233 | 0.220 | 0.316 |
| decode | `model.layers.22.mlp.down_proj` | `[1, 6144]` | 7 | 0.230 | 0.215 | 0.370 |
| decode | `model.layers.5.self_attn.q_proj` | `[1, 2048]` | 7 | 0.228 | 0.178 | 0.362 |
| decode | `model.layers.26.mlp.down_proj` | `[1, 6144]` | 7 | 0.227 | 0.223 | 0.260 |
| decode | `model.layers.12.self_attn.v_proj` | `[1, 2048]` | 7 | 0.226 | 0.196 | 0.354 |
| decode | `model.layers.14.mlp.gate_proj` | `[1, 2048]` | 7 | 0.225 | 0.181 | 0.346 |
| decode | `model.layers.18.mlp.gate_proj` | `[1, 2048]` | 7 | 0.225 | 0.184 | 0.326 |
| decode | `model.layers.25.self_attn.o_proj` | `[1, 2048]` | 7 | 0.225 | 0.156 | 0.349 |
| decode | `model.layers.10.mlp.gate_proj` | `[1, 2048]` | 7 | 0.224 | 0.229 | 0.336 |
| decode | `model.layers.18.self_attn.q_proj` | `[1, 2048]` | 7 | 0.224 | 0.196 | 0.355 |
| decode | `model.layers.13.self_attn.k_proj` | `[1, 2048]` | 7 | 0.224 | 0.153 | 0.383 |
| decode | `model.layers.5.mlp.gate_proj` | `[1, 2048]` | 7 | 0.223 | 0.199 | 0.317 |
| decode | `model.layers.7.mlp.up_proj` | `[1, 2048]` | 7 | 0.221 | 0.145 | 0.351 |
| decode | `model.layers.1.mlp.down_proj` | `[1, 6144]` | 7 | 0.219 | 0.201 | 0.310 |
| decode | `model.layers.17.mlp.gate_proj` | `[1, 2048]` | 7 | 0.219 | 0.189 | 0.309 |
| decode | `model.layers.23.mlp.up_proj` | `[1, 2048]` | 7 | 0.218 | 0.148 | 0.361 |
| decode | `model.layers.19.self_attn.k_proj` | `[1, 2048]` | 7 | 0.217 | 0.152 | 0.353 |
| decode | `model.layers.12.mlp.gate_proj` | `[1, 2048]` | 7 | 0.215 | 0.155 | 0.348 |
| decode | `model.layers.25.mlp.gate_proj` | `[1, 2048]` | 7 | 0.214 | 0.228 | 0.293 |
| decode | `model.layers.14.mlp.down_proj` | `[1, 6144]` | 7 | 0.214 | 0.219 | 0.246 |
| decode | `model.layers.12.self_attn.k_proj` | `[1, 2048]` | 7 | 0.212 | 0.132 | 0.383 |
| decode | `model.layers.10.mlp.down_proj` | `[1, 6144]` | 7 | 0.211 | 0.216 | 0.273 |
| decode | `model.layers.15.self_attn.o_proj` | `[1, 2048]` | 7 | 0.209 | 0.183 | 0.301 |
| decode | `model.layers.8.mlp.gate_proj` | `[1, 2048]` | 7 | 0.208 | 0.184 | 0.296 |
| decode | `model.layers.2.mlp.down_proj` | `[1, 6144]` | 7 | 0.208 | 0.199 | 0.291 |
| decode | `model.layers.10.self_attn.k_proj` | `[1, 2048]` | 7 | 0.208 | 0.186 | 0.295 |
| decode | `model.layers.15.self_attn.q_proj` | `[1, 2048]` | 7 | 0.206 | 0.164 | 0.301 |
| decode | `model.layers.22.self_attn.v_proj` | `[1, 2048]` | 7 | 0.206 | 0.150 | 0.358 |
| decode | `model.layers.17.mlp.up_proj` | `[1, 2048]` | 7 | 0.205 | 0.183 | 0.342 |
| decode | `model.layers.15.mlp.gate_proj` | `[1, 2048]` | 7 | 0.204 | 0.161 | 0.306 |
| decode | `model.layers.26.mlp.up_proj` | `[1, 2048]` | 7 | 0.204 | 0.188 | 0.264 |
| decode | `model.layers.22.self_attn.o_proj` | `[1, 2048]` | 7 | 0.203 | 0.180 | 0.352 |
| decode | `model.layers.18.mlp.down_proj` | `[1, 6144]` | 7 | 0.202 | 0.200 | 0.220 |
| decode | `model.layers.16.self_attn.q_proj` | `[1, 2048]` | 7 | 0.200 | 0.220 | 0.261 |
| decode | `model.layers.17.self_attn.k_proj` | `[1, 2048]` | 7 | 0.200 | 0.220 | 0.279 |
| decode | `model.layers.10.self_attn.q_proj` | `[1, 2048]` | 7 | 0.199 | 0.216 | 0.270 |
| decode | `model.layers.24.mlp.down_proj` | `[1, 6144]` | 7 | 0.199 | 0.178 | 0.244 |
| decode | `model.layers.26.self_attn.o_proj` | `[1, 2048]` | 7 | 0.198 | 0.209 | 0.250 |
| decode | `model.layers.21.self_attn.k_proj` | `[1, 2048]` | 7 | 0.198 | 0.162 | 0.305 |
| decode | `model.layers.13.self_attn.o_proj` | `[1, 2048]` | 7 | 0.197 | 0.193 | 0.273 |
| decode | `model.layers.23.self_attn.o_proj` | `[1, 2048]` | 7 | 0.197 | 0.182 | 0.298 |
| decode | `model.layers.16.self_attn.k_proj` | `[1, 2048]` | 7 | 0.196 | 0.205 | 0.292 |
| decode | `model.layers.23.mlp.down_proj` | `[1, 6144]` | 7 | 0.195 | 0.189 | 0.224 |
| decode | `model.layers.8.self_attn.q_proj` | `[1, 2048]` | 7 | 0.193 | 0.146 | 0.292 |
| decode | `model.layers.5.self_attn.v_proj` | `[1, 2048]` | 7 | 0.193 | 0.161 | 0.281 |
| decode | `model.layers.12.self_attn.o_proj` | `[1, 2048]` | 7 | 0.193 | 0.155 | 0.286 |
| decode | `model.layers.16.self_attn.v_proj` | `[1, 2048]` | 7 | 0.192 | 0.157 | 0.323 |
| decode | `model.layers.23.self_attn.k_proj` | `[1, 2048]` | 7 | 0.192 | 0.143 | 0.391 |
| decode | `model.layers.24.self_attn.q_proj` | `[1, 2048]` | 7 | 0.191 | 0.143 | 0.302 |
| decode | `model.layers.12.self_attn.q_proj` | `[1, 2048]` | 7 | 0.191 | 0.138 | 0.316 |
| decode | `model.layers.6.self_attn.q_proj` | `[1, 2048]` | 7 | 0.190 | 0.172 | 0.248 |
| decode | `model.layers.15.mlp.up_proj` | `[1, 2048]` | 7 | 0.190 | 0.166 | 0.261 |
| decode | `model.layers.2.self_attn.q_proj` | `[1, 2048]` | 7 | 0.189 | 0.212 | 0.284 |
| decode | `model.layers.24.self_attn.o_proj` | `[1, 2048]` | 7 | 0.186 | 0.140 | 0.289 |
| decode | `model.layers.21.self_attn.o_proj` | `[1, 2048]` | 7 | 0.185 | 0.133 | 0.266 |
| decode | `model.layers.19.self_attn.q_proj` | `[1, 2048]` | 7 | 0.185 | 0.167 | 0.268 |
| decode | `model.layers.19.self_attn.v_proj` | `[1, 2048]` | 7 | 0.184 | 0.138 | 0.284 |
| decode | `model.layers.20.self_attn.o_proj` | `[1, 2048]` | 7 | 0.182 | 0.146 | 0.292 |
| decode | `model.layers.11.self_attn.q_proj` | `[1, 2048]` | 7 | 0.182 | 0.184 | 0.268 |
| decode | `model.layers.8.mlp.up_proj` | `[1, 2048]` | 7 | 0.181 | 0.169 | 0.226 |
| decode | `model.layers.5.mlp.up_proj` | `[1, 2048]` | 7 | 0.181 | 0.179 | 0.247 |
| decode | `model.layers.23.mlp.gate_proj` | `[1, 2048]` | 7 | 0.179 | 0.179 | 0.205 |
| decode | `model.layers.0.self_attn.k_proj` | `[1, 2048]` | 7 | 0.178 | 0.178 | 0.220 |
| decode | `model.layers.3.self_attn.q_proj` | `[1, 2048]` | 7 | 0.177 | 0.172 | 0.260 |
| decode | `model.layers.17.self_attn.o_proj` | `[1, 2048]` | 7 | 0.176 | 0.161 | 0.237 |
| decode | `model.layers.20.mlp.gate_proj` | `[1, 2048]` | 7 | 0.175 | 0.150 | 0.225 |
| decode | `model.layers.24.mlp.up_proj` | `[1, 2048]` | 7 | 0.174 | 0.142 | 0.256 |
| decode | `model.layers.8.self_attn.v_proj` | `[1, 2048]` | 7 | 0.174 | 0.183 | 0.226 |
| decode | `model.layers.10.self_attn.o_proj` | `[1, 2048]` | 7 | 0.172 | 0.184 | 0.245 |
| decode | `model.layers.4.self_attn.k_proj` | `[1, 2048]` | 7 | 0.172 | 0.165 | 0.208 |
| decode | `model.layers.6.self_attn.k_proj` | `[1, 2048]` | 7 | 0.171 | 0.146 | 0.245 |
| decode | `model.layers.8.self_attn.k_proj` | `[1, 2048]` | 7 | 0.170 | 0.133 | 0.234 |
| decode | `model.layers.23.self_attn.q_proj` | `[1, 2048]` | 7 | 0.169 | 0.147 | 0.327 |
| decode | `model.layers.25.self_attn.k_proj` | `[1, 2048]` | 7 | 0.167 | 0.147 | 0.251 |
| decode | `model.layers.18.mlp.up_proj` | `[1, 2048]` | 7 | 0.165 | 0.160 | 0.212 |
| decode | `model.layers.2.mlp.up_proj` | `[1, 2048]` | 7 | 0.165 | 0.151 | 0.249 |
| decode | `model.layers.20.self_attn.q_proj` | `[1, 2048]` | 7 | 0.164 | 0.137 | 0.247 |
| decode | `model.layers.3.self_attn.k_proj` | `[1, 2048]` | 7 | 0.164 | 0.161 | 0.224 |
| decode | `model.layers.1.self_attn.o_proj` | `[1, 2048]` | 7 | 0.163 | 0.181 | 0.226 |
| decode | `model.layers.3.self_attn.v_proj` | `[1, 2048]` | 7 | 0.161 | 0.143 | 0.252 |
| decode | `model.layers.2.self_attn.o_proj` | `[1, 2048]` | 7 | 0.161 | 0.145 | 0.237 |
| decode | `model.layers.2.self_attn.k_proj` | `[1, 2048]` | 7 | 0.161 | 0.146 | 0.252 |
| decode | `model.layers.0.self_attn.v_proj` | `[1, 2048]` | 7 | 0.161 | 0.161 | 0.179 |
| decode | `model.layers.10.mlp.up_proj` | `[1, 2048]` | 7 | 0.160 | 0.163 | 0.233 |
| decode | `model.layers.25.self_attn.v_proj` | `[1, 2048]` | 7 | 0.160 | 0.131 | 0.222 |
| decode | `model.layers.11.self_attn.v_proj` | `[1, 2048]` | 7 | 0.160 | 0.161 | 0.213 |
| decode | `model.layers.24.self_attn.k_proj` | `[1, 2048]` | 7 | 0.158 | 0.125 | 0.229 |
| decode | `model.layers.10.self_attn.v_proj` | `[1, 2048]` | 7 | 0.156 | 0.160 | 0.188 |
| decode | `model.layers.6.self_attn.v_proj` | `[1, 2048]` | 7 | 0.155 | 0.145 | 0.187 |
| decode | `model.layers.2.self_attn.v_proj` | `[1, 2048]` | 7 | 0.155 | 0.163 | 0.228 |
| decode | `model.layers.11.self_attn.k_proj` | `[1, 2048]` | 7 | 0.154 | 0.164 | 0.214 |
| decode | `model.layers.20.mlp.up_proj` | `[1, 2048]` | 7 | 0.154 | 0.137 | 0.199 |
| decode | `model.layers.1.mlp.gate_proj` | `[1, 2048]` | 7 | 0.149 | 0.160 | 0.202 |
| decode | `model.layers.17.self_attn.v_proj` | `[1, 2048]` | 7 | 0.142 | 0.138 | 0.177 |
| decode | `model.layers.1.self_attn.v_proj` | `[1, 2048]` | 7 | 0.142 | 0.131 | 0.196 |
| decode | `model.layers.1.self_attn.k_proj` | `[1, 2048]` | 7 | 0.131 | 0.131 | 0.173 |
| decode | `model.layers.20.self_attn.k_proj` | `[1, 2048]` | 7 | 0.130 | 0.121 | 0.166 |
| decode | `model.layers.1.mlp.up_proj` | `[1, 2048]` | 7 | 0.126 | 0.136 | 0.153 |
| decode | `model.layers.24.self_attn.v_proj` | `[1, 2048]` | 7 | 0.125 | 0.130 | 0.157 |
| prefill | `model.layers.24.self_attn.v_proj` | `[20, 2048]` | 1 | 6.510 | 6.510 | 6.510 |
| prefill | `model.layers.22.mlp.up_proj` | `[20, 2048]` | 1 | 2.659 | 2.659 | 2.659 |
| prefill | `model.layers.23.self_attn.k_proj` | `[20, 2048]` | 1 | 2.536 | 2.536 | 2.536 |
| prefill | `model.layers.26.mlp.gate_proj` | `[20, 2048]` | 1 | 1.790 | 1.790 | 1.790 |
| prefill | `model.layers.16.mlp.down_proj` | `[20, 6144]` | 1 | 1.619 | 1.619 | 1.619 |
| prefill | `model.layers.17.self_attn.v_proj` | `[20, 2048]` | 1 | 1.526 | 1.526 | 1.526 |
| prefill | `model.layers.4.self_attn.v_proj` | `[20, 2048]` | 1 | 1.243 | 1.243 | 1.243 |
| prefill | `model.layers.4.self_attn.q_proj` | `[20, 2048]` | 1 | 1.177 | 1.177 | 1.177 |
| prefill | `model.layers.19.self_attn.v_proj` | `[20, 2048]` | 1 | 1.124 | 1.124 | 1.124 |
| prefill | `model.layers.22.mlp.down_proj` | `[20, 6144]` | 1 | 1.102 | 1.102 | 1.102 |
| prefill | `model.layers.17.self_attn.o_proj` | `[20, 2048]` | 1 | 0.983 | 0.983 | 0.983 |
| prefill | `model.layers.3.self_attn.o_proj` | `[20, 2048]` | 1 | 0.802 | 0.802 | 0.802 |
| prefill | `model.layers.19.self_attn.k_proj` | `[20, 2048]` | 1 | 0.723 | 0.723 | 0.723 |
| prefill | `model.layers.26.mlp.up_proj` | `[20, 2048]` | 1 | 0.710 | 0.710 | 0.710 |
| prefill | `model.layers.24.self_attn.o_proj` | `[20, 2048]` | 1 | 0.703 | 0.703 | 0.703 |
| prefill | `model.layers.3.mlp.gate_proj` | `[20, 2048]` | 1 | 0.623 | 0.623 | 0.623 |
| prefill | `model.layers.13.mlp.up_proj` | `[20, 2048]` | 1 | 0.613 | 0.613 | 0.613 |
| prefill | `model.layers.26.self_attn.o_proj` | `[20, 2048]` | 1 | 0.600 | 0.600 | 0.600 |
| prefill | `model.layers.27.mlp.up_proj` | `[20, 2048]` | 1 | 0.593 | 0.593 | 0.593 |
| prefill | `model.layers.4.mlp.gate_proj` | `[20, 2048]` | 1 | 0.566 | 0.566 | 0.566 |
| prefill | `model.layers.4.self_attn.k_proj` | `[20, 2048]` | 1 | 0.559 | 0.559 | 0.559 |
| prefill | `model.layers.20.mlp.gate_proj` | `[20, 2048]` | 1 | 0.514 | 0.514 | 0.514 |
| prefill | `model.layers.14.self_attn.o_proj` | `[20, 2048]` | 1 | 0.486 | 0.486 | 0.486 |
| prefill | `model.layers.21.mlp.down_proj` | `[20, 6144]` | 1 | 0.483 | 0.483 | 0.483 |
| prefill | `model.layers.15.self_attn.v_proj` | `[20, 2048]` | 1 | 0.453 | 0.453 | 0.453 |
| prefill | `model.layers.19.mlp.up_proj` | `[20, 2048]` | 1 | 0.429 | 0.429 | 0.429 |
| prefill | `model.layers.4.mlp.down_proj` | `[20, 6144]` | 1 | 0.416 | 0.416 | 0.416 |
| prefill | `model.layers.26.mlp.down_proj` | `[20, 6144]` | 1 | 0.415 | 0.415 | 0.415 |
| prefill | `model.layers.3.self_attn.q_proj` | `[20, 2048]` | 1 | 0.395 | 0.395 | 0.395 |
| prefill | `model.layers.19.mlp.gate_proj` | `[20, 2048]` | 1 | 0.386 | 0.386 | 0.386 |
| prefill | `model.layers.27.mlp.gate_proj` | `[20, 2048]` | 1 | 0.384 | 0.384 | 0.384 |
| prefill | `model.layers.13.mlp.down_proj` | `[20, 6144]` | 1 | 0.382 | 0.382 | 0.382 |
| prefill | `model.layers.10.self_attn.o_proj` | `[20, 2048]` | 1 | 0.381 | 0.381 | 0.381 |
| prefill | `model.layers.0.self_attn.v_proj` | `[20, 2048]` | 1 | 0.380 | 0.380 | 0.380 |
| prefill | `model.layers.14.mlp.down_proj` | `[20, 6144]` | 1 | 0.379 | 0.379 | 0.379 |
| prefill | `model.layers.15.mlp.down_proj` | `[20, 6144]` | 1 | 0.379 | 0.379 | 0.379 |
| prefill | `model.layers.27.mlp.down_proj` | `[20, 6144]` | 1 | 0.355 | 0.355 | 0.355 |
| prefill | `model.layers.20.mlp.down_proj` | `[20, 6144]` | 1 | 0.353 | 0.353 | 0.353 |
| prefill | `model.layers.19.mlp.down_proj` | `[20, 6144]` | 1 | 0.352 | 0.352 | 0.352 |
| prefill | `model.layers.6.self_attn.o_proj` | `[20, 2048]` | 1 | 0.348 | 0.348 | 0.348 |
| prefill | `model.layers.5.mlp.down_proj` | `[20, 6144]` | 1 | 0.337 | 0.337 | 0.337 |
| prefill | `model.layers.18.mlp.down_proj` | `[20, 6144]` | 1 | 0.331 | 0.331 | 0.331 |
| prefill | `model.layers.23.mlp.down_proj` | `[20, 6144]` | 1 | 0.327 | 0.327 | 0.327 |
| prefill | `model.layers.0.self_attn.o_proj` | `[20, 2048]` | 1 | 0.321 | 0.321 | 0.321 |
| prefill | `model.layers.22.self_attn.o_proj` | `[20, 2048]` | 1 | 0.320 | 0.320 | 0.320 |
| prefill | `model.layers.1.mlp.down_proj` | `[20, 6144]` | 1 | 0.317 | 0.317 | 0.317 |
| prefill | `model.layers.3.mlp.down_proj` | `[20, 6144]` | 1 | 0.317 | 0.317 | 0.317 |
| prefill | `model.layers.11.self_attn.q_proj` | `[20, 2048]` | 1 | 0.310 | 0.310 | 0.310 |
| prefill | `model.layers.0.self_attn.q_proj` | `[20, 2048]` | 1 | 0.310 | 0.310 | 0.310 |
| prefill | `model.layers.10.mlp.down_proj` | `[20, 6144]` | 1 | 0.309 | 0.309 | 0.309 |
| prefill | `model.layers.15.mlp.gate_proj` | `[20, 2048]` | 1 | 0.308 | 0.308 | 0.308 |
| prefill | `model.layers.14.mlp.gate_proj` | `[20, 2048]` | 1 | 0.306 | 0.306 | 0.306 |
| prefill | `model.layers.25.mlp.down_proj` | `[20, 6144]` | 1 | 0.305 | 0.305 | 0.305 |
| prefill | `model.layers.17.self_attn.k_proj` | `[20, 2048]` | 1 | 0.302 | 0.302 | 0.302 |
| prefill | `model.layers.2.mlp.down_proj` | `[20, 6144]` | 1 | 0.298 | 0.298 | 0.298 |
| prefill | `model.layers.0.mlp.down_proj` | `[20, 6144]` | 1 | 0.297 | 0.297 | 0.297 |
| prefill | `model.layers.24.mlp.down_proj` | `[20, 6144]` | 1 | 0.297 | 0.297 | 0.297 |
| prefill | `model.layers.16.self_attn.q_proj` | `[20, 2048]` | 1 | 0.297 | 0.297 | 0.297 |
| prefill | `model.layers.12.mlp.down_proj` | `[20, 6144]` | 1 | 0.295 | 0.295 | 0.295 |
| prefill | `model.layers.20.self_attn.o_proj` | `[20, 2048]` | 1 | 0.294 | 0.294 | 0.294 |
| prefill | `model.layers.17.mlp.down_proj` | `[20, 6144]` | 1 | 0.292 | 0.292 | 0.292 |
| prefill | `model.layers.7.mlp.down_proj` | `[20, 6144]` | 1 | 0.290 | 0.290 | 0.290 |
| prefill | `model.layers.11.mlp.down_proj` | `[20, 6144]` | 1 | 0.289 | 0.289 | 0.289 |
| prefill | `model.layers.22.self_attn.q_proj` | `[20, 2048]` | 1 | 0.289 | 0.289 | 0.289 |
| prefill | `model.layers.8.mlp.down_proj` | `[20, 6144]` | 1 | 0.287 | 0.287 | 0.287 |
| prefill | `model.layers.15.self_attn.q_proj` | `[20, 2048]` | 1 | 0.285 | 0.285 | 0.285 |
| prefill | `model.layers.9.mlp.down_proj` | `[20, 6144]` | 1 | 0.280 | 0.280 | 0.280 |
| prefill | `model.layers.6.mlp.down_proj` | `[20, 6144]` | 1 | 0.278 | 0.278 | 0.278 |
| prefill | `model.layers.17.self_attn.q_proj` | `[20, 2048]` | 1 | 0.275 | 0.275 | 0.275 |
| prefill | `model.layers.15.mlp.up_proj` | `[20, 2048]` | 1 | 0.271 | 0.271 | 0.271 |
| prefill | `model.layers.24.mlp.gate_proj` | `[20, 2048]` | 1 | 0.271 | 0.271 | 0.271 |
| prefill | `model.layers.18.self_attn.q_proj` | `[20, 2048]` | 1 | 0.271 | 0.271 | 0.271 |
| prefill | `model.layers.19.self_attn.o_proj` | `[20, 2048]` | 1 | 0.270 | 0.270 | 0.270 |
| prefill | `model.layers.23.self_attn.q_proj` | `[20, 2048]` | 1 | 0.270 | 0.270 | 0.270 |
| prefill | `model.layers.4.mlp.up_proj` | `[20, 2048]` | 1 | 0.269 | 0.269 | 0.269 |
| prefill | `model.layers.27.self_attn.o_proj` | `[20, 2048]` | 1 | 0.268 | 0.268 | 0.268 |
| prefill | `model.layers.4.self_attn.o_proj` | `[20, 2048]` | 1 | 0.267 | 0.267 | 0.267 |
| prefill | `model.layers.13.mlp.gate_proj` | `[20, 2048]` | 1 | 0.261 | 0.261 | 0.261 |
| prefill | `model.layers.22.mlp.gate_proj` | `[20, 2048]` | 1 | 0.260 | 0.260 | 0.260 |
| prefill | `model.layers.14.self_attn.q_proj` | `[20, 2048]` | 1 | 0.258 | 0.258 | 0.258 |
| prefill | `model.layers.10.mlp.gate_proj` | `[20, 2048]` | 1 | 0.257 | 0.257 | 0.257 |
| prefill | `model.layers.16.self_attn.k_proj` | `[20, 2048]` | 1 | 0.254 | 0.254 | 0.254 |
| prefill | `model.layers.14.mlp.up_proj` | `[20, 2048]` | 1 | 0.254 | 0.254 | 0.254 |
| prefill | `model.layers.13.self_attn.o_proj` | `[20, 2048]` | 1 | 0.253 | 0.253 | 0.253 |
| prefill | `model.layers.3.mlp.up_proj` | `[20, 2048]` | 1 | 0.251 | 0.251 | 0.251 |
| prefill | `model.layers.20.self_attn.k_proj` | `[20, 2048]` | 1 | 0.243 | 0.243 | 0.243 |
| prefill | `model.layers.15.self_attn.k_proj` | `[20, 2048]` | 1 | 0.241 | 0.241 | 0.241 |
| prefill | `model.layers.5.self_attn.q_proj` | `[20, 2048]` | 1 | 0.241 | 0.241 | 0.241 |
| prefill | `model.layers.16.self_attn.v_proj` | `[20, 2048]` | 1 | 0.240 | 0.240 | 0.240 |
| prefill | `model.layers.16.self_attn.o_proj` | `[20, 2048]` | 1 | 0.235 | 0.235 | 0.235 |
| prefill | `model.layers.11.mlp.gate_proj` | `[20, 2048]` | 1 | 0.230 | 0.230 | 0.230 |
| prefill | `model.layers.22.self_attn.k_proj` | `[20, 2048]` | 1 | 0.226 | 0.226 | 0.226 |
| prefill | `model.layers.15.self_attn.o_proj` | `[20, 2048]` | 1 | 0.225 | 0.225 | 0.225 |
| prefill | `model.layers.2.self_attn.v_proj` | `[20, 2048]` | 1 | 0.225 | 0.225 | 0.225 |
| prefill | `model.layers.20.mlp.up_proj` | `[20, 2048]` | 1 | 0.223 | 0.223 | 0.223 |
| prefill | `model.layers.17.mlp.gate_proj` | `[20, 2048]` | 1 | 0.222 | 0.222 | 0.222 |
| prefill | `model.layers.23.self_attn.v_proj` | `[20, 2048]` | 1 | 0.218 | 0.218 | 0.218 |
| prefill | `model.layers.20.self_attn.q_proj` | `[20, 2048]` | 1 | 0.217 | 0.217 | 0.217 |
| prefill | `model.layers.23.mlp.gate_proj` | `[20, 2048]` | 1 | 0.215 | 0.215 | 0.215 |
| prefill | `model.layers.10.mlp.up_proj` | `[20, 2048]` | 1 | 0.213 | 0.213 | 0.213 |
| prefill | `model.layers.27.self_attn.q_proj` | `[20, 2048]` | 1 | 0.210 | 0.210 | 0.210 |
| prefill | `model.layers.13.self_attn.q_proj` | `[20, 2048]` | 1 | 0.210 | 0.210 | 0.210 |
| prefill | `model.layers.23.self_attn.o_proj` | `[20, 2048]` | 1 | 0.207 | 0.207 | 0.207 |
| prefill | `model.layers.26.self_attn.k_proj` | `[20, 2048]` | 1 | 0.205 | 0.205 | 0.205 |
| prefill | `model.layers.2.self_attn.o_proj` | `[20, 2048]` | 1 | 0.203 | 0.203 | 0.203 |
| prefill | `model.layers.0.mlp.gate_proj` | `[20, 2048]` | 1 | 0.202 | 0.202 | 0.202 |
| prefill | `model.layers.5.self_attn.v_proj` | `[20, 2048]` | 1 | 0.202 | 0.202 | 0.202 |
| prefill | `model.layers.26.self_attn.q_proj` | `[20, 2048]` | 1 | 0.202 | 0.202 | 0.202 |
| prefill | `model.layers.21.self_attn.q_proj` | `[20, 2048]` | 1 | 0.201 | 0.201 | 0.201 |
| prefill | `model.layers.22.self_attn.v_proj` | `[20, 2048]` | 1 | 0.201 | 0.201 | 0.201 |
| prefill | `model.layers.5.self_attn.k_proj` | `[20, 2048]` | 1 | 0.200 | 0.200 | 0.200 |
| prefill | `model.layers.6.self_attn.q_proj` | `[20, 2048]` | 1 | 0.199 | 0.199 | 0.199 |
| prefill | `model.layers.5.mlp.gate_proj` | `[20, 2048]` | 1 | 0.199 | 0.199 | 0.199 |
| prefill | `model.layers.2.mlp.up_proj` | `[20, 2048]` | 1 | 0.198 | 0.198 | 0.198 |
| prefill | `model.layers.27.self_attn.v_proj` | `[20, 2048]` | 1 | 0.198 | 0.198 | 0.198 |
| prefill | `model.layers.12.self_attn.q_proj` | `[20, 2048]` | 1 | 0.197 | 0.197 | 0.197 |
| prefill | `model.layers.2.mlp.gate_proj` | `[20, 2048]` | 1 | 0.196 | 0.196 | 0.196 |
| prefill | `model.layers.16.mlp.gate_proj` | `[20, 2048]` | 1 | 0.196 | 0.196 | 0.196 |
| prefill | `model.layers.10.self_attn.k_proj` | `[20, 2048]` | 1 | 0.195 | 0.195 | 0.195 |
| prefill | `model.layers.21.self_attn.o_proj` | `[20, 2048]` | 1 | 0.193 | 0.193 | 0.193 |
| prefill | `model.layers.21.mlp.gate_proj` | `[20, 2048]` | 1 | 0.193 | 0.193 | 0.193 |
| prefill | `model.layers.19.self_attn.q_proj` | `[20, 2048]` | 1 | 0.193 | 0.193 | 0.193 |
| prefill | `model.layers.5.self_attn.o_proj` | `[20, 2048]` | 1 | 0.193 | 0.193 | 0.193 |
| prefill | `model.layers.20.self_attn.v_proj` | `[20, 2048]` | 1 | 0.192 | 0.192 | 0.192 |
| prefill | `model.layers.8.mlp.gate_proj` | `[20, 2048]` | 1 | 0.192 | 0.192 | 0.192 |
| prefill | `model.layers.21.self_attn.k_proj` | `[20, 2048]` | 1 | 0.189 | 0.189 | 0.189 |
| prefill | `model.layers.24.self_attn.k_proj` | `[20, 2048]` | 1 | 0.189 | 0.189 | 0.189 |
| prefill | `model.layers.1.mlp.gate_proj` | `[20, 2048]` | 1 | 0.189 | 0.189 | 0.189 |
| prefill | `model.layers.25.mlp.gate_proj` | `[20, 2048]` | 1 | 0.188 | 0.188 | 0.188 |
| prefill | `model.layers.16.mlp.up_proj` | `[20, 2048]` | 1 | 0.188 | 0.188 | 0.188 |
| prefill | `model.layers.6.mlp.gate_proj` | `[20, 2048]` | 1 | 0.187 | 0.187 | 0.187 |
| prefill | `model.layers.21.self_attn.v_proj` | `[20, 2048]` | 1 | 0.186 | 0.186 | 0.186 |
| prefill | `model.layers.18.mlp.gate_proj` | `[20, 2048]` | 1 | 0.186 | 0.186 | 0.186 |
| prefill | `model.layers.12.self_attn.o_proj` | `[20, 2048]` | 1 | 0.185 | 0.185 | 0.185 |
| prefill | `model.layers.9.mlp.gate_proj` | `[20, 2048]` | 1 | 0.184 | 0.184 | 0.184 |
| prefill | `model.layers.24.self_attn.q_proj` | `[20, 2048]` | 1 | 0.184 | 0.184 | 0.184 |
| prefill | `model.layers.18.self_attn.o_proj` | `[20, 2048]` | 1 | 0.182 | 0.182 | 0.182 |
| prefill | `model.layers.6.self_attn.v_proj` | `[20, 2048]` | 1 | 0.181 | 0.181 | 0.181 |
| prefill | `model.layers.6.self_attn.k_proj` | `[20, 2048]` | 1 | 0.181 | 0.181 | 0.181 |
| prefill | `model.layers.7.self_attn.o_proj` | `[20, 2048]` | 1 | 0.181 | 0.181 | 0.181 |
| prefill | `model.layers.12.mlp.gate_proj` | `[20, 2048]` | 1 | 0.180 | 0.180 | 0.180 |
| prefill | `model.layers.1.self_attn.q_proj` | `[20, 2048]` | 1 | 0.180 | 0.180 | 0.180 |
| prefill | `model.layers.1.self_attn.o_proj` | `[20, 2048]` | 1 | 0.180 | 0.180 | 0.180 |
| prefill | `model.layers.27.self_attn.k_proj` | `[20, 2048]` | 1 | 0.180 | 0.180 | 0.180 |
| prefill | `model.layers.7.self_attn.q_proj` | `[20, 2048]` | 1 | 0.179 | 0.179 | 0.179 |
| prefill | `model.layers.7.mlp.gate_proj` | `[20, 2048]` | 1 | 0.179 | 0.179 | 0.179 |
| prefill | `model.layers.2.self_attn.q_proj` | `[20, 2048]` | 1 | 0.179 | 0.179 | 0.179 |
| prefill | `model.layers.11.self_attn.o_proj` | `[20, 2048]` | 1 | 0.178 | 0.178 | 0.178 |
| prefill | `model.layers.8.mlp.up_proj` | `[20, 2048]` | 1 | 0.177 | 0.177 | 0.177 |
| prefill | `model.layers.14.self_attn.v_proj` | `[20, 2048]` | 1 | 0.176 | 0.176 | 0.176 |
| prefill | `model.layers.8.self_attn.q_proj` | `[20, 2048]` | 1 | 0.175 | 0.175 | 0.175 |
| prefill | `model.layers.3.self_attn.v_proj` | `[20, 2048]` | 1 | 0.175 | 0.175 | 0.175 |
| prefill | `model.layers.25.self_attn.k_proj` | `[20, 2048]` | 1 | 0.174 | 0.174 | 0.174 |
| prefill | `model.layers.3.self_attn.k_proj` | `[20, 2048]` | 1 | 0.174 | 0.174 | 0.174 |
| prefill | `model.layers.25.self_attn.q_proj` | `[20, 2048]` | 1 | 0.173 | 0.173 | 0.173 |
| prefill | `model.layers.26.self_attn.v_proj` | `[20, 2048]` | 1 | 0.173 | 0.173 | 0.173 |
| prefill | `model.layers.14.self_attn.k_proj` | `[20, 2048]` | 1 | 0.170 | 0.170 | 0.170 |
| prefill | `model.layers.9.self_attn.q_proj` | `[20, 2048]` | 1 | 0.169 | 0.169 | 0.169 |
| prefill | `model.layers.11.self_attn.k_proj` | `[20, 2048]` | 1 | 0.169 | 0.169 | 0.169 |
| prefill | `model.layers.18.self_attn.v_proj` | `[20, 2048]` | 1 | 0.169 | 0.169 | 0.169 |
| prefill | `model.layers.5.mlp.up_proj` | `[20, 2048]` | 1 | 0.169 | 0.169 | 0.169 |
| prefill | `model.layers.10.self_attn.q_proj` | `[20, 2048]` | 1 | 0.168 | 0.168 | 0.168 |
| prefill | `model.layers.24.mlp.up_proj` | `[20, 2048]` | 1 | 0.168 | 0.168 | 0.168 |
| prefill | `model.layers.0.self_attn.k_proj` | `[20, 2048]` | 1 | 0.167 | 0.167 | 0.167 |
| prefill | `model.layers.17.mlp.up_proj` | `[20, 2048]` | 1 | 0.167 | 0.167 | 0.167 |
| prefill | `model.layers.12.self_attn.k_proj` | `[20, 2048]` | 1 | 0.167 | 0.167 | 0.167 |
| prefill | `model.layers.2.self_attn.k_proj` | `[20, 2048]` | 1 | 0.166 | 0.166 | 0.166 |
| prefill | `model.layers.0.mlp.up_proj` | `[20, 2048]` | 1 | 0.165 | 0.165 | 0.165 |
| prefill | `model.layers.25.self_attn.o_proj` | `[20, 2048]` | 1 | 0.164 | 0.164 | 0.164 |
| prefill | `model.layers.21.mlp.up_proj` | `[20, 2048]` | 1 | 0.163 | 0.163 | 0.163 |
| prefill | `model.layers.1.mlp.up_proj` | `[20, 2048]` | 1 | 0.163 | 0.163 | 0.163 |
| prefill | `model.layers.18.self_attn.k_proj` | `[20, 2048]` | 1 | 0.162 | 0.162 | 0.162 |
| prefill | `model.layers.12.mlp.up_proj` | `[20, 2048]` | 1 | 0.162 | 0.162 | 0.162 |
| prefill | `model.layers.18.mlp.up_proj` | `[20, 2048]` | 1 | 0.160 | 0.160 | 0.160 |
| prefill | `model.layers.11.mlp.up_proj` | `[20, 2048]` | 1 | 0.160 | 0.160 | 0.160 |
| prefill | `model.layers.11.self_attn.v_proj` | `[20, 2048]` | 1 | 0.157 | 0.157 | 0.157 |
| prefill | `model.layers.13.self_attn.k_proj` | `[20, 2048]` | 1 | 0.155 | 0.155 | 0.155 |
| prefill | `model.layers.9.self_attn.o_proj` | `[20, 2048]` | 1 | 0.153 | 0.153 | 0.153 |
| prefill | `model.layers.8.self_attn.o_proj` | `[20, 2048]` | 1 | 0.153 | 0.153 | 0.153 |
| prefill | `model.layers.23.mlp.up_proj` | `[20, 2048]` | 1 | 0.151 | 0.151 | 0.151 |
| prefill | `model.layers.25.mlp.up_proj` | `[20, 2048]` | 1 | 0.149 | 0.149 | 0.149 |
| prefill | `model.layers.1.self_attn.k_proj` | `[20, 2048]` | 1 | 0.149 | 0.149 | 0.149 |
| prefill | `model.layers.10.self_attn.v_proj` | `[20, 2048]` | 1 | 0.149 | 0.149 | 0.149 |
| prefill | `model.layers.13.self_attn.v_proj` | `[20, 2048]` | 1 | 0.148 | 0.148 | 0.148 |
| prefill | `model.layers.9.self_attn.v_proj` | `[20, 2048]` | 1 | 0.147 | 0.147 | 0.147 |
| prefill | `model.layers.7.self_attn.k_proj` | `[20, 2048]` | 1 | 0.146 | 0.146 | 0.146 |
| prefill | `model.layers.8.self_attn.k_proj` | `[20, 2048]` | 1 | 0.146 | 0.146 | 0.146 |
| prefill | `model.layers.6.mlp.up_proj` | `[20, 2048]` | 1 | 0.145 | 0.145 | 0.145 |
| prefill | `model.layers.7.mlp.up_proj` | `[20, 2048]` | 1 | 0.141 | 0.141 | 0.141 |
| prefill | `model.layers.12.self_attn.v_proj` | `[20, 2048]` | 1 | 0.140 | 0.140 | 0.140 |
| prefill | `model.layers.25.self_attn.v_proj` | `[20, 2048]` | 1 | 0.140 | 0.140 | 0.140 |
| prefill | `model.layers.8.self_attn.v_proj` | `[20, 2048]` | 1 | 0.139 | 0.139 | 0.139 |
| prefill | `model.layers.9.self_attn.k_proj` | `[20, 2048]` | 1 | 0.137 | 0.137 | 0.137 |
| prefill | `model.layers.9.mlp.up_proj` | `[20, 2048]` | 1 | 0.137 | 0.137 | 0.137 |
| prefill | `model.layers.1.self_attn.v_proj` | `[20, 2048]` | 1 | 0.136 | 0.136 | 0.136 |
| prefill | `model.layers.7.self_attn.v_proj` | `[20, 2048]` | 1 | 0.134 | 0.134 | 0.134 |

## Profiler 热点

按 self device time 排序。设备时间 unavailable 表示本次未取得 CUDA kernel activity；它不代表耗时为零。

| Kernel/范围 | 调用次数 | Self CPU ms | CPU total ms | Self device ms | Device total ms |
|---|---:|---:|---:|---:|---:|
| `minivllm.decode.engine_step` | 7 | 262.867 | 881.172 | unavailable | unavailable |
| `minivllm.prefill.engine_step` | 1 | 41.149 | 147.938 | unavailable | unavailable |
| `## Call CompiledFxGraph fmfuv5i3nqtdmdg6aeyozimpba2zcaixbbbfel7jxzzpbnrw3tgn ##` | 392 | 35.808 | 55.995 | unavailable | unavailable |
| `TorchDynamo Cache Lookup` | 1360 | 34.651 | 34.651 | unavailable | unavailable |
| `## Call CompiledFxGraph ffvf7qxxpt7qrrdceopzzwi7ee6wf5ryetiqw3g7mfykkm6b3afd ##` | 399 | 28.753 | 42.595 | unavailable | unavailable |
| `Torch-Compiled Region: 0/5` | 392 | 21.345 | 87.936 | unavailable | unavailable |
| `aten::copy_` | 38 | 21.041 | 21.041 | unavailable | unavailable |
| `Torch-Compiled Region: 0/4` | 399 | 17.836 | 64.006 | unavailable | unavailable |
| `## Call CompiledFxGraph fhwel7axcf4i23zceukczppz5a2iulevtgccmcfip6ks72gahuze ##` | 196 | 17.756 | 31.560 | unavailable | unavailable |
| `## Call CompiledFxGraph fbpcpdnscprhs5imt4potezua4yibn25rtuduszr7b7qditcbkwr ##` | 196 | 16.075 | 24.431 | unavailable | unavailable |
| `aten::cat` | 448 | 15.886 | 15.886 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 399 | 13.842 | 13.842 | unavailable | unavailable |
| `aten::empty` | 1728 | 12.333 | 12.333 | unavailable | unavailable |
| `aten::add` | 392 | 12.206 | 12.206 | unavailable | unavailable |
| `AOTDispatcher Runtime Wrapper Prologue` | 1360 | 11.682 | 11.682 | unavailable | unavailable |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 196 | 11.359 | 11.359 | unavailable | unavailable |
| `Pregraph bytecode` | 1360 | 9.893 | 9.893 | unavailable | unavailable |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 196 | 8.828 | 8.828 | unavailable | unavailable |
| `triton_poi_fused_mul_silu_split_0` | 196 | 8.356 | 8.356 | unavailable | unavailable |
| `Torch-Compiled Region: 2/2` | 196 | 8.354 | 34.132 | unavailable | unavailable |
| `Torch-Compiled Region: 1/2` | 196 | 7.913 | 42.044 | unavailable | unavailable |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_0` | 196 | 7.712 | 7.712 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.24.self_attn.v_proj` | 8 | 6.545 | 6.605 | unavailable | unavailable |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_1` | 196 | 6.091 | 6.091 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.20.mlp.down_proj` | 8 | 6.041 | 6.139 | unavailable | unavailable |
| `aten::sub` | 204 | 5.991 | 5.991 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.27.mlp.gate_proj` | 8 | 5.729 | 5.886 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.7.self_attn.k_proj` | 8 | 5.076 | 5.241 | unavailable | unavailable |
| `aten::reshape` | 2940 | 4.915 | 8.038 | unavailable | unavailable |
| `## Call CompiledFxGraph fp6vfnrengrg37jjoqnfkaszxxlshqogby4uuy5eutjhvcnqq3ua ##` | 56 | 4.860 | 7.246 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.4.self_attn.q_proj` | 8 | 4.704 | 4.821 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.27.mlp.up_proj` | 8 | 4.546 | 5.005 | unavailable | unavailable |
| `Torch-Compiled Region: 0/2` | 57 | 4.143 | 10.549 | unavailable | unavailable |
| `aten::copy_` | 58 | 3.678 | 3.678 | unavailable | unavailable |
| `## Call CompiledFxGraph fintnwvbeurtnqc423scnhreq5lm7h27cuym6zp2q52ydf57gzfi ##` | 57 | 3.661 | 5.314 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.23.self_attn.k_proj` | 8 | 3.389 | 3.679 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.22.mlp.up_proj` | 8 | 3.346 | 3.464 | unavailable | unavailable |
| `aten::view` | 3136 | 3.335 | 3.335 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.27.self_attn.k_proj` | 8 | 3.264 | 4.774 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.9.self_attn.v_proj` | 8 | 3.155 | 3.856 | unavailable | unavailable |

## 输出文件

- Chrome trace：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-210656-776045/trace.json`
- 原始 JSON：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-210656-776045/summary.json`
- CUDA device activity：`缺失（0 个 device events）`
