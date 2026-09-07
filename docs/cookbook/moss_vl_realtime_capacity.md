# MOSS-VL Realtime 并发容量规划

## 部署配置

包含 Demo memory 的长时视频服务，参考配置为每实例 4 个会话。
后端设置 `--max-running-requests 4`，Demo 设置
`SGLANG_OMNI_SESSIONS_PER_REPLICA=4`，两端 context 均为 131,072。
每路输入 1 FPS、最长边 512，视觉 raw window 为 60 秒，pooling 关闭，
并启用 memory rollover。更高并发应同时评估 GPU、CPU 和 memory 队列容量。

## 测量条件

2026-09-07 的隔离测试使用单张 H200（143,771 MiB），context 131,072，
`--mem-fraction-static 0.60`，CUDA Graph 开启，async decode 关闭。
实测 KV 池为 336,874 slots。视觉 raw window 为 60 秒，pooling 关闭；
每路输入 1 FPS，最长边 512，JPEG quality 60，生成速率目标 4 tokens/s。

完整 Demo 开启文本和图像 memory，使用 CPU BGE-M3、Chinese-CLIP、
96 个 Torch 线程、独立会话存储，以及 Pi 对接的独立 Qwen3-4B GPU 服务。
rollover 阈值为 idle 8K / hard 12K；ASR、TTS 未开启。
采样使用 Demo 默认 temperature 0.7、top_p 0.8；纯 VLM 测试使用 greedy。

完整 memory 链路的四会话配置完成了 20 分钟测试；六会话与八会话的
短测存在写入积压。实例扩容应以整条服务链路的持续处理能力为依据。

## 容量参考

| 链路 | 并发 | 时长 | 结果 |
| --- | ---: | ---: | --- |
| 完整 memory | 4 | 150 秒 | 600/600 帧转发成功，4 次同时触发的 rollover 完成，写入队列归零 |
| 完整 memory | 4 | 20 分钟 | 4,795/4,800 帧转发成功，每路两次自然 rollover，写入丢弃 0、队列 0 |
| 完整 memory | 6 | 180 秒 | 1,077/1,080 帧转发成功，6 次 rollover 完成，结束时写入队列积压 231 项 |
| 完整 memory | 8 | 150 秒 | 1,187/1,200 帧转发成功，8 次 rollover 完成，结束时写入队列积压 366 项 |
| 纯 VLM，512 最长边 | 8 / 12 / 16 / 24 / 28 | 每档 90 秒 | 所有输入帧处理完成 |
| 纯 VLM，512 最长边 | 30 | 90 秒 | KV 容量保护终止 1 路，其余继续运行 |
| 纯 VLM，512 最长边 | 32 | 90 秒 | KV 容量保护终止 2 路，其余继续运行 |
| 纯 VLM，1024 x 576 | 4 / 8 | 每档 90 秒 | 360 / 720 帧全部处理完成 |

以上测试关闭会话后 KV 均完全回收。4 路长测的峰值为 71,443 slots；
28 路纯 VLM 短测达到 316,032 slots（池的 93.8%），不能作为长期部署上限。
1024 x 576 的 8 路短测达到 298,503 slots，分辨率同样需要计入容量预算。
完整 memory 长测结束时，4 路都能回答各自预设的会话标记，未检测到其他
会话的标记出现在输出或该会话的 memory 中。

这里的帧转发数是 Demo 输入侧统计；模型接收数还包含问题附带帧和
rollover 恢复的最新帧。4 路长测模型共接收并处理 4,967 帧。
输入帧转发率、模型处理率和关闭后的 KV 回收分别统计。

## 资源规划

- 视觉滑窗回收老帧的物理视觉 KV，但不清除历史 context 位置。
- 文本 decoder KV 仍会增长，需要通过 memory rollover 重建上下文。
- memory 编码、写入、检索和摘要另有 CPU、队列和服务容量成本。
- 同时 rollover 会带来摘要与重新 prefill 的瞬时负载，应预留余量。

扩容时需一起调整后端 `--max-running-requests` 和 Demo 对应实例的 slot
配置，并在实际视频尺寸、问答频率和 memory 配置下复测。90 秒纯 VLM
结果不代表多小时、完整音视频链路或任意输入下的最大并发。

Demo 与后端独立部署，版本和进程生命周期分别管理。每路实际 KV、
共享权重、80GB 级别显存预算和连接池观测字段见
[VLM 显存与并发配置](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/docs/vlm_memory_capacity.md)。
协议与验证入口见 [Realtime Cookbook](./moss_vl_realtime.md#validation)。
