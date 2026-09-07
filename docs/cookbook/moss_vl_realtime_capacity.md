# MOSS-VL Realtime 并发容量测试

## 配置与结论

2026-09-07 的隔离测试使用单张 H200（143,771 MiB），context 131,072，
`--mem-fraction-static 0.60`，CUDA Graph 开启，async decode 关闭。
实测 KV 池为 336,874 slots。视觉 raw window 为 60 秒，pooling 关闭；
每路输入 1 FPS，最长边 512，JPEG quality 60，生成速率目标 4 tokens/s。

完整 Demo 开启文本和图像 memory，使用 CPU BGE-M3、Chinese-CLIP、
96 个 Torch 线程、独立会话存储，以及 Pi 对接的独立 Qwen3-4B GPU 服务。
rollover 阈值为 idle 8K / hard 12K；ASR、TTS 未开启。
采样使用 Demo 默认 temperature 0.7、top_p 0.8；纯 VLM 测试使用 greedy。

当前保留每实例 4 路配置。该配置完成了 20 分钟测试；6 路和 8 路完整 memory
短测出现明显写入积压，不能仅凭 VLM 显存剩余就提高 Demo 的会话上限。
测试没有修改或重启线上服务。

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
转发期间的少量丢帧和关闭时的 KV 回收是不同指标，不应混为一谈。

## 为什么滑窗不等于固定会话成本

- 视觉滑窗回收老帧的物理视觉 KV，但不清除历史 context 位置。
- 文本 decoder KV 仍会增长，需要通过 memory rollover 重建上下文。
- memory 编码、写入、检索和摘要另有 CPU、队列和服务容量成本。
- 同时 rollover 会带来摘要与重新 prefill 的瞬时负载，应预留余量。

扩容时需一起调整后端 `--max-running-requests` 和 Demo 对应实例的 slot
配置，并在实际视频尺寸、问答频率和 memory 配置下复测。90 秒纯 VLM
结果不代表多小时、完整音视频链路或任意输入下的最大并发。

测试发现的 Demo 握手预留计数问题已在
[Demo `215eabf`](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/commit/215eabf3912edb6eb6797dd5364480050d5e9ec8)
修复：探活和容量重试不再清除本地握手预留。部署时需要一起更新 Demo，
仅更新本后端不会替换 Demo 进程中的连接池代码。每路实际 KV、共享权重
和缩减显存预算测试见 [VLM 显存说明](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/docs/vlm_memory_capacity.md)。

## 本次调度修复

原实现先检查 decode 限速，再进入 SGLang 的调度规划。限速返回 `None`
时，刚完成的 prefill 还没有从 `last_batch` 合并进 `running_batch`；
事件循环随后清空 `last_batch`，导致会话仍占用连接和 KV，却不再被调度。
该问题在 KV 池基本空闲时也能复现，不是显存不足。

现在使用 SGLang 0.5.16 已有的 post-handoff prefill hook：先完成请求
交接，再处理视觉窗口、增量输入和新 prefill，最后决定是否延后 decode。
延后发生在 decode 分配本步 KV 之前，且保留合并后的 running batch。
模型计算、TF 实现、通用事件循环和安装目录中的 SGLang 均未修改。

验证包括 279 项 CPU 回归通过、5 项跳过，以及同步/异步的单卡
1/2/4 路、TP2 同步/异步 4 路 GPU 测试（每路 75 帧、4 tokens/s）。
GPU 正确性用例全部处理完输入，关闭后各 rank 的 KV 全部回收。
最初新增的 8 项回归在修复前均失败，修复后通过。
