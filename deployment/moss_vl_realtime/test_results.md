# 测试实例与结果

## 实例配置

| 项目 | 配置 |
| --- | --- |
| 测量日期 | 2026-09-07 |
| GPU | NVIDIA H200，单卡 143,771 MiB |
| 模型 | [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG)，BF16 |
| 环境 | PyTorch 2.11.0、Transformers 5.12.1、SGLang 0.5.16 |
| 单卡方式 | TF 与 SGLang 在同一张 GPU 上顺序执行 |
| 双卡方式 | TF 与 SGLang 各用一张 GPU 并行执行；每个后端仍为单卡 |
| TF 多 session | 共享模型、独立 KV、轮询执行 |
| SGLang 多 session | 同一实例的独立 WS 会话、连续批处理 |
| 功能用例 | 仓库车辆图片与绘图视频，每路 12 帧，1 FPS |
| 窗口用例 | SGLang 同卡 4 路，每路 75 帧，raw window 60 秒 |
| 附加模型 | 不加载 Demo memory、摘要模型、ASR 或 TTS |

## 单与多 Session 对照

表中为通过会话数 / 测试会话数。单 session 档分别运行两种素材，
因此包含两次独立的单 session 测试。通过条件包括输入处理、可见文本、
会话结束；SGLang 还检查 KV 全回收，窗口用例检查视觉淘汰。

| 后端 / 用例 | Session 数 | 单卡 A | 双卡并行 | 单卡 B |
| --- | ---: | ---: | ---: | ---: |
| TF 功能 | 1 | 2/2 | 2/2 | 2/2 |
| TF 功能 | 2 | 2/2 | 2/2 | 2/2 |
| TF 功能 | 4 | 4/4 | 4/4 | 4/4 |
| SGLang 功能 | 1 | 2/2 | 2/2 | 2/2 |
| SGLang 功能 | 2 | 2/2 | 2/2 | 2/2 |
| SGLang 功能 | 4 | 3/4 | 4/4 | 2/4 |
| SGLang 视觉窗口 | 4 | 4/4 | 4/4 | 4/4 |
| 合计 | | 19/20 | 20/20 | 18/20 |

单卡 A 的四会话短绘图用例有一路无可见文本，单卡 B 有两路无可见文本。
这些会话的帧和 prompt 均完成处理，正常结束且 KV 全回收；失败项为
`visible_output`。两次单卡流程均完整执行，并按判据返回非零退出码。
无输出原因尚未确定，不能将某一次全通过视为稳定回答质量的保证。
双卡模式的 SGLang 四 session 仍位于同一张卡上，不是通过 TP 扩容。

| 运行标识 | 方式 | 物理 GPU |
| --- | --- | --- |
| `20260907T132609Z-5db304` | 单卡 A | TF / SGLang 均为 GPU 2，顺序执行 |
| `20260907T133200Z-8ca9c7` | 双卡并行 | TF 为 GPU 4，SGLang 为 GPU 5 |
| `20260907T133754Z-e3e7cf` | 单卡 B | TF / SGLang 均为 GPU 2，顺序执行 |

## 显存样本

单卡 B 的进程显存采样如下；SGLang 数值包含按 H200 的静态比例 0.60
预分配的 KV 池，不是每路增量，也不是 80GB 卡上的池大小。

| 后端 / 阶段 | Session 数 | 进程采样峰值 GiB | SGLang 活跃 KV 合计峰值 GiB |
| --- | ---: | ---: | ---: |
| TF 功能 | 1 | 22.04 | 不适用 |
| TF 功能 | 2 | 22.16 | 不适用 |
| TF 功能 | 4 | 22.40 | 不适用 |
| SGLang 功能 | 1 | 84.70 | 0.36 |
| SGLang 功能 | 2 | 84.70 | 0.71 |
| SGLang 功能 | 4 | 85.00 | 1.42 |
| SGLang 视觉窗口 | 4 | 85.00 | 7.15 |

TF 使用动态缓存，SGLang 使用预分配池，两者的进程占用不能直接解释为
相同有效 KV 的存储效率。包含 memory rollover 的长时显存规划见
[VLM 显存与并发配置](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/docs/vlm_memory_capacity.md)。

## 启动入口验证

`start.sh` 在单张 H200 上启动四会话实例，健康检查和模型信息接口正常。
发送仓库车辆图片后收到可见回答；同端口二次启动被拒绝，原实例保持运行。
验证结束后，启动入口与测试入口创建的进程均已清理。

CPU 回归为 292 passed、5 skipped，其中一键入口相关测试为 13 项。
复现实例与生成完整输出表格的命令见 [启动与测试说明](./README.md)。
