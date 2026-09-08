# MOSS-VL Realtime 测试

## 环境

使用已安装本仓库依赖的 Python 3.12 / Transformers 5.12.1 后端环境，
准备一张 80 GB 级别及以上显存的空闲 GPU，以及本地
[MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) 模型目录。

以下命令在仓库根目录执行。模型路径是唯一必填参数，测试样例已内置，无需额外准备。

## 三项测试

| 入口 | 测试内容 | 默认配置 |
| --- | --- | --- |
| [test_accuracy.sh](./test_accuracy.sh) | HF/SGLang 任务结果、token 和逐事件输出对齐，以及各后端单/多路一致性 | 1、2、4 路，自然生成 |
| [test_latency.sh](./test_latency.sh) | HF/SGLang 单路 prefill、帧处理、TTFT、TPOT 和吞吐 | 固定输入与 64 个生成 token，不限速，预热后计时 5 轮 |
| [test_concurrency.sh](./test_concurrency.sh) | SGLang 并发对每一路时延和速度的影响 | 1、2、4、8 路；每路 1 FPS、10 token/s，重复 3 轮 |

精度测试按受控输入顺序比较输出；多路时延测试中，每路独立按时间表送帧、提问和接收，
不等待其他会话，也不等待回答结束。输出为自然生成，10 token/s 是配置目标，实际速度单独测量。

默认使用 BF16、HF eager、SGLang FlashInfer + decode CUDA Graph；
测试关闭视觉滑窗、pooling 和 async decode。时延为后端进程内测量，不包含网络、网关或 Demo。

## 使用

```bash
bash deployment/moss_vl_realtime/test_accuracy.sh /path/to/model
bash deployment/moss_vl_realtime/test_latency.sh /path/to/model
bash deployment/moss_vl_realtime/test_concurrency.sh /path/to/model
```

默认自动选择一张空闲卡。HF/SGLang 对照在同一卡上顺序执行，多路测试的全部会话也在同一卡上。

```bash
# 指定 GPU，并调整多路实验参数
bash deployment/moss_vl_realtime/test_concurrency.sh /path/to/model \
  --gpus 2 --sessions 1 2 4 8 --fps 1 --token-rate 10 --repeats 3
```

- 通用：`--gpus` 指定设备，`--output-dir` 指定新的结果目录，`--dry-run` 检查配置。
- 精度：`--strict-tokens` 将 token/事件不一致视为失败。
- 时延：`--repeats` 调整重复次数；多路另支持 `--sessions`、`--fps`、`--token-rate`。
- HF/SGLang 对照可用 `--gpus 2,3` 并行加速；正式速度比较优先使用同一卡。多路测试只接受一张卡。

其余选项见各入口的 `--help`。`PYTHON=/path/to/python` 可指定解释器。
测试使用本地模型，不使用代理，不停止已有服务。

## 结果

报告分别写入 `results/accuracy/`、`results/latency/`、`results/concurrency/` 下的独立运行目录。
每次保留 Markdown 报告、`summary.json`、原始输出、运行参数和日志。

`PASS` 表示对应自动检查通过；`FAIL` 表示执行或检查失败；
`REVIEW` 表示需要核对输出差异。多路报告额外记录逐路指标、发送迟到、待处理事件和未响应情况。

## 参考结果

2026-09-08，NVIDIA H200 单卡，BF16；PyTorch 2.11.0、Transformers 5.12.1、
SGLang 0.5.16、FlashInfer 0.6.14。以下时间单位均为毫秒。

### 精度对齐

| Sessions | HF 任务通过 | SGLang 任务通过 | 跨后端 token 与事件一致 | 两端各自与单路一致 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4/4 | 4/4 | 4/4 | 4/4 |
| 2 | 4/4 | 4/4 | 4/4 | 4/4 |
| 4 | 4/4 | 4/4 | 4/4 | 4/4 |

共 12 组配对，严格对齐检查全部通过。

### 单路时延

| 指标 | HF 均值 / P95 | SGLang 均值 / P95 | 均值加速比 |
| --- | --- | --- | ---: |
| Initial prefill | 38.78 / 48.93 | 36.68 / 56.57 | 1.06x |
| Frame extend | 117.51 / 218.47 | 81.42 / 133.30 | 1.44x |
| Prompt TTFT | 69.57 / 114.15 | 40.34 / 65.63 | 1.72x |
| TPOT | 50.31 / 62.97 | 9.28 / 13.58 | 5.42x |

Decode 吞吐：HF **19.88 token/s**，SGLang **107.73 token/s**。

### 多路时延

| Sessions | 帧处理 P95 | TTFT 均值 / P95 | TPOT 均值 / P95 | 每路平均 token/s | 最慢一路 token/s |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 144.01 | 515.10 / 910.30 | 99.26 / 123.92 | 10.08 | 10.08 |
| 2 | 235.95 | 568.50 / 1030.32 | 104.59 / 152.41 | 9.56 | 9.53 |
| 4 | 276.04 | 574.35 / 1074.18 | 116.69 / 285.79 | 8.54 | 8.38 |
| 8 | 391.46 | 728.29 / 1270.60 | 145.97 / 476.11 | 6.79 | 5.48 |

多路 TTFT 以首段可见文本为终点；TPOT 和 token/s 按连续回答期间计算，排除静默期。
该表使用 10 token/s 配置下重新测量的单路基线，不与上面的不限速固定 token 基准混比。

本次全部输入处理完成、全部问题有可见回答，无执行异常。4 路的平均 token 间隔比单路增加约 18%，
8 路增加约 47%；并发增加会降低单路速度并放大尾部时延。这是当前测试负载的性能参考，不是语义质量结论。

原始运行记录保留在对应结果目录：精度 `20260908T063856Z-fc36c9`、
单路时延 `20260908T061633Z-e6f223`、多路时延 `20260908T073150Z-4ddd3d`。

## 推理启动

```bash
bash deployment/moss_vl_realtime/start.sh /path/to/model
```

默认单卡、4 个 session，服务地址 `http://127.0.0.1:18500`；
健康检查 `GET /health`，实时接口 `WS /v1/video/realtime`。
可选 `--gpus 2 --port 18510`，其余默认值见 [config.json](./config.json)。
