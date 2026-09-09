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
| [test_latency.sh](./test_latency.sh) | HF/SGLang 不限速单路基准：prefill、帧处理、首 token、TPOT 和吞吐 | 固定输入与 64 个生成 token，预热后计时 5 轮 |
| [test_concurrency.sh](./test_concurrency.sh) | 实时单/多路场景：首段可见文本时延、token 间隔和速度 | 1、2、4、8 路；每路 1 FPS、10 token/s，重复 3 轮 |

精度测试按受控输入顺序比较输出；多路时延测试中，每路独立按时间表送帧、提问和接收，
不等待其他会话，也不等待回答结束。输出为自然生成，10 token/s 是配置目标，实际速度单独测量。

默认使用 BF16、HF eager、SGLang FlashInfer + decode CUDA Graph；
启动与 SGLang 测试共用 `config.json` 中的 `mem_fraction_static=0.5`。
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
显存使用 NVML 每 50 ms 采样，覆盖模型加载和测试；`*_memory.json` 保留字节值与峰值。
80 GB 为容量参考目标，不是测试通过的硬门槛；显存峰值单独展示，执行、输出和采样异常仍按原规则检查。

`PASS` 表示对应自动检查通过；`FAIL` 表示执行或检查失败；
`REVIEW` 表示需要核对输出差异。多路报告额外记录逐路指标、发送迟到、待处理事件和未响应情况。

## 参考结果

2026-09-08，NVIDIA H200 单卡，BF16；PyTorch 2.11.0、Transformers 5.12.1、
SGLang 0.5.16、FlashInfer 0.6.14，SGLang `mem_fraction_static=0.5`。以下时间单位均为毫秒。

### 精度对齐

结果：**PASS**。

| Sessions | HF 任务通过 | SGLang 任务通过 | 跨后端 token 与事件一致 | 两端各自与单路一致 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4/4 | 4/4 | 4/4 | 4/4 |
| 2 | 4/4 | 4/4 | 4/4 | 4/4 |
| 4 | 4/4 | 4/4 | 4/4 | 4/4 |

共 12 组配对，任务结果、token、逐事件输出及单/多路一致性检查全部通过。

### 单路基准（不限速）

结果：**PASS**。

| 指标 | HF 均值 / P95 | SGLang 均值 / P95 | 均值加速比 |
| --- | --- | --- | ---: |
| Initial prefill | 45.58 / 49.30 | 42.62 / 55.71 | 1.07x |
| Frame extend | 106.99 / 218.37 | 88.95 / 129.57 | 1.20x |
| Prompt TTFT | 91.77 / 124.42 | 33.57 / 35.17 | 2.73x |
| TPOT | 52.05 / 61.73 | 11.08 / 26.97 | 4.70x |

Decode 吞吐：HF **19.21 token/s**，SGLang **90.26 token/s**。

### 实时多路时延（10 token/s）

结果：**PASS**。

| Sessions | 帧处理 P95 | TTFT 均值 / P95 | TPOT 均值 / P95 | 每路平均 token/s | 最慢一路 token/s |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 142.56 | 518.70 / 910.41 | 99.21 / 128.02 | 10.08 | 10.08 |
| 2 | 222.93 | 532.85 / 1025.32 | 104.23 / 152.86 | 9.59 | 9.58 |
| 4 | 298.19 | 625.84 / 1135.57 | 114.13 / 235.76 | 8.76 | 8.65 |
| 8 | 385.91 | 794.75 / 1315.16 | 148.31 / 490.91 | 6.71 | 6.03 |

多路 TTFT 以首段可见文本为终点；TPOT 和 token/s 按连续回答期间计算，排除静默期。
该表使用 10 token/s 配置下重新测量的单路基线，不与上面的不限速固定 token 基准混比。
10 token/s 对应约 100 ms 的 token 间隔；不限速基准中的 11.08 ms 衡量另一种运行条件。
实时 TTFT 还包含调度、静默及等待后续帧的时间；本次单路每轮首次提问约 0.88–0.91 秒出现文字，
第二次约 0.14–0.15 秒，平均约 0.52 秒。并发影响应在本表内比较 1、2、4、8 路。

本次全部输入处理完成、全部问题有可见回答，无执行异常。4 路的平均 token 间隔比单路增加约 15%，
8 路增加约 49%。

三项测试的显存采样峰值如下；整卡列包含驱动及既有基础占用。

| 实验 | VLM 进程峰值 GiB | 整卡峰值 GiB | 整卡峰值 GB |
| --- | ---: | ---: | ---: |
| 精度对齐 | 74.779 | 76.093 | 81.705 |
| 单路时延 | 71.014 | 72.328 | 77.661 |
| 独立多路时延 | 72.879 | 74.193 | 79.664 |

`1 GiB=2^30 bytes`，`1 GB=10^9 bytes`。显存约为 80 GB 级别，不影响三项测试的通过结论。

## 推理启动

```bash
bash deployment/moss_vl_realtime/start.sh /path/to/model
```

默认单卡、4 个 session，服务地址 `http://127.0.0.1:18500`；
健康检查 `GET /health`，实时接口 `WS /v1/video/realtime`。
可选 `--gpus 2 --port 18510`，其余默认值见 [config.json](./config.json)。
