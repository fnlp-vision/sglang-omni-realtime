# MOSS-VL Realtime 启动与测试

本目录提供单卡推理启动和 TF/SGLang 测试入口。模型目录是唯一必填参数；
GPU、端口和输出目录可按部署环境选填，其余默认参数位于 [config.json](./config.json)。

## 环境准备

使用 Linux、NVIDIA GPU、Python 3.12，以及本仓库的独立后端环境。
默认 128K context 配置面向 80GB 级别及以上显存的空闲单卡。
测试不需要 Demo 环境、额外数据集或在线下载测试素材。

```bash
git clone https://github.com/fnlp-vision/sglang-omni-realtime.git
cd sglang-omni-realtime
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -e .

hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --revision bcfd9ccf1e9db2896ad852301cc8dde4a6349c78 \
  --local-dir /path/to/model
```

模型访问需要获授权的 Hugging Face 账号。启动和测试从本地目录读取完整模型，
不自动下载模型或依赖，不使用代理。`PYTHON` 可选指定解释器；已激活环境时
使用该环境，否则优先使用仓库 `.venv`。

## 推理启动

```bash
bash deployment/moss_vl_realtime/start.sh /path/to/model
```

默认选择一张空闲 GPU，以单卡、四个会话 slot 启动前台服务：

| 项目 | 默认配置 |
| --- | --- |
| HTTP 基础地址 | `http://127.0.0.1:18500` |
| 健康检查 | `GET /health` |
| 模型信息 | `GET /v1/models` |
| 实时协议 | `WS /v1/video/realtime` |
| Context / 静态显存比例 | `131072` / `0.60` |
| 最大会话数 | `4`，同一 GPU 上的独立会话 |
| 视觉 KV 窗口 | 60 秒，pooling 关闭 |
| Decode | CUDA Graph 开启，async decode 关闭 |

生成参数由客户端 `session.configure` 指定。配置文件中的
`tokens_per_second` 和 `max_new_tokens` 用于测试客户端，不替代协议默认值。

指定单卡和端口：

```bash
bash deployment/moss_vl_realtime/start.sh /path/to/model --gpus 2 --port 18510
```

端口占用时启动失败，不关闭已有服务，也不自动切换生产端口。
Ctrl-C 停止前台服务。默认仅监听本机；跨机器部署可指定 `--host`，并在
网络层配置访问控制。此入口不是鉴权网关，不能直接作为公网服务开放。

```bash
curl --fail http://127.0.0.1:18500/health
curl --fail http://127.0.0.1:18500/v1/models
```

配合 Demo 时设置 `SGLANG_OMNI_URLS=http://127.0.0.1:18500`、
`SGLANG_OMNI_SESSIONS_PER_REPLICA=4` 和 `SGLANG_OMNI_CONTEXT_LENGTH=131072`。
长会话的文本 memory 与 rollover 由 Demo 提供，参见
[VLM 显存与并发配置](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/docs/vlm_memory_capacity.md)。

## 一键测试

### 默认单卡

```bash
bash deployment/moss_vl_realtime/test.sh /path/to/model
```

自动选择一张空闲 GPU，先运行 TF，释放模型进程后再在同一张卡上运行
SGLang。单卡执行完整矩阵，不省略多 session 用例，也不同时加载两份模型。
默认测试流程如下：

| 后端 | 会话数 | 用例与执行方式 |
| --- | --- | --- |
| TF / Transformers 5.12.1 | 1、2、4 | 共享一份模型、每路独立 KV，显式轮询执行 |
| SGLang-Omni | 1、2、4 | 每路独立 WebSocket，同卡连续批处理 |
| SGLang-Omni 视觉窗口 | 4 | 每路 75 帧、1 FPS，观察视觉淘汰与关闭后的 KV 回收 |

功能用例使用仓库中的 [cars.jpg](../../tests/data/cars.jpg) 和
[draw.mp4](../../tests/data/draw.mp4)，统一缩放和编码后供两后端使用。
TF/SGLang 对照用例每路 12 帧，不跨越 60 秒视觉窗口；窗口用例单独测试
SGLang。TF 多会话为独立状态轮询，不是连续批处理服务。该脚本不加载
Demo memory、摘要模型、ASR 或 TTS。

### 可选双卡加速

```bash
bash deployment/moss_vl_realtime/test.sh /path/to/model --gpus 2,3
```

TF 放在第一张卡，SGLang 放在第二张卡，两后端并行运行各自完整矩阵。
每个后端仍使用单卡，四 session 仍位于同一张卡上；此选项不是 TP。
默认只需要一张卡，双卡仅用于减少测试总耗时。

`--gpus` 使用 `nvidia-smi` 的 GPU 编号或 UUID，并遵守已有的
`CUDA_VISIBLE_DEVICES` 限制。自动选择和手动选择都检查显存占用，不会
停止其他进程。若没有合适的空闲卡，脚本报错并退出。

### 测试结果

默认写入 `results/moss_vl_realtime/<时间戳>-<唯一标识>/`。可通过
`--output-dir /path/to/new-results` 指定尚不存在的输出目录。

| 文件 | 内容 |
| --- | --- |
| `report.md` | TF/SGLang 单、多 session 结果表与输出文本对照 |
| `summary.json` | 总体状态、环境版本及各路结果 |
| `tf.json` / `sglang.json` | 文本、事件、延迟和资源采样 |
| `allocator.jsonl` | SGLang KV 使用量、窗口状态和进程显存采样 |
| `metadata.json` / `cases.json` | 参数、代码版本、输入校验值与测试用例 |
| `tf.log` / `sglang.log` / `server.log` | 执行与模型加载日志 |

退出码 0 表示链路检查通过；加载失败、输入处理异常、无可见输出或 KV
回收失败均返回非零。测试服务使用动态本地端口，结束、异常和中断时清理
本次创建的进程，不复用已有在线实例。

结果表的 PASS 表示输入处理、可见输出和生命周期检查通过，不是语义
准确率。TF 使用固定步数驱动，SGLang 使用实时调度，输出文本供对照评估，
不要求逐 token 相同，也不以包含输入等待的会话耗时计算后端加速比。
TF 的同步 frame-step 延迟与 SGLang 的 ACK-to-processed 延迟分别保存。

查看参数配置和模型路径、不加载模型：

```bash
bash deployment/moss_vl_realtime/test.sh /path/to/model --dry-run
```

## 交付资源

| 项目 | 地址 |
| --- | --- |
| 推理启动脚本 | [start.sh](./start.sh) |
| 测试脚本 | [test.sh](./test.sh) |
| 测试实例与结果表 | [test_results.md](./test_results.md) |
| 历史容量参考 | [并发容量规划](../../docs/cookbook/moss_vl_realtime_capacity.md) |
| Demo 与网关代码 | [MOSS-VL-Realtime_Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo) |
| 模型下载 | [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) |

Demo 链接为代码仓库，模型链接为权重与配置仓库。容器镜像的 registry、
tag 和 digest 尚未提供，不能将上述地址作为 `docker pull` 地址使用。
