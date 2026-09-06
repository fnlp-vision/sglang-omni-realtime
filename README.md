# SGLang-Omni Realtime for MOSS-VL

**基于 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 开发的 MOSS-VL 实时视频理解推理后端。**

本仓库在 SGLang-Omni 的执行框架、请求调度和服务接口基础上，增加 MOSS-VL Realtime 的持续视频输入、增量视觉 KV、文本换轮、静默挂起/唤醒及多会话推理能力。底层模型执行依赖 [SGLang](https://github.com/sgl-project/sglang)，不是独立重写的推理引擎，也不是 SGLang-Omni 官方发行版。

客户端通过 WebSocket 持续提交带时间戳的图像帧和文本提示。每条连接对应一个持续存活的推理会话，模型可以主动输出文本、进入静默，并在收到后续输入时继续推理，而不是对每帧重新发起一次独立问答。

## 项目来源与仓库关系

| 仓库 | 角色 | 本地远端约定 |
| --- | --- | --- |
| [fnlp-vision/sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) | 当前 MOSS-VL Realtime 特化版本，开发与发布分支为 `main` | `origin` |
| [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) | 官方上游，提供 SGLang-Omni 通用框架 | `upstream` |
| [CloudRipple/sglang-omni](https://github.com/CloudRipple/sglang-omni) | 原开发环境使用的 fork，保留作历史来源与代码比较；不是官方上游 | `cloudripple` |
| [fnlp-vision/MOSS-VL-Realtime_Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo) | 独立的 Demo、会话编排及薄网关仓库 | 不属于本仓库 |

本仓库保留原有 Git 历史和上游授权信息。此前本地分支名为 `moss-vl-realtime`，现以本仓库的 `main` 维护；这不代表特化改动已合入官方上游或旧 fork。特化版本问题请提交到[本仓库 Issues](https://github.com/fnlp-vision/sglang-omni-realtime/issues)。

新克隆只会自动配置 `origin`。需要比较上游时，可另外添加远端；已有同名远端时先用 `git remote -v` 核对，不要重复添加：

```bash
git clone https://github.com/fnlp-vision/sglang-omni-realtime.git
cd sglang-omni-realtime
git remote add upstream https://github.com/sgl-project/sglang-omni.git
git remote add cloudripple https://github.com/CloudRipple/sglang-omni.git
git remote -v
```

不要用官方上游或旧 fork 的文件直接覆盖本仓库。合入其更新前，应检查实时请求调度、KV 生命周期和协议行为，并运行相关回归测试。

## 后端能力与职责边界

| 能力 | 当前实现 |
| --- | --- |
| 持续视频输入 | JPEG / PNG / WebP 二进制帧，显式时间戳，增量视觉前向与 KV 提交 |
| 文本换轮 | `input.prompt` 或带 prompt 的帧在同一会话内追加新问题，保留已有上下文 |
| 主动输出与静默 | 流式文本、`response.turn.silence`、静默挂起及后续输入唤醒 |
| 多会话 | 单实例支持多个独立会话；`--max-running-requests` 控制上限，parked 会话仍占 slot |
| 推理执行 | 单卡或 TP 多卡；decode CUDA Graph 默认开启，async decode 可选 |
| 输入背压 | 有界输入队列、ready/accepted/processed 确认，不提供无限缓冲 |
| 上下文观测 | 协商后通过 `session.usage` 报告历史位置空间和现存视觉 KV 占用 |
| 资源管理 | KV 容量预检、压力下终止会话、调度线程内串行处理外部 abort |
| 视觉 KV 滑窗 | 可选 raw 帧淘汰；pooling 是独立的实验开关，默认关闭 |

典型部署关系：

```text
直接协议客户端 / Demo / 平台薄网关
    -> WebSocket /v1/video/realtime
    -> SGLang-Omni 服务与请求生命周期
    -> MOSS-VL Realtime scheduler / model step / vision KV
    -> SGLang 模型执行与 CUDA
```

本仓库提供的是**推理后端**。Demo 的浏览器界面、ASR/TTS 编排、memory rollover，以及薄网关的 REST 创建/reset、一次性 ws_token、实例池路由，均在 [Demo 仓库](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo)维护。客户鉴权、额度、计费和公网 TLS 需由部署平台完成，不能直接将本后端当作已完成这些能力的公网产品。

上游其他模型和通用模块仍保留在源码中，但本特化分支的验证重点是 MOSS-VL Realtime；不能将上游全部模型、硬件的支持声明当作本分支的重新验收结论。

## 安装与模型准备

### 环境

当前 MOSS-VL Realtime 路径主要在 Linux / NVIDIA CUDA 环境验证，已有 H200 单卡与 TP2 测试。NPU、XPU、CPU 的完整实时模型推理不在本分支已验证范围内。

以当前 [pyproject.toml](./pyproject.toml) 为依赖依据：Python 支持范围为 `>=3.10,<3.13`，已有验证环境使用 Python 3.12；关键依赖包含 SGLang `0.5.16`、Transformers `5.12.1`、PyTorch `2.11.0`、FlashInfer `0.6.14`。依赖中含 CUDA 13 相关 wheel，需匹配驱动、CUDA 和编译工具链；这不是适用于任意 CUDA 环境的通用安装组合。

在准备好兼容 CUDA 环境并安装 `uv` 后，从本仓库根目录执行：

```bash
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -e .
```

必须安装**本仓库源码**。Python 包名仍为 `sglang-omni`，直接执行 `pip install sglang-omni` 安装的是公开发行包，不能保证包含本仓库的特化代码。原有[安装文档](./docs/get_started/installation.md)可用于了解基础依赖，但其中官方 PyPI、Docker 和克隆地址不应替代本仓库安装步骤。未在本次 README 更新中重新验证全新机器安装。

### 模型

使用与实时推理协议匹配的 MOSS-VL Realtime checkpoint，并将权重放在源码目录之外。模型信息可参考 [OpenMOSS-Team/MOSS-VL-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime)。

模型目录应提供匹配的 tokenizer、processor、配置和所需自定义模型代码。普通离线视频模型即使能够加载，也不代表具备训练过的时间戳、静默或主动响应语义。

本 Git 仓库**不包含权重，也不自动更新 checkpoint 目录中的 TF/HF 参考推理代码**。进行 HF/SGLang 对齐时，应单独固定模型、processor、TF 参考代码及本后端的版本，不能仅凭目录名或“成功加载”认定数值/语义一致。

## 快速启动

以下命令在仓库根目录、已激活环境中执行。将 `MODEL_PATH` 改为实际 checkpoint 路径。

### 单卡

```bash
export MODEL_PATH=/path/to/moss-vl-realtime-checkpoint

python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --gpu 0 \
  --host 127.0.0.1 \
  --port 8000 \
  --context-length 131072 \
  --mem-fraction-static 0.60 \
  --max-running-requests 1
```

这里显式使用 128K context 和 0.60 内存比例作为起步示例，**不是所有 GPU 都能运行的容量保证**。启动器代码默认值实际为 256K context、0.40 内存比例、1 个会话；这组默认值也不保证适配所有设备。启动会检查 KV 池是否能容纳一个完整 context；失败时应在可用显存范围内调整内存比例，或降低 context。多会话更不能只按 slot 数推断可用容量。

默认启动预热会通过真实帧输入路径执行一次内部请求，然后释放资源。首次加载和内核 JIT 可能需要较长时间；不要仅凭进程已创建就认为服务就绪。

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

### TP 多卡与并发

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --tp-size 2 --gpus 0,1 \
  --host 127.0.0.1 --port 8000 \
  --context-length 131072 \
  --mem-fraction-static 0.60 \
  --max-running-requests 2
```

`--gpus` 必须恰好列出 `--tp-size` 个不重复设备。TP>1 时由入口 rank 解析帧并向其他 rank 广播，各 rank 保持相同的请求更新与执行顺序。上例的 2 个会话只是配置示例，不是吞吐或 SLA 承诺。

decode CUDA Graph 默认使用 FlashInfer；可用 `--disable-decode-cuda-graph` 切换 eager decode，或用 `--enable-async-decode` 显式启用异步 decode。完整参数以 [启动器](./examples/run_moss_vl_realtime_server.py)及其 `--help` 为准，不要为普通服务启用 `--enable-benchmark-mode`。

### 发送视频帧

准备两张实际图像，运行随仓库提供的客户端：

```bash
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:8000/v1/video/realtime \
  --prompt "Describe relevant changes." \
  --fps 1 \
  --frame /path/to/frame_000.jpg --timestamp 0.0 \
  --frame /path/to/frame_001.jpg --timestamp 1.0
```

示例客户端会将最后一帧标记为 `final`，适合有限输入的烟测；它不是完整的摄像头 Demo。实时语义的已有验证基线为 1 FPS，其他输入速率被接口接受不代表已经完成模型质量验证，也不代表后端实现了业务 FPS 限流。

## 实时协议要点

后端入口是 `ws://HOST:PORT/v1/video/realtime`，不是 Demo 薄网关的 `/v1/realtime`。主要流程如下，模型输出事件可以穿插到输入确认过程之中：

```text
server -> session.created
client -> session.configure
server -> session.configured / session.ready
client -> input.frame (JSON metadata)
server -> input.frame.ready
client -> binary image bytes
server -> input.frame.accepted
server -> input.frame.processed
server -> response.text.delta / response.turn.silence / ...
server -> response.done / session.done (会话终结)
```

| 约定 | 含义 |
| --- | --- |
| `seq_no` | 帧和纯文本 prompt 共用从 0 开始的严格连续序号 |
| `timestamp` | 视频时间，单位秒，要求非递减；允许相同时间戳 |
| `accepted` / `processed` | 前者表示已接收提交，后者表示模型已消费；不能混用 |
| `input.prompt` | 在同一会话内追加新问题，由 `response.turn.interrupted` 确认换轮 |
| `response.turn.silence` | 模型静默，不是会话已结束 |
| `response.done` / `session.done` | 会话级结束，不能当作每段主动回答的结束；`turn_id` 也不等于独立回答 ID |
| `session.abort` | 终止整个会话，不是只暂停当前语音或打断一段回答 |
| `max_tokens_per_turn` | 历史命名，实际是 token/秒的生成速率目标；并发下为软目标 |
| `max_new_tokens` | 每次输入 extend 后重新锚定的 decode 余量，不是整个会话总输出预算 |
| `input_queue_capacity` | 待处理输入事件容量，默认 4，范围 1-256；不是累计帧数上限 |

`session.created.capabilities` 通告 `session.usage` 能力，客户端可在 `session.configure` 中设置 `include_usage: true` 开启统计。事件包含 `decoder_tokens`、`encoder_tokens`、`encoder_kv_tokens`、`token_space_used`、`context_limit`、`context_remaining`。历史 token 位置空间与实际保留的视觉 KV 是不同量；该统计不是客户计费账单。

客户端必须遵守 ready/accepted 背压并读取 `session.configured.max_frame_bytes`；收到输入错误或 ACK 超时，不应盲目重放状态不确定的输入。客户端输出应按 `turn_id` 和换轮事件处理旧回答；当前协议没有替换/撤回已发送历史文本的事件。

更多细节见[Realtime Cookbook](./docs/cookbook/moss_vl_realtime.md)、[协议实现与字段模型](./sglang_omni/serve/video_realtime.py)和[示例客户端](./examples/moss_vl_realtime_client.py)。

## 长会话、滑窗与容量

滑窗总开关默认关闭；开启后，可按 raw 时间窗口淘汰历史视觉 KV。**pooling 是另一个开关，默认关闭，不会因为开启滑窗就自动启用。** 例如在启动服务前设置：

```bash
export REALTIME_FRAME_WINDOW_ENABLED=1
export REALTIME_FRAME_WINDOW_RAW_S=60
export REALTIME_FRAME_POOLING_ENABLED=0
```

60 秒只是配置示例，不是默认产品时长或模型能力承诺。滑窗淘汰会改变可见视觉历史，pooling 还会引入额外近似；二者都不能声明与完整历史 KV 的参考推理严格等价。pooling 仅作为实验功能保留。

滑窗回收的是 KV 内存，**不代表无限上下文**。历史 token 位置仍受 context 上限约束；Demo 的 memory rollover 也不是本后端提供的透明续接能力。

实例满额时新连接收到 `session_capacity_exceeded` 并 close 1013。运行中 KV 不足时会按资源占用选择会话终止，客户端需要显式重建；这不是无损迁移。静默挂起还受 `--parked-request-timeout` 约束，代码默认 300 秒，与最大连接总时长不同。

## 验证与已知边界

基础回归可在仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/pipeline/test_async_decode.py
```

2026-09-06 在后端提交 [`e1b5fcf`](https://github.com/fnlp-vision/sglang-omni-realtime/commit/e1b5fcfa22fe0eec818c50d61894b69a1848f658) 对应源码上，该命令验证了 **268 项通过、5 项跳过**：4 项依赖 CUDA，1 项需显式设置模型路径。模型步进与 processor 对齐测试见 [test_model_step.py](./tests/unit_test/moss_vl_realtime/test_model_step.py) 和 [test_processor_parity.py](./tests/unit_test/moss_vl_realtime/test_processor_parity.py)；运行这些测试需另外提供 GPU 和匹配的模型/测试输入。

已有真实模型单卡、TP2、会话打断与视觉 KV 回收测试，但不能据此推导任意负载下的稳定并发或生产 SLA。历史高压长短会话混跑曾出现短会话完成超时，最终发布配置下的高压复测仍待闭环。正式部署还需验证目标分辨率、FPS、会话时长、并发和平台全链路。

## 代码导航

| 入口 | 用途 |
| --- | --- |
| [sglang_omni/models/moss_vl_realtime/](./sglang_omni/models/moss_vl_realtime/) | 模型接入、增量输入、runtime state、scheduler 与 frame window |
| [sglang_omni/serve/video_realtime.py](./sglang_omni/serve/video_realtime.py) | WebSocket 字段校验、事件转换、背压与连接生命周期 |
| [sglang_omni/scheduling/](./sglang_omni/scheduling/) | 复用并扩展的通用调度基础设施 |
| [examples/run_moss_vl_realtime_server.py](./examples/run_moss_vl_realtime_server.py) | 单卡/TP 服务启动器 |
| [examples/moss_vl_realtime_client.py](./examples/moss_vl_realtime_client.py) | 二进制帧和 manifest 协议客户端 |
| [scripts/](./scripts/) | `moss_vl_realtime_*` 推理、参考对齐和压测工具 |
| [tests/unit_test/moss_vl_realtime/](./tests/unit_test/moss_vl_realtime/) | 模型接入与调度回归 |

## 致谢与许可证

本项目基于 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 开发，感谢其开发者与社区提供执行框架、调度、通信和模型接入基础设施，同时感谢 [SGLang](https://github.com/sgl-project/sglang) 与 MOSS-VL 模型团队。原开发 fork [CloudRipple/sglang-omni](https://github.com/CloudRipple/sglang-omni) 保留为来源链接。

代码沿用 [Apache License 2.0](./LICENSE)，保留上游版权及许可证信息。模型权重、数据和第三方依赖分别遵循其自身许可证。通用框架资料见 [SGLang-Omni 官方文档](https://sgl-project.github.io/sglang-omni/)，本仓库的特化行为以这里的源码、测试和 Realtime 文档为准。
