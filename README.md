# SGLang-Omni Realtime for MOSS-VL

基于 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 开发的 MOSS-VL 实时视频理解推理后端，使用 [SGLang](https://github.com/sgl-project/sglang) 执行模型推理。

客户端通过 WebSocket 持续发送带时间戳的视频帧和文本问题。模型在同一会话内保留上下文，支持主动回答、静默等待和新问题打断，并可通过多会话调度与 TP 多卡部署服务多个客户端。

**配套模型：[OpenMOSS-Team/MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG)**，包含 Transformers 5.12.1 兼容代码、processor、tokenizer 和权重。

## 功能

- 持续输入 JPEG、PNG、WebP 图像帧，增量计算视觉特征与 KV。
- 在视频流中追加文本问题，保留上下文并切换回答轮次。
- 流式返回文本、静默和输入处理确认事件。
- 单实例多会话、单卡及 TP 多卡推理。
- Decode CUDA Graph、可选 async decode 和视觉 KV 滑窗。
- 有界输入队列、上下文用量统计及会话资源回收。

浏览器视频/语音交互、ASR/TTS、记忆编排和 REST 薄网关见 [MOSS-VL-Realtime_Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo)。

## 安装

使用 Linux、NVIDIA GPU 和 Python 3.12。当前依赖组合为 SGLang 0.5.16、Transformers 5.12.1、PyTorch 2.11.0、FlashInfer 0.6.14；CUDA 组件和编译环境要求见[安装指南](./docs/get_started/installation.md)。

在已安装 `uv` 的环境中执行：

```bash
git clone https://github.com/fnlp-vision/sglang-omni-realtime.git
cd sglang-omni-realtime
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -e .
```

Python 包名沿用 `sglang-omni`，请从本仓库安装以获得 MOSS-VL Realtime 的实现。Demo、TTS 引擎和本后端分别使用独立环境。

## 下载模型

```bash
hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --local-dir /path/to/MOSS-VL-Realtime-SGLANG
```

模型仓库目前需要访问授权；如提示无权访问，先用获授权的 Hugging Face 账号执行 `hf auth login`。下载时保留完整配置、自定义 Python 文件、tokenizer、processor 和五个权重分片。

模型包与原版使用相同权重，提供 Transformers 5.12.1 兼容实现。Transformers 4.57 系列参考实现见 [MOSS-VL-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime)。部署和实验可通过 `hf download --revision <commit>` 固定模型版本。

## 启动服务

### 单卡

以下示例使用 128K context、一个会话 slot：

```bash
export MODEL_PATH=/path/to/MOSS-VL-Realtime-SGLANG

python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --gpu 0 --host 127.0.0.1 --port 8000 \
  --context-length 131072 \
  --mem-fraction-static 0.60 \
  --max-running-requests 1
```

启动包含模型加载和一次图像预热。服务启动后检查：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

接口返回的 `model` 为服务名 `moss-vl-realtime`；实际加载的 checkpoint 由 `--model-path` 指定。

启动器的默认 context 为 256K，默认内存比例为 0.40。上面的命令显式覆盖这两个值；若日志提示 KV 池无法容纳一个完整 context，请降低 `--context-length`，或根据空闲显存提高 `--mem-fraction-static`。

### 多卡与并发

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --tp-size 2 --gpus 0,1 \
  --host 127.0.0.1 --port 8000 \
  --context-length 131072 \
  --mem-fraction-static 0.60 \
  --max-running-requests 2
```

`--gpus` 数量应与 `--tp-size` 一致且不重复。`--max-running-requests` 控制实例会话上限，静默挂起的会话也占用 slot；并发和 context 共同决定 KV 内存需求。

四会话长时视频服务可采用 128K context、60 秒视觉窗口和每路 1 FPS 输入，并配合 Demo memory rollover。纯 VLM 与包含 memory 的服务链路需要分别规划 GPU、CPU 和队列容量，配置与测量条件见 [并发容量规划](./docs/cookbook/moss_vl_realtime_capacity.md)。

Decode CUDA Graph 默认开启并使用 FlashInfer。`--disable-decode-cuda-graph` 切换到 eager decode，`--enable-async-decode` 开启异步 decode。完整参数可运行：

```bash
python examples/run_moss_vl_realtime_server.py --help
```

## 调用示例

准备两张实际视频帧，使用示例客户端按视频时间回放：

```bash
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:8000/v1/video/realtime \
  --prompt "Describe relevant changes." \
  --frame /path/to/frame_000.jpg --timestamp 0.0 \
  --frame /path/to/frame_001.jpg --timestamp 1.0
```

客户端将最后一帧标记为 `final`，随后等待会话完成。持续摄像头输入可使用 Demo 或自行实现客户端；中途帧保持 `final=false`，结束时提交最后一个输入或发送 `session.abort`。

### WebSocket 协议

入口：`ws://HOST:PORT/v1/video/realtime`。输入握手和模型输出在同一连接上进行：

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
server -> response.done / session.done
```

| 字段或事件 | 用法 |
| --- | --- |
| `seq_no` | 帧和 prompt 共用从 0 开始的连续序号 |
| `timestamp` | 视频时间，单位秒，非递减 |
| `input.prompt` | 追加新问题，由 `response.turn.interrupted` 确认换轮 |
| `input.frame.accepted` / `processed` | 分别表示输入已接收和已被模型消费 |
| `response.turn.silence` | 模型进入静默，后续输入可继续唤醒 |
| `response.done` / `session.done` | 会话结束；同一个 `turn_id` 内可以有多段主动回答 |
| `max_tokens_per_turn` | token/秒生成速率目标，并发时为软目标 |
| `max_new_tokens` | 每次输入 extend 后重新锚定的生成余量 |
| `input_queue_capacity` | 待处理输入事件容量，默认 4，可配置为 1-256 |

客户端应等待 `input.frame.ready` 后发送图像，遵守 `session.configured.max_frame_bytes`。新问题可在现有会话中追加；`session.abort` 会终止整个会话。

在 `session.configure` 中设置 `include_usage=true` 可订阅 `session.usage`，获取历史 token 位置、现存视觉 KV 和剩余 context。详细字段、事件顺序、错误处理及回放节奏见 [Realtime Cookbook](./docs/cookbook/moss_vl_realtime.md)。

## 长会话与视觉 KV

需要限制历史视觉 KV 时，在启动服务前开启 raw 滑窗：

```bash
export REALTIME_FRAME_WINDOW_ENABLED=1
export REALTIME_FRAME_WINDOW_RAW_S=60
export REALTIME_FRAME_POOLING_ENABLED=0
```

滑窗默认关闭。上述配置保留最近 60 秒的原始视觉 KV，并淘汰更早的帧。Pooling 是独立实验选项，默认关闭。

滑窗会改变模型可访问的视觉历史；回收 KV 后，历史 token 位置仍计入 context。需要跨 context 延续交互时，可使用 Demo 的 memory rollover。实例会话满额时返回 `session_capacity_exceeded`；运行中 KV 不足可能终止会话，客户端应处理错误并重新建立连接。

## 配合 Demo 使用

在 [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo) 的 `.env.deploy` 中设置：

```dotenv
VLM_DEPLOY=sglang_omni
SGLANG_OMNI_URLS=http://127.0.0.1:8000
SGLANG_OMNI_SESSIONS_PER_REPLICA=1
SGLANG_OMNI_CONTEXT_LENGTH=131072
MODEL_PATH=/path/to/MOSS-VL-Realtime-SGLANG
```

地址填写后端基础 URL，slot 和 context 与后端启动参数对应。两者运行在同一机器时，Demo API 应使用其他端口。浏览器使用 Demo 的 `/api/...` 协议；外部平台也可通过其 REST/WS 薄网关接入。具体启动、语音和记忆配置见 [Demo README](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo#readme)。

## 开发与测试

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/pipeline/test_async_decode.py
```

模型步进、processor 对齐及性能测试需要 GPU 和测试输入，见 [Cookbook 的验证说明](./docs/cookbook/moss_vl_realtime.md#validation)。

| 代码入口 | 内容 |
| --- | --- |
| [models/moss_vl_realtime](./sglang_omni/models/moss_vl_realtime/) | 模型接入、增量输入、调度、runtime state 和视觉 KV |
| [serve/video_realtime.py](./sglang_omni/serve/video_realtime.py) | WebSocket 协议、输入背压与连接生命周期 |
| [examples](./examples/) | 启动器与帧回放客户端 |
| [tests/unit_test/moss_vl_realtime](./tests/unit_test/moss_vl_realtime/) | 模型接入与调度回归 |

## 来源与致谢

本项目基于 [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) 开发，保留原有框架、Git 历史和许可证。开发环境此前使用的 fork 为 [CloudRipple/sglang-omni](https://github.com/CloudRipple/sglang-omni)，当前特化版本在 [fnlp-vision/sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) 维护。

感谢 SGLang-Omni、SGLang 和 MOSS-VL 团队。代码使用 [Apache License 2.0](./LICENSE)。问题反馈请提交至[本仓库 Issues](https://github.com/fnlp-vision/sglang-omni-realtime/issues)，通用框架资料见 [SGLang-Omni 官方文档](https://sgl-project.github.io/sglang-omni/)。
