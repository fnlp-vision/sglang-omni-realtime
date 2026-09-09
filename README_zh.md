# SGLang-Omni Realtime for MOSS-VL

[English](./README.md) | 简体中文

基于 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 和 [SGLang](https://github.com/sgl-project/sglang) 的实时视频理解后端。客户端通过 WebSocket 持续发送带时间戳的帧和问题，接收增量文本及输入处理事件。

## 配套项目

- [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG)：模型权重与 Transformers 5.12.1 兼容的模型/processor 代码。
- [MOSS-VL-Realtime Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/README_zh.md)：浏览器界面、ASR/TTS、memory 和 REST 网关。需要完整应用时使用其安装器，不必单独安装本后端。

## 功能

- JPEG、PNG、WebP 视频帧的增量视觉特征与 KV 缓存。
- 持续会话、新问题打断、静默后唤醒。
- 动态多会话调度、单卡及张量并行推理。
- Decode CUDA Graph、视觉 KV 滑窗和有界输入队列。

## 安装

需要 Linux x86_64、Python 3.12、兼容 CUDA 13 的 NVIDIA 驱动、Git、C/C++ 编译器、CMake 和 FFmpeg，并先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。不要与 Demo 或其他推理引擎混用环境。

在仓库根目录执行：

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip sync --require-hashes \
  --build-constraint deployment/repro/build-constraints.txt \
  --index-url https://pypi.org/simple deployment/repro/requirements.lock
uv pip install --no-deps --no-build-isolation -e .
uv pip check
```

依赖文件采用 SGLang 0.5.16、Transformers 5.12.1、Torch 2.11.0 和 CUDA 13.0 工具链。系统软件包及常见问题见[安装指南](./docs/get_started/installation.md)。

## 下载与启动

```bash
export MODEL_PATH="$HOME/models/MOSS-VL-Realtime-SGLANG"
hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --revision bcfd9ccf1e9db2896ad852301cc8dde4a6349c78 \
  --local-dir "$MODEL_PATH"

export CUDA_HOME="$(python deployment/repro/cuda_toolkit.py)"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
python deployment/moss_vl_realtime/check_env.py "$MODEL_PATH"
bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"
```

模型公开可访问。请保留完整配置、自定义代码、tokenizer、processor 和所有权重分片。新终端中需重新激活 `.venv`，并在启动前设置相同的模型与工具链变量。

启动器自动选择空闲 GPU 并预热模型；用 `--gpus 0` 指定 GPU，`--port 18510` 更换端口，`--dry-run` 查看配置。

| 默认配置 | 值 |
| --- | --- |
| 服务地址 | `http://127.0.0.1:18500` |
| 会话容量 / context | 4 / 131072 |
| 显存比例 | 0.5 |
| 视觉 KV 窗口 | 60 秒，开启 |
| Pooling / async decode | 关闭 |

这些值来自 [config.json](./deployment/moss_vl_realtime/config.json)，适用于此 `start.sh`。底层 Python launcher 和 Demo 托管部署有各自默认配置，请按 GPU 调整显存与并发。

```bash
curl --fail http://127.0.0.1:18500/health
curl --fail http://127.0.0.1:18500/v1/models
```

## 调用示例

以下使用仓库自带视频帧：

```bash
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:18500/v1/video/realtime \
  --prompt "Describe the visible scene." \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0000.png --timestamp 0.0 \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0001.png --timestamp 1.0
```

客户端将最后一帧标为 `final`，持续摄像头交互使用 Demo。帧和问题共用递增的 `seq_no`；发送图像前等待 `input.frame.ready`。`session.abort` 结束会话。

`max_tokens_per_turn` 是每路 tokens/s 目标，不是回答长度。协议字段和高级部署参数见 [Realtime Cookbook](./docs/cookbook/moss_vl_realtime.md)。

## 连接已有 Demo

手工连接时，在 Demo 的 `.env.deploy` 中设置：

```dotenv
VLM_DEPLOY=sglang_omni
SGLANG_OMNI_URLS=http://127.0.0.1:18500
SGLANG_OMNI_SESSIONS_PER_REPLICA=4
SGLANG_OMNI_CONTEXT_LENGTH=131072
MODEL_PATH=/absolute/path/to/MOSS-VL-Realtime-SGLANG
```

URL、会话容量和 context 应与后端一致，模型/tokenizer 路径应在 Demo 所在机器可读。视觉 KV 回收不会阻止文本上下文增长，跨 context 延续由 Demo memory rollover 管理。

服务默认绑定 loopback，对外访问需配置鉴权、TLS 和访问控制。

## 测试与文档

- [精度与时延测试](./deployment/moss_vl_realtime/README.md)。
- [协议与高级配置](./docs/cookbook/moss_vl_realtime.md)。
- [容量规划](./docs/cookbook/moss_vl_realtime_capacity.md)。

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime*.py
```

## 许可证

本项目保留上游 [Apache License 2.0](./LICENSE)。感谢 SGLang-Omni、SGLang 和 MOSS-VL 团队。问题反馈请提交至[本仓库 Issues](https://github.com/fnlp-vision/sglang-omni-realtime/issues)。
