# Ascend 集成

[English](./README.md) | **简体中文**

在 Ascend 上运行 MOSS-VL 原生后端，并使用统一的 [VL 旧协议适配层](../../vl_legacy_adapter/README_zh.md)。后端提供 `/v1/video/realtime`，独立适配层提供 `/v1/realtime`，不在后端内维护第二套旧协议实现。

## 环境

目标环境：Ascend 910B2C、CANN 9.0.0、PyTorch/torch_npu 2.11、支持 Ascend 的 SGLang 0.5.14 或 0.5.16、Transformers 5.12.1，以及 Python 3.12 或 3.13。使用 NPU 维护方提供的配套环境和本地 [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) 权重。不要在 NPU 环境安装 CUDA 依赖锁。两个 SGLang 版本均为实现目标，需分别完成运行验收；厂商构建必须提供对应的上游 API。

| SGLang API | 集成行为 |
| --- | --- |
| 0.5.16：`ModelRunner(ps=...)`、原生 `NextBatchPlan` 和请求 range | 使用原生接口，不应用旧版运行时补丁 |
| 0.5.14：独立 rank 参数、调度器持有 batch、`fill_len`/`extend_input_len` | 实例内调度适配、请求 range 同步、显式 forward 参数和 KV 容量适配 |

兼容判断依据实际 API 签名，不仅依赖版本字符串。未知构造接口和不支持的 forward 参数明确失败；不替换上游 Scheduler 类的方法，不跳过显存分配，也不静默丢弃 forward 参数。源码依据：[0.5.14 调度器](https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/managers/scheduler.py)、[0.5.14 ModelRunner](https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/model_executor/model_runner.py)。

在仓库根目录，使用该环境的 Python 执行：

```bash
python -m pip install --no-deps -e .
bash patches/npu/apply_npu_patches.sh
export ASCEND_USE_FA=false
export ASCEND_USE_FIA=false
```

补丁安装器默认修改当前解释器的 SGLang，可选参数为对应的 `site-packages` 目录。根据已安装的 ModelRunner 签名选择 0.5.14 或 0.5.16 补丁集；厂商回移代码可用 `--patch-set 0.5.14` 或 `--patch-set 0.5.16` 显式选择源码布局，不兼容时仍会失败。修改前先停止服务。全部补丁先在临时副本中处理：已经应用的允许跳过，不兼容的明确失败；被修改的文件保留 `.moss-npu.bak` 备份。安装后重启服务。环境应预先包含 FastAPI、websockets、Pillow、psutil、pytest、pytest-asyncio 等应用依赖。

帧可见性通过 native SDPA extend 路径的逐请求 mask 实现。带 mask 的 cross-attention 不接受 FA/FIA、推测解码或 context parallel 路径。补丁缺失时拒绝启动 MOSS-VL NPU 服务；decode 保持原有的视觉 KV 全可见行为。

## 启动

```bash
MODEL_PATH=/path/to/model bash deploy.sh start
# 两组 TP2，逻辑设备分别为 0,1 和 2,3：
MODEL_PATH=/path/to/model bash deploy.sh start4
# 两组 TP4，逻辑设备分别为 0-3 和 4-7：
MODEL_PATH=/path/to/model bash deploy.sh start8
```

命令前台运行，通过 Ctrl-C 或服务管理器发送 SIGTERM 停止，仅清理自己启动的进程。不再使用按进程名匹配的 `stop`/`restart` 命令。日志写入启动时打印的独立 `/tmp/moss-npu-*` 目录。任一子进程失败时停止该部署。

| 配置 | 默认值 |
| --- | --- |
| 原生后端端口 | `PORT=8000`，第二组为 8001 |
| 旧协议适配层端口 | `ADAPTER_PORT=18600`，第二组为 18601 |
| 单组设备 | `GPU=0`，或用 `GPUS=0,1` 指定 TP |
| 分组覆盖 | `NPU_GROUPS='0,1;2,3'` |
| 后端会话容量 | 每组设备数，可用 `MAX_RUNNING_REQUESTS` 覆盖 |
| 适配层容量 | `MAX_INFLIGHT=1`，并发调用时独立调整 |
| Context / 静态显存比例 | TP2：8192 / 0.80；其他：32768 / 0.70 |

`CONTEXT_LENGTH`、`MEM_FRACTION`、`HOST` 可覆盖默认值。设备编号是可见设备列表中的逻辑索引，必须核对实际机器的 HCCS 分组，示例不代表所有机器的拓扑。每个适配层连接自己的后端，不做负载均衡。容量和显存默认值需要在 NPU 上验收。

NPU 启动锁通过 `ASCEND_RT_VISIBLE_DEVICES` 解析进程内编号，使用独立的 NPU 锁名称；CUDA 保持原有的锁映射。TP 通信初始化沿用标准 ModelRunner 调用链接收部署层分配的 rendezvous 端口。

## 验收

**当前状态：实现交接，尚未验证。本次集成修正未运行测试或推理。** 接收方应记录提交版本、环境、命令、原始输出和结果，验收后再合入 main。其他文档中的历史 CUDA 数据不代表本次 NPU 修订的结果。

```bash
python -m pytest tests/unit_test/vendor/test_sglang_server_args.py \
  tests/unit_test/vendor/test_sglang_versions.py \
  tests/unit_test/pipeline/test_stage_process_env.py \
  tests/unit_test/pipeline/test_npu_startup_lock.py \
  vl_legacy_adapter/tests/test_lifecycle.py \
  tests/unit_test/moss_vl_realtime/test_legacy_perf_probe.py -q
MOSSVL_TEST_NPU_PATCHES=1 NPU_TEST_DEVICE=cpu python -m pytest \
  tests/unit_test/moss_vl_realtime/test_npu_attention.py -q
MOSSVL_TEST_NPU_PATCHES=1 NPU_TEST_DEVICE=npu:0 python -m pytest \
  tests/unit_test/moss_vl_realtime/test_npu_attention.py -q
python vl_legacy_adapter/tests/smoke_client.py \
  --url ws://127.0.0.1:18600/v1/realtime \
  --testdata deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122
bash perf.sh
```

`perf.sh` 默认运行五轮，每轮使用四张不同的仓库样例帧，速率配置为 160 token/s，并对每轮实施 10 秒超时。必须收齐 ACK 且有可见文字；保留与 ACK 交错的输出，TTFT 从建连开始计到首段可见文字。可通过 `FRAMES_DIR`、`NFRAMES`、`ROUNDS`、`TOKEN_RATE`、`TIMEOUT_S`、`VL_MODEL_WS_URL` 配置。这些是协议与时延检查，不是 HF 等价性证明。

应在两个 SGLang 环境分别执行上述检查。合入 main 前还需完成：NPU HF/SGLang 多帧 extend、问题位于帧边界前后、历史视觉 KV、视觉滑窗切换、无可见帧行，以及长度不同的 1/2/4 会话对比。使用维护方 HF 脚本，保持输入和采样一致，保留输出及首次分歧证据。同时覆盖延迟/缺失二进制、损坏图片、启动失败、取消重连、显存回收和 CUDA 回归。仓库现有 CUDA 精度启动器不是 NPU 测评脚本。
