# VL API v2 验证

[English](./VALIDATION.md) | **简体中文**

## 原版精度测试

2026-09-11，在原 main `68a0eef` 和当前分支分别原样执行仓库精度入口，启用严格 token/事件检查：

```bash
bash deployment/moss_vl_realtime/test_accuracy.sh /path/to/model --strict-tokens
```

结果：**PASS**。测试脚本、判定逻辑、manifest、任务规则均未修改。保留原始 system/initial prompt、完整 PNG 帧、时间戳及问题事件，使用原来的生成参数，每个事件生成到 silence 后推进。HF 与 SGLang 分别占用一张 H200；每个后端的全部并发会话共享单卡。

| Sessions | 原 main HF/SGLang token 与事件一致 | 当前分支 HF/SGLang token 与事件一致 | main/v1/v2 协议逐事件文本一致 | 三个协议入口各自与单路一致 |
| --- | --- | --- | --- | --- |
| 1 | 4/4 | 4/4 | 4/4 | 4/4 |
| 2 | 4/4 | 4/4 | 4/4 | 4/4 |
| 4 | 4/4 | 4/4 | 4/4 | 4/4 |

两个分支的任务检查和单/多路一致性均通过；跨分支直接比较也得到 HF 12/12、SGLang 12/12 的原始 token 与逐事件文本完全一致。

另将同一 manifest 回放到原 main、当前 v1、当前 v2 的 WebSocket 入口，共 36 个会话，原始 PNG 字节经过校验，不转码、不改问题。完整文本及逐输入事件文本均与原版引擎测试一致，任务检查均通过。协议不暴露原始 token ID，因此这里不将 WebSocket 文本一致称为原始 token 一致；v2 的逐段 done 与原生终态 done 按各自协议检查，不要求相同。

原版精度测试的全通过结论得到复现，当前改动未破坏这组回归。下文是不同输入条件的补充实验，不替代原版测试。

## 补充实验环境

2026-09-11，H200，单服务单卡 TP=1，BF16，SGLang 0.5.16、Transformers 5.12.1、本地 MOSS-VL Realtime SGLANG 模型。使用生产启动配置：`mem_fraction_static=0.5`、context 131072、4 会话、60 秒视觉 KV 滑窗，未接入 memory/ASR/TTS。

对照原 main `68a0eef`、当前分支原生 v1 和当前分支 v2。使用仓库四组测试图像，温度 0，逐帧等待 processed，按 `turn_id` 拼接可见文本；不要求新旧协议的 done 事件相同。生成配置为 `max_new_tokens=512`、`max_tokens_per_turn=160`。本次不是时延基准或 HF 模型精度评测。

## 结果

| 检查项 | 结果 |
| --- | --- |
| CPU 回归 | 499 通过，2 跳过 |
| 四组单会话样例 | main、v1、v2 可见文本完全一致；最终视觉/文本/总用量一致 |
| 后续帧与新问题 | v1/v2 文本及最终用量一致 |
| 同轮多段回答 | 同一个 turn 返回三个不同 response ID；拼接文本与 v1 一致；纯静默不重复结算 |
| 输出中打断 | v1/v2 文本一致；被打断段不返回正常 done，其用量计入后续结算 |
| 视觉滑窗 | 16 帧跨度 150 秒；历史视觉位置 3536、驻留位置 1547；v1/v2 文本及最终用量一致 |
| 输入校验 | 未知字段、跳号、损坏图片被拒；同序号有效重试成功 |
| 中止与超时 | 空闲、输出中及输入在途 abort，缺失二进制超时均正确收尾；关闭 telemetry 仍有最终用量 |
| 上下文上限 | 初始/追加超长输入均返回 context_exhausted；追加错误带 seq_no，未提交内容不计量 |
| 容量与恢复 | v1/v2 共用 4 个槽位；超额请求被拒；断连后新会话正常，无残留请求 |

所有已记录 v2 会话均检查了结算算术、累计值单调性、唯一终态、终态后无输出和 telemetry 频率。隔离测试服务正常退出，无残留子进程。

### 开放描述并发逐字对齐

本表使用前 4 帧转 JPEG 后的开放描述问题，与上面的原版 manifest 测试不同。下表为各并发会话与对应单路文本完全一致的数量，不是语义正确率。会话独立运行，没有强制相同 batch 组成。

| 版本 | 2 会话 | 4 会话 |
| --- | --- | --- |
| 原 main | 1/2 | 3/4 |
| 当前 v1 | 0/2 | 3/4 |
| 当前 v2 | 0/2 | 2/4 |

未发现串会话、缺失终态或用量计算异常，但不能宣称并发逐字对齐通过。原 main 同样存在措辞变化；仅凭这些结果不能将差异归因于 v2，也未完成逐 token/logit 的根因定位。样例主体描述可对应输入，文本对齐不等于描述中的每个细节都真实正确。

## 回归命令

```bash
python -m pytest tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_vl_api_v2.py \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/serve/test_video_realtime_lifecycle.py \
  vl_legacy_adapter/tests -q
```

上述是 CPU 回归命令，不会复现 GPU 对照。两项跳过分别为需要 CUDA 的模型步对照、需要显式模型路径的 processor 对照。未执行 TP 多卡/NPU、完整 Demo/memory/ASR/TTS 端到端、长时间压测或真实 worker 崩溃测试；最终快照不可用等故障路径由 CPU 测试覆盖。
