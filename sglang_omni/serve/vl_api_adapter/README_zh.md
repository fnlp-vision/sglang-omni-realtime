# VL API v2

[English](./README.md) | **简体中文**

[完整 API 接口文档](./API_zh.md) | [验证结果](../../../deployment/vl_api_v2/VALIDATION_zh.md)

面向 MOSS-VL realtime 的显式启用协议层。有可见文字后收到**模型发出的 silence 事件**，返回一次 `response.done`；没有新文字的连续静默不生成空回答。后端请求继续运行，直到 final、abort、容量限制或故障使其结束。

原生 v1 和 `vl_legacy_adapter` 保持原语义。现有 Demo／薄网关将 `response.done` 视为请求终态，不能直接改连 v2。`turn_id` 仍由新问题推进，`response_id`、`response_seq` 区分同一轮内的多段回答。不通过网络静默计时、修改生成额度、重载模型或重建 KV 切分回答。

## 启动

准备仓库的后端环境，在仓库根目录执行：

```bash
bash deployment/vl_api_v2/start.sh /path/to/MOSS-VL-Realtime-SGLANG
```

复用原有单卡生产配置和空闲卡选择。原生 v1 默认在 18500，v2 默认在 18610，路径为 `/v1/video/realtime`，共享模型和总会话容量。`--gpus`、`--host`、`--port` 配置原有启动器，`VL_API_V2_PORT` 修改 v2 端口。原启动脚本默认不启用 v2。

自定义或 TP 部署可给 `examples/run_moss_vl_realtime_server.py` 增加 `--vl-api-v2-port 18610`。`VL_API_V2_MODEL_VERSION` 可提供部署版本，缺失时返回 null/unknown。设置 `VL_API_V2_API_KEY` 后，v2 WS 要求 `Authorization: Bearer ...`；未设置时应仅用于可信网络或有鉴权的网关之后。v2 不暴露管理操作路由。

```bash
python examples/vl_api_v2_client.py \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0003.png
```

`--frame`、`--prompt` 可重复。参考客户端不会在逐段 done 后结束连接，仅在 session done 后结束。

## 输入与限制

字段定义及三阶段帧握手沿用[原生输入参考](../../../docs/cookbook/moss_vl_realtime.md)，v2 的输出差异见下文。

保留原生 configure/frame/prompt 字段：初始用户 `prompt`、独立 `system_prompt`、每次 extend 的 `max_new_tokens` 余量、tokens/s 的 `max_tokens_per_turn`、输入队列和 `include_usage`。图像支持 JPEG/PNG/WebP，保留帧附带 prompt/final。拒绝未知字段、非有限数值、非法序号和编码与 MIME 不匹配的图像。帧／问题共用连续 `seq_no`；只有明确拒绝且未 accepted 的输入可以复用序号。

`session.created` 通告 `response.done.per_response`、计量、响应标识及错误关联能力；`session.configured` 返回实际后端 `context_limit`。帧按 metadata -> ready -> binary -> accepted 提交，processed 表示模型已消费。ready 后接收二进制的上限为 10 秒。configure 期限沿用原生管理器，不包含预填充；parked 超时和其他限制按实际值通告。

## 回答与计量

```json
{"type":"response.done","response_id":"response_...","response_seq":1,"turn_id":1,"finish_reason":"stop","boundary":"silence","usage":{"vision_tokens":145,"text_input_tokens":78,"text_output_tokens":10,"text_tokens":88,"total_tokens":233,"cumulative":{"vision_tokens":145,"text_input_tokens":78,"text_output_tokens":10,"text_tokens":88,"total_tokens":233}}}
```

- `response.text.delta` 带当前回答 ID；首次可见文字之前的空白增量可使用 null 响应标识。
- 同轮继续输出创建新回答 ID。新问题打断时不结算未完成段落，其消耗进入下一次增量或最终残差。
- 回答序号包括被打断的段落，因此 done 事件之间允许跳号；它与必须连续的输入 seq_no 不同。`boundary` 区分模型静默和原生请求完成，静默结算使用 `finish_reason=stop`。
- 请求自然结束时先结算尚未结束的回答段，再发 `session.done`；已在 silence 结算的段落不重复。abort／故障不伪造正常回答完成。
- 计量是已提交的逻辑模型位置，不是 FLOPs、字符数或驻留 KV。视觉位置包括 encoder 分隔位置；文本输入包括实际模板、帧相关文本和问题。确认的生成 token 包括 silence/EOS/控制 token。生成 token 下一步作为输入时不重复计费；失败未提交部分和丢弃的前瞻步骤不计。
- 每次增量为累计值减去上次结算水位。`text_tokens = text_input_tokens + text_output_tokens`，`total_tokens = vision_tokens + text_tokens`。淘汰不减少历史总值，不用截断负差值掩盖计数回退。
- `session.usage` 是可选观测通道，仅在 v2 最多 1 Hz，保持原 context／驻留含义；结算精度不依赖其推送频率。v1 的快照时序不变。
- `session.done.usage` 是后端冻结的总值，覆盖最后一次回答后的残差。reset／rollover 新建账本，跨请求聚合由调用方负责。

## 错误与终态

```json
{"type":"session.done","session_id":"video_sess_...","request_id":"video_req_...","reason":"completed","aborted":false,"usage":{"vision_tokens":145,"text_input_tokens":78,"text_output_tokens":10,"text_tokens":88,"total_tokens":233}}
```

可恢复错误返回 `invalid_request`，能识别合法整数序号时带 `seq_no`。被拒帧另外返回 `input.frame.rejected`，缺失／非法序号不能虚构，此事件中可为 null。拒绝另一条元数据不会清除当前仍有效的待接收帧。

有序致命错误按 error -> 唯一 session.done 收尾：上下文用尽为 `context_exhausted`，配置／二进制／parked 超时为 `session_timeout`，其他后端故障为 `response_failed`。会话 reason 为 completed、aborted、context_exhausted 或 error。session done 后不再发输出。

最终结算通过内部管理通道，在调度器所有者线程处理目标请求的在途步骤后返回确认，不把 `Client.abort()` 返回直接当作 GPU 已完成。后端保留不含张量的结束账本，按五分钟期限和 4096 条目标上限在新请求进入时清理；TP 仅取 leader 的逻辑总值，不按 rank 数相乘。

断网／进程故障不能保证终态送达。如果后端无法提供可信的最终结算，适配层尽力报错并以 1011 异常关闭，不伪造零用量或完整终值；这不属于有序且结算成功的会话。仍会执行清理。模型接纳前的容量拒绝返回零用量和关闭码 1013。

## 验收交接

原版精度测试在 1/2/4 会话下全部通过，同一 manifest 的新旧协议逐事件文本也全部一致。另完成 CPU 回归、同轮多段回答、打断、滑窗、计量和异常清理检查。验证范围及开放描述补充实验见[验证结果](../../../deployment/vl_api_v2/VALIDATION_zh.md)；固定样例通过不代表任意并发输入都保证逐字一致。

```bash
python -m pytest tests/unit_test/serve/test_vl_api_v2.py \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/serve/test_video_realtime_lifecycle.py \
  vl_legacy_adapter/tests/test_lifecycle.py -q
```

逐段 done 是有意的新协议语义，不是原地透明升级。AGW 必须识别 v2 能力／入口，并使用唯一回答 ID 结算。
