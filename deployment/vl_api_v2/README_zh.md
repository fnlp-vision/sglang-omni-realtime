# VL API v2 部署

[English](./README.md) | **简体中文**

安装后端环境后，在仓库根目录运行：

```bash
bash deployment/vl_api_v2/start.sh /path/to/MOSS-VL-Realtime-SGLANG
```

原生入口仍为 `ws://127.0.0.1:18500/v1/video/realtime`，v2 入口为 `ws://127.0.0.1:18610/v1/video/realtime`，共享同一模型和会话容量。`VL_API_V2_PORT` 修改新端口，`--gpus 0` 指定 GPU。沿用原生产配置及不使用代理的启动器。

完整字段、事件和计量规则见 [API 接口文档](../../sglang_omni/serve/vl_api_adapter/API_zh.md)，参考客户端与测试命令见[使用说明](../../sglang_omni/serve/vl_api_adapter/README_zh.md)，实测范围见[验证结果](./VALIDATION_zh.md)。旧 Demo／网关继续使用原生入口。
