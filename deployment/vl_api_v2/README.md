# VL API v2 Deployment

**English** | [简体中文](./README_zh.md)

Run from the repository root after installing the backend environment:

```bash
bash deployment/vl_api_v2/start.sh /path/to/MOSS-VL-Realtime-SGLANG
```

The native endpoint stays at `ws://127.0.0.1:18500/v1/video/realtime`; v2 is available at `ws://127.0.0.1:18610/v1/video/realtime`. One model instance and its admission capacity are shared. Use `VL_API_V2_PORT` to change the new port, and `--gpus 0` to select a GPU. The underlying production profile and proxy-free launcher are unchanged.

See the [API reference](../../sglang_omni/serve/vl_api_adapter/API.md) for complete fields, events and accounting rules, the [usage guide](../../sglang_omni/serve/vl_api_adapter/README.md) for the reference client and test commands, and [validation results](./VALIDATION.md) for measured coverage. Old Demo/gateway clients must keep using the native endpoint.
