# 后端依赖与容器

`requirements.lock` 固定 Linux / Python 3.12 的后端依赖，原生安装步骤见[安装指南](../../docs/get_started/installation.md)。完整应用请从 [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo#quick-start) 开始。

## 容器

Dockerfile 使用构建上下文中的源码，不在启动时克隆或更新代码。模型在运行时挂载，凭据不放入镜像。

```bash
docker build -f docker/Dockerfile -t moss-vl-realtime:local .
docker run --rm --gpus all --ipc=host -p 18500:18500 \
  -v /absolute/model:/models/moss-vl:ro moss-vl-realtime:local \
  python3 examples/run_moss_vl_realtime_server.py --model-path /models/moss-vl \
  --gpu 0 --host 0.0.0.0 --port 18500 --mem-fraction-static 0.5 --max-running-requests 4
```

需要 Docker 与 NVIDIA Container Toolkit。容器路径尚未验证；使用前需完成镜像构建和 GPU 运行检查。不要将内部服务直接暴露公网。
