<div align="center">
    <img src="logo.png"/>
</div>

不同框架进行推理的目标检测推理代码

- 测试的图片统一放在 images 文件夹中
- 模型大部分都是 fp16，int8 精度太低 fp32 速度太慢
- 主要是 YOLO 模型

如何使用该代码

```bash
pip install -r requirements.txt
```

代码在具体的框架里面的 infer.py

## Onnxruntime GPU

肯定推理快，但是为什么实际使用慢，在初始化的时候时间非常长如果是初始化完以后就很快了实测 CPU600ms 但是 GPU 版本只要 30ms

## 学习尽量看 cpu 版本的代码因为不清楚 CUDA 安装和 CUDNN 安装是否成功，但是 cpu 版本都能跑
