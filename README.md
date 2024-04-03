<div align="center">
    <img src="logo.png"/>
</div>

不同框架进行推理的目标检测推理代码

- 测试的图片统一放在images文件夹中
- 模型大部分都是fp16，int8精度太低fp32速度太慢
- 主要是YOLO模型

如何使用该代码

```bash
pip install -r requirements.txt
```

## Onnxruntime GPU

肯定推理快，但是为什么实际使用慢，在初始化的时候时间非常长如果是初始化完以后就很快了实测CPU600ms但是GPU版本只要30ms
