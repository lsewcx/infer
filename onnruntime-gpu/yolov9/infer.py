from rich.console import Console
import onnxruntime
import cv2
import numpy as np
from typing import Tuple, List
import yaml
import time

console = Console()
console.log("使用的设备为:",onnxruntime.get_device())


class YOLOv9:
    def __init__(self,
                 model_path: str,  # 模型文件路径
                 class_mapping_path: str,  # 类别映射文件路径
                 original_size: Tuple[int, int] = (1280, 720),  # 原始图像尺寸，默认为(1280, 720)
                 score_threshold: float = 0.1,  # 分数阈值，用于过滤检测结果，默认为0.1
                 conf_thresold: float = 0.4,  # 置信度阈值，用于过滤检测结果，默认为0.4
                 iou_threshold: float = 0.4,  # IOU阈值，用于判断两个物体是否重叠，默认为0.4
                 device: str = "GPU") -> None:  # 模型推理使用的设备，默认为"CPU"
        """
        初始化函数，设置模型加载、推理的相关参数及设备选择。
        GPU初次加载较慢，所以采用for循环才能看到真实的时间
        """
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path

        self.device = device
        self.score_threshold = score_threshold
        self.conf_thresold = conf_thresold
        self.iou_threshold = iou_threshold
        self.image_width, self.image_height = original_size
        self.create_session()  # 创建推理会话

    def create_session(self) -> None:
        """
        创建一个ONNX运行时会话实例，用于执行ONNX模型推理。
        
        根据模型指定的设备类型（CPU或GPU），配置会话选项并初始化会话实例。
        同时，获取并记录模型的输入、输出信息，以及输入形状和类别映射（如果存在）。
        
        返回:
            None
        """
        opt_session = onnxruntime.SessionOptions()  # 创建会话选项实例
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL  # 禁用所有图优化
        providers = [] 
        if self.device.casefold() != "cpu":  # 如果设备不是CPU，则添加CUDA执行提供者
            providers.append("CUDAExecutionProvider")
        session = onnxruntime.InferenceSession(self.model_path, providers=providers)  # 创建会话实例
        self.session = session
        self.model_inputs = self.session.get_inputs()  # 获取模型输入
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]  # 记录输入名称
        self.input_shape = self.model_inputs[0].shape  # 获取第一个输入的形状
        self.model_output = self.session.get_outputs()  # 获取模型输出
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]  # 记录输出名称
        self.input_height, self.input_width = self.input_shape[2:]  # 从输入形状中提取高度和宽度

        if self.class_mapping_path is not None:  # 如果存在类别映射文件路径
            with open(self.class_mapping_path, 'r') as file:  # 加载类别映射文件
                yaml_file = yaml.safe_load(file)
                self.classes = yaml_file['names']  # 提取类别名称
                self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))  # 为每个类别随机生成颜色码

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        图像预处理函数
        
        参数:
        img: np.ndarray - 输入图像，格式为BGR的numpy数组。
        
        返回值:
        np.ndarray - 预处理后的图像张量，格式为RGB，尺度为(1, C, H, W)，其中C是通道数，H是高度，W是宽度。
        """
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将输入图像从BGR转换为RGB
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))  # 将图像调整到目标大小

        # 将图像像素值缩放到0到1之间，并调整维度顺序以符合模型输入要求
        input_image = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)  # 从(H, W, C)调整到(C, H, W)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float16)  # 在第一个维度上增加一个新轴，转换为张量，并指定数据类型为float16
        return input_tensor
    
    def xywh2xyxy(self, x):
        """
        将给定的xywh格式的边界框坐标转换为xyxy格式。
        
        参数:
        - x: 输入的边界框坐标，格式为[x_min, y_min, width, height]，其中x_min和y_min是中心点的坐标，
            width和height分别是框的宽度和高度。x可以是一个numpy数组，也可以是一个包含多个边界框的批量数据。
        
        返回值:
        - y: 转换后的边界框坐标，格式为[x_min, y_min, x_max, y_max]，其中x_min和y_min是左上角点的坐标，
            x_max和y_max是右下角点的坐标。返回值与输入x的形状相同。
        """
        y = np.copy(x)  # 复制输入的坐标数据，防止原始数据被修改
        # 计算并更新左上角和右下角的坐标
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_min = center_x - width / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_min = center_y - height / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_max = center_x + width / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_max = center_y + height / 2
        return y
    
    def postprocess(self, outputs):
        """
        对模型输出进行后处理，生成检测到的物体的详细信息列表。
        
        参数:
        - outputs: 模型的输出，一个numpy数组，包含预测的边界框、类别得分等信息。
        
        返回值:
        - detections: 一个列表，包含每个检测到的物体的信息，如类别索引、置信度、边界框坐标和类别名称。
        """
        # 从模型输出中提取预测并获取最高得分的类别
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # 调整边界框的尺寸以适应原始图像大小
        boxes = predictions[:, :4]
        
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.float16)
        
        # 应用非最大抑制以减少重叠框的数量，自行百度什么是nms这是很常见的后处理手段
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.score_threshold, nms_threshold=self.iou_threshold)
        
        # 创建最终的检测结果列表
        detections = []
        for bbox, score, label in zip(self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
                "class_name": self.get_label_name(label)
            })
        
        return detections
    
    def get_label_name(self, class_id: int) -> str:
        return self.classes[class_id]
        
    def detect(self, img: np.ndarray) -> List:
        '''
        核心代码
        预处理

        AI推理

        和后处理
        '''
        start_time = time.time()  # 获取推理开始的时间戳
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        end_time = time.time()  # 获取推理结束的时间戳
        inference_time = (end_time - start_time) * 1000  # 计算推理时间，并转换为毫秒
        console.log(f"Inference time: {inference_time} milliseconds")  # 打印推理时间
        fps = 1000 / inference_time  # 计算FPS
        console.log(f"FPS: {fps}")  # 打印FPS
        return self.postprocess(outputs)
    
    def draw_detections(self, img, detections: List)->None:
        """
        在图像上绘制检测到的对象边界框和标签。
        
        :param img: 输入图像，将会在上面绘制检测结果。
        :param detections: 检测结果列表，每个结果包含边界框坐标、类别索引和置信度。
        """
        for detection in detections:
            # 提取边界框坐标，转换为整数类型
            x1, y1, x2, y2 = detection['box'].astype(int)
            # 获取对象的类别索引和置信度
            class_id = detection['class_index']
            confidence = detection['confidence']
            # 根据类别索引获取颜色
            color = self.color_palette[class_id]
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # 生成并计算标签的尺寸
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # 计算标签的放置位置
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            # 绘制标签的背景矩形
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )
            # 在背景矩形上绘制文本标签
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__=="__main__":
    weight_path = r"models\yolov9\fp16\yolov9-c-converted.onnx"
    image = cv2.imread("images/people-4273127_960_720.jpg")
    h, w = image.shape[:2]
    detector = YOLOv9(model_path=f"{weight_path}",
                      class_mapping_path="onnruntime-gpu/yolov9/class.yaml",#yaml中是检测的类别
                      original_size=(w, h))
    for i in range(10):
        detections = detector.detect(image)
        detector.draw_detections(image, detections=detections)
        # cv2.imshow("结果", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
