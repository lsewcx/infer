from rich.console import Console
import onnxruntime
import cv2
import numpy as np
from typing import Tuple, List
import yaml

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
                 device: str = "CPU") -> None:  # 模型推理使用的设备，默认为"CPU"
        """
        初始化函数，设置模型加载、推理的相关参数及设备选择。
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
        providers = ['CPUExecutionProvider']  # 默认使用CPU执行提供者
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
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))

        # Scale input pixel value to 0 to 1
        input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float16)
        return input_tensor
    
    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y 
    
    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Rescale box
        boxes = predictions[:, :4]
        
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.float16)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.score_threshold, nms_threshold=self.iou_threshold)
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
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        return self.postprocess(outputs)
    
    def draw_detections(self, img, detections: List):
        for detection in detections:
            x1, y1, x2, y2 = detection['box'].astype(int)
            class_id = detection['class_index']
            confidence = detection['confidence']
            color = self.color_palette[class_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__=="__main__":

    weight_path = r"onnruntime-gpu\models\yolov9\fp16\yolov9-c-converted.onnx"
    image = cv2.imread("images/people-4273127_960_720.jpg")
    h, w = image.shape[:2]
    detector = YOLOv9(model_path=f"{weight_path}",
                      class_mapping_path="onnruntime-gpu/yolov9/class.yaml",
                      original_size=(w, h))
    detections = detector.detect(image)
    detector.draw_detections(image, detections=detections)
    cv2.imshow("结果", image)
    cv2.waitKey(0)
