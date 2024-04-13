import cv2
import numpy as np
def visualize(img, detections, classes,score_threshold=0.5, conf_threshold=0.5, iou_threshold=0.5):
    '''
    可视化检测结果，将检测到的物体绘制在原始图像上。
    
    参数:
    - img: 原始图像,一个numpy数组,格式为[H, W, C],其中H、W和C分别表示图像的高度、宽度和通道数。
    - detections: 检测到的物体信息列表，每个元素是一个字典，包含类别索引、置信度、边界框坐标和类别名称。这个参数是模型跑出来直接得到的
    '''
    for detection in detections:
        # 提取边界框坐标，转换为整数类型
        x1, y1, x2, y2 = detection['box'].astype(int)
        # 获取对象的类别索引和置信度
        class_id = detection['class_index']
        confidence = detection['confidence']
        # 根据类别索引获取颜色
        color_palette = np.random.uniform(0, 255, size=(len(classes), 3))
        color = color_palette[class_id]
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 生成并计算标签的尺寸
        label = f"{classes[class_id]}: {confidence:.2f}"
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