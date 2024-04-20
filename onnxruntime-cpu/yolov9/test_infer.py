import unittest
import numpy as np
from infer import YOLOv9
from htmltestreport import HTMLTestReport

class TestYOLOv9(unittest.TestCase):
    def test_detect(self):
        weight_path = r"models\yolov9\fp16\yolov9-c-converted.onnx"
        image = np.random.rand(720, 1280, 3).astype(np.float32)  # 创建一个32位浮点数图像
        detector = YOLOv9(model_path=f"{weight_path}",
                          class_mapping_path="onnruntime-gpu/yolov9/class.yaml",
                          original_size=(1280, 720))
        for i in range(5):
            detections = detector.detect(image)

        # 检查detect方法的返回值
        self.assertIsInstance(detections, list)  # 确保返回值是一个列表
        for detection in detections:
            self.assertIsInstance(detection, dict)  # 确保每个元素是一个字典
            self.assertIn('class_index', detection)  # 确保字典包含正确的键
            self.assertIn('confidence', detection)
            self.assertIn('box', detection)
            self.assertIn('class_name', detection)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestYOLOv9))

    # 实例化HTMLTestReport对象
    report_path = "report.html"
    report = HTMLTestReport(report_path, title='单元测试报告', description='')
    report.run(suite)