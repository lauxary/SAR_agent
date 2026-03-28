from ultralytics import YOLO
from resources.config import YOLO_WEIGHTS

class SARVisionDetector:
    def __init__(self):
        # 实例化网络并常驻 GPU 内存
        self.model = YOLO(YOLO_WEIGHTS)    # 加载训练好的权重文件
        
    def get_saliency_box(self, img_path, conf_threshold=0.25):
        """执行二元异常检测，返回张量归一化坐标"""
        results = self.model.predict(source=img_path, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None
        # 提取置信度最高的顶层边界框
        xywhn = boxes.xywhn[0].cpu().numpy() 
        return xywhn