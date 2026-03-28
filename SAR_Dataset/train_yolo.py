from ultralytics import YOLO

# 载入预训练的轻量级权重
model = YOLO('yolov8n.pt') 

# 启动训练流程
results = model.train(
    data='sar_prior.yaml', 
    epochs=50, 
    imgsz=2048, 
    device=0, 
    workers=0,  
    batch=8,    
    plots=True 
)