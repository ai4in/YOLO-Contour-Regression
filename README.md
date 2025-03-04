<details open>
<summary>背景</summary>
背景：为了减少基于像素分割的深度学习方法算力消耗问题，基于yolov8标签分配策略进行极坐标建模。

优点：推理速度和后处理速度在模型相同量级下，与yolov8目标检测速度一致。

缺点： 

1.训练时间是目标检测的1~2倍。

2.针对凹形物体建模较差。



</details>


References: 

1.[PolarMask: Single Shot Instance Segmentation with Polar Representation](https://arxiv.org/abs/1909.13226)

2.[yolov8](https://github.com/ultralytics/ultralytics)


<details open>
<summary>标签</summary>

txt标签格式(与v8分割一致)

```bash
cls, x1, y1, x2, y2, x3, y3, ....., xn, yn, 
```
</details>

请参阅下面的快速安装和使用示例
<details open>
<summary>安装</summary>

使用Pip

```bash
pip install -r requirment.txt
```
</details>

#### Python

在 Python 环境中直接使用

```python
from ultralytics import YOLO

# 加载模型
model=YOLO('yolov8n-seg.yaml')
# 使用模型
model.train(data='/your/path/dataset.yaml',epochs=300,task='segment',mixup=0,mosaic=1,imgsz=640,workers=2,batch=32,patience=50,device=0)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```
实例:train_seg_car.py

</details>

## <div align="center">模型</div>
基于yolov8训练框架以及标签分配策略，融合PolarMask建模思路，与原来的yolov8代码相比，改动的主要函数如下：


</details>

#### Python
```python
#ultralytics\utils\loss.py
class v8DetectionLoss()
class v8SegmentationLoss(v8DetectionLoss)
#ultralytics\utils\tal.py
class TaskAlignedAssigner(nn.Module)
#ultralytics\nn\modules\head.py
class Detect(nn.Module)
class Segment(Detect)

#ultralytics\utils\plotting.py
def output_to_target()

#ultralytics\utils\ops.py
def non_max_suppression()

#ultralytics\models\yolo\segment\val.py
def postprocess(self, preds)
def update_metrics(self, preds, batch)

#ultralytics\models\yolo\segment\train.py
def get_validator(self)
```
</details>

