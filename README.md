
# deepsort_yolov3


对比原项目

+ 更改了网络结构，完善了yolo输入的类别
+ 更改了reid的输入
+ 增加`--count`实现人流统计的功能

+ 实现reid多baseline的选择，

+ 修改了一下bug，

+ 并且注释上加上了自己的理解



实现多目标追踪，能够使用`--count`简单实现人流的计数功能

使用yolov3作为目标检测网络，mobilenet作为特征提取网络

默认权重存放位置

./weights/yolov3-spp.pt

./deep_sort/deep/checkpoint/mobilenetv2_x1_0/mobilenetv2_x1_0_best.pt



# 权重和数据集下载地址：

[baidupan](https://pan.baidu.com/s/101Rsj4k0tm7J6p60qZbyYw)

提取码： zdfn



# 主体文件：

## deep_sort.py

````python
deep_sort.py [--yolo_cfg][--yolo_weights][--deepsort_checkpoint]
				[--ignore_display][--count][--save_path]
    			[--data_cfg][--img_size] 'VIDEO_PATH' 
````

参数解释：

+ 'VIDEO_PATH' 处理的视频文件目录，**必选参数**
+  --yolo_cfg  组成yolo框架的cfg地址，默认使用`yolov3-spp.cfg`
+ --yolo_weights yolo权重地址，默认使用 `yolov3-spp.pt`
+ --deepsort_checkpoint deepsort中Reid的权重文件目录，默认`mobilenetv2_x1_0_best.pt`
+ --ignore_display 是否实时显示，默认显示
+ --count 是否计数人流量， 默认不显示
+ --save_path 输出结果存放地址
+ --data_cfg yolo用，对应数据集的标签名 默认`coco.data`
+ --img_size yolo用，输入yolo图片Size，默认416 x 416
+ 其他args参数使用默认即可，或者自行修改

会生成目标轨迹文件在./data/videosample/predicts.txt



# 其他文件：

### 训练Reid：

在训练之前：

先利用 deep_sort_yolov3_pytorch/deep_sort/deep/prepare.py 处理Market1501数据集，转换成类似mot16的格式



./deep_sort/deep/train_wo_center.pytrain.py能实现在训练集market1501上的Reid网络的训练

**train_wo_center.pytrain.py**

```python
train_wo_center.pytrain.py [--data-dir][--interval][--model][--pretrained]
```

+ --data-dir 处理后的Market1501训练集的存放位置

+ --interval 训练每隔多少轮显示一次loss和acc

+ --model 使用的baseline的模型类型，可使用模型见deep_sort_yolov3_pytorch/deep_sort/deep/models/\__init__.py，默认为mobilenetv2_x1_0

  > **注意**：
  >
  > 如果要使用mobilenet以外的模型的模型，需要到deep_sort_yolov3_pytorch/deep_sort/deep/models/中对应网络结构中做如下修改:
  >
  > 1.在模型的类的`def __init__()`中增加reid变量，默认为False
  >
  > 2.在`def forward()`中最后一层全连接层之前，加入reid变量提取：
  >
  > ```
  >         if self.reid:
  >             x = v
  >             x = x.div(x.norm(p=2, dim=1, keepdim=True))
  >             return x
  > ```
  >
  > 以下以mobilenet为例：
  >
  > ```python
  > ...
  > class MobileNetV2(nn.Module):
  >     def __init__(..., reid = False, **kwargs):
  >     ...
  >     ...
  >     self.reid = reid
  >     
  > ...
  > ...
  >     def forward(self, x):
  >     ...
  >     if self.fc is not None:
  >             v = self.fc(v)
  > 
  >     if self.reid:
  >         x = v
  >         x = x.div(x.norm(p=2, dim=1, keepdim=True))
  >         return x
  > 
  >     y = self.classifier(v)
  >     ...
  > ```

+ --pretrained 是否使用预训练集



deep_sort_yolov3_pytorch/predict.py 用来测试yolo网络用



------

Reference:

1. [pprp/deep_sort_yolov3_pytorch: Add attention blocks such as cbam, se. Add deep sort, sort and some tracking algorithm using opencv 目标跟踪算法重实现 (github.com)](https://github.com/pprp/deep_sort_yolov3_pytorch)

2. [ZQPei/deep_sort_pytorch (github.com)](https://github.com/ZQPei/deep_sort_pytorch/tree/8cfe2467a4b1f4421ebf9fcbc157921144ffe7cf)