
# YOLOv11 车牌识别系统

本项目实现了一个基于改进 YOLOv11 架构和 PaddleOCR 的鲁棒两阶段车牌识别系统。该系统专为应对光照剧烈变化和遮挡等复杂环境而设计。为了提升检测精度，我们在模型中集成了大选择性核（LSK）、高效多尺度注意力（EMA）和轻量级特征融合模块（LightweightFFM）。

## Requirements: software

为了运行本项目，请确保您的环境满足以下要求：

**操作系统**：Windows、Linux 或 macOS
**Python 版本**：Python 3.8 或更高版本
**依赖库**：

请使用提供的 `requirements.txt` 文件安装必要的 Python 包。核心依赖包括 `ultralytics`、`paddlepaddle`、`paddleocr` 和 `opencv-python`。

请在终端中运行以下命令进行安装：

```bash
pip install -r requirements.txt

```

## Pretrained models

本仓库包含推理所需的模型配置和权重文件。

1. **训练权重**：
模型训练完成后，性能最佳的权重文件保存在 `runs/` 目录下。
* 路径示例：`runs/train/exp/weights/best.pt`
* 注意：`runs/train/` 下的具体子目录名称（如 `exp`）可能因您的训练实验名称而异，请根据实际情况查找。


2. **基础模型**：
项目使用根目录下的 `yolo11n.pt` 作为迁移学习的基准模型。
3. **模块集成**：
改进模型架构所需的自定义模块源代码位于根目录：
* `LSKblock.py`
* `EMA.py`
* `LightweightFFM.py`



## Preparation for testing

在运行测试脚本之前，请按照以下步骤配置环境并准备数据。

### 1. 数据准备

* **测试图片**：请将您的测试图片放置在根目录下。默认的测试文件名为 `test.jpg`。
* **数据集配置**：`CCPD.yaml` 文件定义了训练和验证的数据集路径。如果您打算运行评估指标或重新训练，请确保 `dataset` 文件夹包含必要的图像数据。

### 2. 配置检查

打开 `preparation for testing.py` 文件并验证模型路径。请确保代码中的路径指向您实际训练好的权重文件（`.pt` 文件）。

`preparation for testing.py` 中的配置示例：

```python
# 确保此路径指向您 runs 目录下训练好的 .pt 文件
model_path = 'runs/train/weights/best.pt' 
image_path = 'test.jpg'

```

### 3. 运行测试

执行测试脚本以检测并识别目标图像中的车牌。

**运行命令：**

```bash
python "preparation for testing.py"

```

### 4. 输出结果

处理结果将保存在 `results` 目录下。输出包含：

* 绘制了检测框和识别文本的图像文件。
* 控制台输出，显示识别出的车牌号码和置信度分数。

## Model Training (Optional)

如果您希望在 CCPD 数据集上重新训练模型，请使用 `train.py` 脚本。

**运行命令：**

```bash
python train.py

```

在开始训练过程之前，请确保 `CCPD.yaml` 文件中的路径正确指向您的 `dataset/` 目录。

## Project Structure

* **preparation for testing.py**：单张图像推理的主入口脚本。
* **train.py**：用于训练 YOLOv11 模型的脚本。
* **LSKblock.py / EMA.py / LightweightFFM.py**：改进网络模块的源代码。
* **CCPD.yaml**：数据集路径配置文件。
* **requirements.txt**：Python 依赖包列表。
* **dataset/**：包含训练和验证数据的目录。
* **results/**：保存检测和识别结果的目录。
* **runs/**：包含训练日志和训练好模型权重的目录。
* **ultralytics/**：YOLOv11 核心库文件。