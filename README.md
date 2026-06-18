***复现：基于 CNN 的汉语语音情感识别项目***

**项目简介**

本项目为人工智能课程算法复现作业，完整实现 **MFCC 声学特征 + 轻量化卷积神经网络（EmotionCNN）** 六分类汉语语音情感识别。

依托 CASIA 汉语情感语音数据集完成模型训练、早停收敛、单音频离线推理，全部代码模块化拆分，按照文档步骤可完整复现全部实验流程。

可识别 6 类语音情绪：

- angry（愤怒）
- fear（恐惧）
- happy（开心）
- neutral（中性）
- sad（悲伤）
- surprise（惊喜）

**一、环境依赖配置**

**1. 基础运行环境**

- Python 版本：3.9 / 3.10
- 兼容系统：Windows 10/11、Linux、MacOS
- 硬件适配：自动检测 CUDA 显卡加速，无独立 GPU 也可使用 CPU 完成完整训练

**2. 依赖库一键安装命令**

打开项目根目录终端，复制执行以下命令安装全部所需库：

pip install torch torchvision librosa numpy matplotlib



**各依赖库作用说明**

- torch：PyTorch 深度学习框架，用于搭建 CNN、完成训练与梯度反向传播
- librosa：音频处理核心库，负责加载 wav 音频、提取 MFCC 特征
- numpy：数值矩阵运算，存储、处理语音频谱特征数组
- matplotlib：可选工具，用于绘制 MFCC 热力图、训练损失变化曲线

**二、数据准备说明**

**1. 数据集来源**

实验使用 CASIA 公开汉语情感语音数据集，数据集包含 2 男 2 女朗读语音，总计 1200 条均衡样本，6 类情绪各 200 条，音频统一为单声道 wav 格式。

数据集下载仓库地址：[(https://aistudio.baidu.com/datasetdetail/209512)](https://aistudio.baidu.com/datasetdetail/209512)]

**2. 数据集存放规范**

在项目根目录新建 data 文件夹，将解压后的 CASIA 数据集放入该目录，严格遵循以下层级结构：

voice-emotion-recognize/

└── data/

    └── CASIA/

        ├── angry/

        ├── fear/

        ├── happy/

        ├── neutral/

        ├── sad/

        └── surprise/

替代方案：若无完整数据集，可自行录制单条 wav 音频命名为 test.wav，仅用于离线推理测试。

**3. 自动预处理流程**

代码运行时自动完成全部音频标准化处理：

- 统一音频采样率为 16000Hz
- 提取 40 维 MFCC 梅尔倒谱声学特征
- 所有语音时序统一至 300 帧：短时音频末尾补零，长音频截断前 300 帧
- 维度转换：(时间帧, 特征数) → (1, 40, 300)，适配 2D 卷积输入格式

**三、完整运行步骤**

**步骤 1：模型训练，生成权重文件 emotion_cnn.pth**

终端定位至项目根目录，执行训练脚本：

python train.py

**训练内置规则**

- 数据集自动按 8:2 比例随机拆分训练集、测试集
- 固定超参数：批次大小 batch_size=32、Adam 优化器学习率 lr=0.001、最大迭代 40 轮
- 早停策略：单轮平均训练损失低于 0.05 时自动终止训练，防止过拟合
- 训练结束自动生成模型权重 emotion_cnn.pth 保存至项目根目录
- 终端实时输出每一轮平均损失，直观查看模型收敛速度

**步骤 2：单条语音离线情感预测**

将待识别音频重命名为 test.wav，放置项目根目录。

终端执行推理脚本：

python test.py


程序自动加载训练完成的模型，输出预测情绪，示例打印结果：

Predicted emotion: fear


**四、项目完整文件结构**

voice-emotion-recognize/

├── README.md              # 项目完整运行说明文档

├── utils.py               # 音频MFCC特征提取工具模块

├── dataset.py             # CASIA数据集自定义加载类

├── model.py               # EmotionCNN卷积网络定义

├── train.py               # 模型训练主程序（含早停逻辑）

├── test.py                # 单条wav语音推理脚本

├── emotion_cnn.pth        # 训练后生成模型权重文件

├── test.wav               # 自定义测试音频（用户自行添加）

├── .gitignore             # Git上传忽略配置文件

└── data/

    └── CASIA/             # CASIA语音数据集存放文件夹
