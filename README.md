基于 CNN 的汉语语音情感识别算法复现项目
项目简介
本项目为人工智能课程算法复现作业，完整实现基于 MFCC 声学特征 + 轻量化卷积神经网络（EmotionCNN）的六分类汉语语音情感识别。依托 CASIA 汉语情感语音数据集完成模型训练、早停收敛、单音频离线推理，全部代码模块化拆分，可一键复现完整实验流程。
识别情绪类别：angry(愤怒)、fear(恐惧)、happy(开心)、neutral(中性)、sad(悲伤)、surprise(惊喜)
一、环境依赖配置
1. 基础运行环境
Python 版本：3.9 / 3.10
运行系统：Windows 10/11、Linux、MacOS 全兼容
硬件支持：自动识别 CUDA GPU 加速，无 GPU 可使用 CPU 完整训练
2. 第三方库一键安装命令
打开项目根目录终端，执行以下 pip 命令安装全部依赖：
bash
运行
pip install torch torchvision librosa numpy matplotlib
依赖库功能说明：
torch：PyTorch 深度学习框架，搭建、训练卷积神经网络
librosa：音频加载、MFCC 语音特征提取核心库
numpy：数值矩阵运算，处理音频特征数组
matplotlib：可选，用于绘制 MFCC 热力图、训练损失曲线
二、数据准备说明
1. 数据集来源
实验使用 CASIA 公开汉语情感语音数据集，数据集包含 2 男 2 女共 1200 条平衡语音样本，6 类情绪每类 200 条，存储格式为 wav 单声道音频。
数据集开源参考地址：https://gitcode.com/Premium-Resources/fa78c
2. 文件夹放置规范
在项目根目录新建data文件夹；
将解压后的 CASIA 数据集放入data/内，目录层级必须严格如下：
plaintext
voice-emotion-recognize/
└── data/
    └── CASIA/
        ├── angry/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
数据集可选替代方案：若无完整 CASIA 数据集，可自行录制 wav 语音放入test.wav用于单样本推理测试。
3. 数据预处理规则（代码自动执行，无需手动操作）
统一音频采样率：16000Hz
提取 40 维 MFCC 梅尔倒谱特征
时序长度标准化至 300 帧，短音频补零、长音频截断
自动转换为 CNN 标准输入张量维度 (1, 40, 300)
三、完整运行步骤
步骤 1：模型训练（生成训练权重文件 emotion_cnn.pth）
终端进入项目根目录，执行训练脚本：
bash
运行
python train.py
训练内置逻辑说明：
数据集按 8:2 自动随机划分为训练集、测试集；
超参数：batch_size=32，Adam 优化器 lr=0.001，最大迭代 40 轮；
早停机制：单轮平均训练损失低于 0.05 自动终止训练；
训练完成后自动在根目录生成模型权重文件emotion_cnn.pth；
终端实时打印每一轮 Epoch 平均损失，可直观观察模型收敛过程。
步骤 2：单条语音情感推理测试
将待识别 wav 音频重命名为test.wav，放置项目根目录；
终端执行推理脚本：
bash
运行
python test.py
程序自动加载训练好的emotion_cnn.pth模型，输出识别情绪结果；
示例终端输出：Predicted emotion: fear
四、项目完整文件结构说明
plaintext
voice-emotion-recognize/
├── README.md              # 项目运行说明文档（本文档）
├── utils.py               # 音频预处理工具模块
├── dataset.py             # CASIA数据集自定义加载类
├── model.py               # EmotionCNN卷积网络模型定义
├── train.py               # 模型训练主程序（含早停机制）
├── test.py                # 单wav音频离线推理测试程序
├── emotion_cnn.pth        # 训练完成后生成的模型权重文件
├── test.wav               # 自定义测试语音（自行放置）
├── .gitignore             # Git忽略配置，过滤大文件、缓存
└── data/
    └── CASIA/             # CASIA情感语音数据集存放目录
