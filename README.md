# BERT Chinese Sentiment Classification (BERT 中文情感分类)

本项目基于 PyTorch 和 Hugging Face Transformers 库，使用预训练模型 `bert-base-chinese` 在 `ChnSentiCorp` 数据集上进行微调，实现中文文本的情感二分类（正向/负向）。

## 项目结构
- `MyData.py`: 数据加载与预处理，自定义 Dataset。
- `net.py`: 模型结构定义（冻结 BERT 底座，添加全连接层）。
- `trainer.py`: 模型训练脚本，包含训练循环与模型保存。
- `bert_test.py`: 模型评估脚本，计算测试集准确率。
- `run.py`: 本地交互式推理脚本，用户可在控制台输入文本实时测试。

## 运行环境
请确保安装了以下依赖：
```bash
pip install -r requirements.txt
```
注意：运行前请确保已下载 bert-base-chinese 模型和 ChnSentiCorp 数据集，并在代码中修改为你本地的路径。