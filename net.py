# 1. 先加上镜像（防止下载卡住，必须写在最上面）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"  # 强行关掉坑人的加速节点

#加载预训练模型
from transformers import BertModel
import torch

#定义训练设备
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 下载并加载模型
pretrained = BertModel.from_pretrained(
"bert-base-chinese",
    cache_dir="./model" # 👈 存到当前项目下的 model 文件夹里
).to(DEVICE)  #把模型“搬家”，搬到和数据、计算环境相同的设备上去。

print(pretrained)
print(pretrained.embeddings)

#定义下游任务模型（将主干网络所提取的特征进行分类）
class Model(torch.nn.Module):
    #模型结构设计
    def __init__(self):
        # super(Model, self).__init__()这在Python2里是必须的，
        # 但在Python3里，直接写super()就行了，它是个“语法糖”
        super().__init__()
        self.fc=torch.nn.Linear(768,3)
    #三个参数是bert模型的输入
    def forward(self,input_ids,attention_mask,token_type_ids):
        # 上游任务不参与训练
        # 1. 冻结底座：只让 BERT 干活，不让 BERT 改错
        with torch.no_grad():

            #模型不参与反向传播，只参与前向训练
            out=pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)

        #下游任务参与训练
        #把最后一层的每一句话的“带头大哥（[CLS]）”的特征单独拎出来,即：拿最后一层 + 找第 0 个位置
        #经过 12 层的“深思熟虑”，最后一层包含了最丰富的上下文信息，最准确。

        # 2. 正常训练：把你新加的线性层拿出来，算梯度、准备改错
        out=self.fc(out.last_hidden_state[:,0])
        return out
