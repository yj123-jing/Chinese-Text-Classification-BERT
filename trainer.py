#学生在做练习题（带答案，边做边改错）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from MyData import Mydataset #把“数据搬运工”叫过来
from torch.utils.data import DataLoader #把“流水线打包机”拿过来
from net import Model #把“AI 大脑”请过来
from transformers import BertTokenizer,AdamW #导入分词器和优化器

#定义训练设备
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH=100

#加载分词器
token=BertTokenizer.from_pretrained("bert-base-chinese",cache_dir="./model")

#把32句话拿过来，用你的“翻译官（tokenizer）”把它们全变成数字，
# 统一长度（补齐或截断），然后再打包成一个规规矩矩的矩阵，喂给 AI。
def collate_fn(data):#(形式参数)
    sentence=[i[0] for i in data]# 把 32 句话单独抽出来，变成列表
    label=[i[1] for i in data]# 把 32 个标签单独抽出来，变成列表
    #编码语句(一次性全给你转成数字矩阵)
    data=token.batch_encode_plus(
        batch_text_or_text_pairs=sentence,# 把刚才挑出来的32句话给翻译官
        truncation=True, #超过350个字的直接砍掉
        padding="max_length", #不够350个字的，在后面补0，强制统一长度
        max_length=350, #规定最长350个token
        return_tensors="pt", #重点:把结果变成PyTorch认识的Tensor矩阵！
        return_length=True,
    )
    #把编码结果取出来
    input_ids=data["input_ids"]
    attention_mask=data["attention_mask"]
    token_type_ids=data["token_type_ids"]
    labels=torch.LongTensor(label)
    return input_ids,attention_mask,token_type_ids,labels
#创建数据集
train_dataset=Mydataset("train")
#创建DataLoader
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True, #凑不够一整箱的尾货，直接扔掉
    # 因为在深度学习里，有时候最后那个“小箱子”的数据太少，计算出来的误差（梯度）会很不稳定，
    # 容易把模型“带偏”。扔掉最后一点数据，虽然少用了几条数据，但能让训练更稳定。
    collate_fn=collate_fn
    # Mydataset 每次吐出来的是 ("这酒店真差", 0)
    #DataLoader默认的打包工人(默认的 collate_fn)只会打包数字（张量）
)

if __name__=="__main__":
    #开始训练
    print(DEVICE)
    model=Model().to(DEVICE)#把模型放到设备上去
    optimizer=AdamW(model.parameters(),lr=5e-4)#优化器,e-4=0.0001
    #定义损失函数（二分类==>交叉熵）
    loss_func=torch.nn.CrossEntropyLoss()

    model.train()#开始训练（ PyTorch 模型自带的、写死在底层框架里的一个固定函数，不能赋值）
    for epoch in range(EPOCH):
        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
            #将数据放到DEVISE上
            input_ids,attention_mask,token_type_ids,labels=(input_ids.to(DEVICE),attention_mask.to(DEVICE),
                                                            token_type_ids.to(DEVICE),labels.to(DEVICE))
            out=model(input_ids,attention_mask,token_type_ids)#前向计算得到输出
            loss=loss_func(out,labels)#计算损失
            optimizer.zero_grad#梯度清零
            loss.backward()#反向传播
            optimizer.step()#更新梯度

            if i % 5 == 0:  # 如果 i 能被 5 整除（比如第0次、第5次、第10次...）
                out = out.argmax(dim=1)  # 找出模型预测概率最大的那个类别（0或1）
                acc = (out == labels).sum().item() / len(labels)  # 算准确率
                print(epoch, i, loss.item(), acc)  # 打印：第几个大轮次，第几批数据，误差是多少，准确率是多少
        # 保存模型参数
        torch.save(model.state_dict(), f"params/{epoch}bert.pt")
        print(epoch, "参数保存成功！")
