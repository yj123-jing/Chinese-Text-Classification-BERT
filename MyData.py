from torch.utils.data import Dataset
from datasets import load_from_disk #从磁盘加载数据

#涉及Dataset类
class Mydataset(Dataset):
    #初始化数据
    def __init__(self,split):#__init__只干一次
        #从磁盘加载数据
        self.dataset=load_from_disk(r"数据路径")
        if split=="train":
            self.dataset=self.dataset["train"]
        elif split=="test":
            self.dataset=self.dataset["test"]
        elif split=="validation":
            self.dataset=self.dataset["validation"]
        else:
            print("数据集名称错误！")
    #获取数据集大小
    def __len__(self):
        return len(self.dataset)
    #对数据做定制化处理
    def __getitem__(self, idx):#__getitem__做千万次
        item=self.dataset[idx]
        return item["text"],item["label"]#它最终吐出来的结果是：("这个酒店环境真好", 1)。

if __name__=="__main__":
    dataset=Mydataset("validation")
    print(f"验证集长度: {len(dataset)}")

    # 只打印前两条，免得刷屏
    for i in range(2):
        print(dataset[i])