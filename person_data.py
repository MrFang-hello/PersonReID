import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from PIL import Image
import random
import numpy as np
import json

# 自定义transforms方法
class Random2DTranslation():
    def __init__(self,height,width,p=0.5,interpolation=Image.BILINEAR):
        self.height=height  # 设置高度
        self.width=width    # 设置宽度
        self.p=p            # 变换概率
        self.interpolation=interpolation    # 双线性插值
    def __call__(self,img):
        if random.uniform(0,1)>self.p:  # 从均匀分布[0,1)中随机取一个数,若大于p,则直接resize到设置的宽高
            return img.resize((self.width,self.height),self.interpolation)
        # 若随机取的数小于p,则做随机裁剪操作
        # 设置新的宽高(比原宽高要大一些)
        new_width,new_height=int(round(self.width*1.125)),int(round(self.height*1.125))
        # 将原图resize到新的宽高
        resized_img=img.resize((new_width,new_height),self.interpolation)
        # 计算新宽高与原宽高的差值
        x_maxrange=new_width-self.width
        y_maxrange=new_height-self.height
        # 随机取点(x1,y1)
        x1=int(round(random.uniform(0,x_maxrange)))
        y1=int(round(random.uniform(0,y_maxrange)))
        # 基于随机点(x1,y1)对img做裁剪,宽高还是原设置的宽高
        croped_img=resized_img.crop((x1,y1,x1+self.width,y1+self.height))
        return croped_img

# 自定义transforms方法
class RectScale():
    def __init__(self,height,width,interpolation=Image.BILINEAR):
        self.height=height  # 设置高度
        self.width=width    # 设置宽度
        self.interpolation=interpolation    # 双线性插值
    def __call__(self,img):
        w,h=img.size    # 获取原图的宽高
        if h==self.height and w==self.width:    # 若原图宽高与设置的宽高一样,则直接返回原图
            return img
        # 若原图宽高与设置的宽高不同,则resize到设置的宽高
        return img.resize((self.width,self.height),self.interpolation)

# 设置宽高
height,width=256,128
# 封装训练集的transforms方法
train_transformer=transforms.Compose([
    Random2DTranslation(height,width),  # 以0.5的概率随机裁剪
    transforms.RandomHorizontalFlip(),  # 以默认值0.5的概率水平翻转图片
    transforms.ToTensor()   # 把图片转换为张量,并进行归一化处理,即(h,w,c)转成(c,h,w);(0-255转成0-1)
])
# 封装测试集的transforms方法
test_transformer=transforms.Compose([
    RectScale(height,width),    # resize到指定的宽高
    transforms.ToTensor()       # 把图片转换为张量,并进行归一化处理
])

# 加载json文件
def read_json(fpath):
    with open(fpath,"r") as f:
        obj=json.load(f)
    return obj

# 数据集的类
class CUHK03():
    def __init__(self,split_path):      # split_path为包含训练集和测试集信息json文件的路径
        split=read_json(split_path)[0]  # 读取json文件
        self.train=split["train"]       # 取出训练集
        self.query=split["query"]       # 取出query集
        self.gallery=split["gallery"]   # 取出gallery集
        self.num_train_pids=split["num_train_pids"]     # 训练集行人个数767
        self.num_query_pids=split["num_query_pids"]     # query集行人个数700
        self.num_gallery_pids=split["num_gallery_pids"] # gallery集行人个数700

# 继承Dataset，定义数据集
class Preprocessor(Dataset):
    def __init__(self,dataset,transform=None):
        super(Preprocessor,self).__init__()
        self.dataset=dataset        # 训练集or测试集
        self.transform=transform    # 封装的transforms方法
    def __len__(self):
        return len(self.dataset)                # 重写__len__方法
    def __getitem__(self,indices):              # 重写__getitem__方法
        if isinstance(indices,(tuple,list)):    # 一次性取出indices长度个元素
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)
    def _get_single_item(self,index):       # 获取单个元素
        fname,pid,camid=self.dataset[index] # 图片路径,行人标号,摄像头标号
        fpath=fname 
        img=Image.open(fpath).convert("RGB")    # 通过图片路径读取图片
        if self.transform is not None:          # 将图片做transforms变换
            img=self.transform(img)
        return img,fname,pid,camid              # 以元组(img,fname,pid,camid)返回

# 继承Sampler,定义数据读取的规则
class RandomIdentitySampler(Sampler):
    def __init__(self,dataset,num_instances=4):
        self.dataset=dataset                # 训练集
        self.num_instances=num_instances    # 每个行人取几张图像
        self.index_dic=defaultdict(list)    # 标记每个行人对应的数据索引
        for index,(_,pid,_) in enumerate(dataset):
            self.index_dic[pid].append(index)
        self.pids=list(self.index_dic.keys())   # 训练集行人标号
        self.num_samples=len(self.pids)         # 训练集行人个数767
    def __len__(self):      # 每个行人取4张图像,共767*4张图像
        return self.num_samples*self.num_instances
    def __iter__(self):
        indices=torch.randperm(self.num_samples)    # 随机返回一个0~766的数组
        ret=[]  # 输出
        for i in indices:           # 遍历打乱的每个行人(0~766)
            pid=self.pids[i]        # 得到行人标号
            t=self.index_dic[pid]   # 得到当前行人对应的所有数据索引(每个行人至少10个图像)
            if len(t)>=self.num_instances:  # 若当前行人对应的数据个数大于等于4,则不放回地随机取4个数据索引
                t=np.random.choice(t,size=self.num_instances,replace=False)
            else:                           # 若当前行人对应的数据个数小于4,则放回地随机取4个数据索引
                t=np.random.choice(t,size=self.num_instances,replace=True)
            ret.extend(t)                   # 将当前行人对应的4个数据索引放入ret
        return iter(ret)    # 767个行人对应的数据索引都放入ret后,将ret转成迭代器