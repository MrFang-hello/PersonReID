import torch 
import torchvision.transforms as transforms
from torch.autograd import Variable 
import random 
import math 
import time
# Top-k准确率
def accuracy(output,target,topk=(1,)):
    '''
    输入output: type为tensor,shape为m*n;
    表示有m个n维的向量,元素为类别的概率
    输入target: type为tensor,shape为(m,)
    表示样本的标签,如[0,0,0,0,7,7,7,7]
    输入topk: type为tuple,元素为k
    输出: type为list,表示top-k对应的准确率
    '''
    maxk=max(topk)  # 最大k
    batch_size=target.size(0)   # 样本个数
    # 获取top-k的概率以及类别
    _,pred=output.topk(maxk,dim=1,largest=True,sorted=True)
    pred=pred.t()
    # top-k的类别与真实标签做比较
    correct=pred.eq(target.view(1,-1).expand_as(pred))
    ret=[]  # 记录top-k准确率
    for k in topk:  # 遍历每一个k
        # 统计top-k预测正确的样本数量
        correct_k=correct[:k].contiguous().view(-1).float().sum(dim=0,keepdim=True)
        # 计算top-k准确率
        ret.append(correct_k.mul_(1./batch_size))
    return ret
# 随机擦除(RandomErasing)
class RandomErasing(object):
    def __init__(self,probability=0.5,sl=0.02,sh=0.4,r1=0.3,mean=[0.,0.,0.]):
        self.probability=probability    # 执行随机擦除操作的概率
        self.sl=sl      # 最小擦除面积占原图的比例
        self.sh=sh      # 最大擦除面积占原图的比例
        self.r1=r1      # 擦除区域的最小长宽比
        self.mean=mean  # 用于替换的像素值
    def __call__(self,img):
        '''
        输入img: type为Tensor,shape为C*H*W
        输出: type为Tensor,shape为C*H*W
        '''
        # 从均匀分布的[0,1)区间等概率取值,若大于0.5则不擦除
        if random.uniform(0,1)>self.probability:
            return img
        # 若小于等于0.5,则执行擦除操作
        for _ in range(100):    # 尝试100次
            area=img.size()[1]*img.size()[2]    # 原图面积H*W
            target_area=random.uniform(self.sl,self.sh)*area    # 擦除区域的面积h*w,取值为[0.02*H*W,0.4*H*W)
            aspect_ratio=random.uniform(self.r1,1/self.r1)      # 擦除区域的长宽比h/w,取值为[0.3,3.33]
            h=int(round(math.sqrt(target_area*aspect_ratio)))   # 擦除区域的高h
            w=int(round(math.sqrt(target_area/aspect_ratio)))   # 擦除区域的宽w
            # 找到合适的擦除区域h*w
            if w<img.size()[2] and h<img.size()[1]:
                x1=random.randint(0,img.size()[2]-w)    # 擦除区域左上角x坐标
                y1=random.randint(0,img.size()[1]-h)    # 擦除区域左上角y坐标
                if img.size()[0]==3:    # 3通道彩色图
                    img[0,y1:y1+h,x1:x1+w]=self.mean[0] # 0号通道擦除区域的像素值替换为0
                    img[1,y1:y1+h,x1:x1+w]=self.mean[1] # 1号通道擦除区域的像素值替换为0
                    img[2,y1:y1+h,x1:x1+w]=self.mean[2] # 2号通道擦除区域的像素值替换为0
                else:   # 单通道灰度图
                    img[0,y1:y1+h,x1:x1+w]=self.mean[0] # 擦除区域的像素值替换为0
                return img
        return img
# 数据记录器
class AverageMeter():
    def __init__(self):
        self.val=0      # 当前值
        self.avg=0      # 平均值
        self.sum=0      # 累加值
        self.count=0    # 累加个数
    # 重置
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    # 更新,val表示当前值,n表示有几个当前值
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count
# 训练过程
class ImgTrainer():
    def __init__(self,model,criterion,summary_writer=None):
        self.model=model    # 模型
        self.criterion=criterion    # [交叉熵损失,三元组损失]
        self.summary_writer=summary_writer  # Tensorboard
        # 数据增强—标准化
        self.normlizer=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 数据增强—随机擦除
        self.eraser=RandomErasing(probability=0.5,mean=[0.,0.,0.])
    def train(self,epoch,data_loader,optimizer,random_erasing=True,print_freq=1):
        '''
        输入epoch:迭代次数;
        输入data_loader:训练集;
        输入optimizer:优化器;
        输入random_erasing:是否随机擦除;
        输入print_freq:打印频率;
        '''
        self.model.train()
        data_time=AverageMeter()    # batch读取时间记录器
        batch_time=AverageMeter()   # batch处理时间记录器
        precisions=AverageMeter()   # Top-k准确率记录器
        end=time.time()     # 获取当前时间
        for i,inputs in enumerate(data_loader):     # 读取数据,batch=8
            data_time.update(time.time()-end)   # 记录读取一个batch的时间
            ori_inputs,targets=self._parse_data(inputs)     # ori_inputs:8*3*256*128; targets:(8,)
            # 对一个batch的8张图片做数据增强(标准化;随机擦除)
            in_size=inputs[0].size()    # 8*3*256*128
            for j in range(in_size[0]): # 遍历8张图片
                ori_inputs[j,:,:,:]=self.normlizer(ori_inputs[j,:,:,:]) # 标准化
                if random_erasing:      # 随机擦除
                    ori_inputs[j,:,:,:]=self.eraser(ori_inputs[j,:,:,:])
            # 将一个batch的数据输入网络,根据结果计算损失和准确率
            loss,all_loss,prec=self._forward(ori_inputs,targets)
            # 记录Top-k准确率
            precisions.update(prec,targets.size(0))
            # 利用Tensorboard记录损失
            if self.summary_writer is not None:
                global_step=epoch*len(data_loader)+i
                self.summary_writer.add_scalar("loss",loss.item(),global_step)
                self.summary_writer.add_scalar("loss_cls",all_loss[0],global_step)
                self.summary_writer.add_scalar("loss_tri",all_loss[1],global_step)
                self.summary_writer.add_scalar("precision",prec,global_step)
            # 反向传播;优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录处理一个batch的时间
            batch_time.update(time.time()-end)
            end=time.time()     # 更新当前时间
            # 每处理一个batch就打印信息:
            if (i+1)%print_freq==0:
                print("Epoch:[{}][{}/{}]\t"
                        "DataTime{:.3f}({:.3f})\t"
                        "BatchTime{:.3f}({:.3f})\t"
                        "Loss_cls{:.3f} Loss_tri{:.3f}\t"
                        "prec{:.2%}({:.2%})\t"
                        .format(epoch,i+1,len(data_loader),
                        data_time.val,data_time.avg,
                        batch_time.val,batch_time.avg,
                        all_loss[0],all_loss[1],
                        precisions.val,precisions.avg))
            if (epoch+1)%10==0 and (epoch+1)>=300:
                fpath="D:\\MrFang\\TX2\\ReID\\PersonReID2\\weights\\"+"epoch"+str(epoch)+"_resnet50_rga.pt"
                torch.save(self.model.module.state_dict(),fpath)
    def _parse_data(self,inputs):
        '''
        输入inputs:[imgs,fnames,pids,camids],长度为4的list;
        imgs: type为Tensor,shape为8*3*256*128;
        fnames: 长度为8的list,记录图片的路径;
        pids: 长度为8的一维Tensor,记录行人的标号;
        camid: 长度为8的一维Tensor,记录摄像头的标号;
        '''
        # 取出imgs和pids
        imgs,_,pids,_=inputs
        inputs=Variable(imgs)
        targets=Variable(pids.cuda())
        return inputs,targets
    def _forward(self,inputs,targets):
        '''
        输入inputs: type为Tensor,shape为8*3*256*128;
        输入targets: type为Tensor,shape为(8,)
        输出(loss,losses,prec):
        loss:交叉熵损失+三元组损失,type为Tensor
        losses:[交叉熵损失,三元组损失],type为list,元素为Tensor
        prec:Top-k准确率,type为Tensor
        '''
        outputs=self.model(inputs,training=True)    # (feat1,feat2,cls_feat)
        loss_cls=self.criterion[0](outputs[2],targets)  # 对cls_feat计算交叉熵损失
        loss_tri=self.criterion[1](outputs[0],targets)  # 对feat1计算三元组损失
        loss=loss_cls+loss_tri      # 总损失值
        losses=[loss_cls,loss_tri]  # [交叉熵损失,三元组损失]
        # 根据预测结果cls_feat和标签targets计算Top-k准确率
        prec,=accuracy(outputs[2].data,targets.data)
        prec=prec[0]
        return loss,losses,prec