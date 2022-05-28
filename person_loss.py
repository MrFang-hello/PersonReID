import torch
import torch.nn as nn
# 定义标签平滑的交叉熵损失
class CrossEntropyLabelSmoothLoss(nn.Module):
    def __init__(self,num_classes=767,e=0.1,use_gpu=True):
        super(CrossEntropyLabelSmoothLoss,self).__init__()
        self.num_classes=num_classes    # 类别个数767
        self.e=e    # 超参数e
        self.use_gpu=use_gpu    # 是否使用GPU
        self.logsoftmax=nn.LogSoftmax(dim=1)    # LogSoftmax函数
    def forward(self,inputs,targets):
        '''
        输入:
        inputs: cls_feat特征向量,type为Tensor,shape为8*767;
        targets: 1个batch中8张图像的行人索引值,取值为0~766,type为Tensor,是长度为8的一维向量
        输出:
        交叉熵损失值,type为Tensor,一个数值;
        '''
        log_probs=self.logsoftmax(inputs)   # 对cls_feat做LogSoftmax变换
        targets=torch.zeros(log_probs.size()).scatter_(1,targets.unsqueeze(1).cpu(),1)  # 索引值生成标签值
        if self.use_gpu:    # 是否使用cuda加速
            targets=targets.cuda()
        targets=(1-self.e)*targets+self.e/self.num_classes  # Label Smooth
        loss=(-targets*log_probs).mean(0).sum()     # 基于标签值和预测值计算交叉熵损失
        return loss

# L2范数归一化
def normalize(X):
    '''
    输入X: type为Tensor,shape为m*n,
    表示有m个特征向量,每个特征向量n维;
    '''
    X=X/(torch.norm(X,p=2,dim=1,keepdim=True).expand_as(X)+1e-12)
    return X

# 基于欧式距离的代价矩阵构建函数
def euclidean_dist(X,Y):
    '''
    输入X: type为Tensor,shape为m1*n,
    表示有m1个n维的特征向量
    输入Y: type为Tensor,shape为m2*n,
    表示有m2个n维的特征向量
    输出dist: type为Tensor,shape为m1*m2,
    表示两两特征向量之间的欧式距离
    '''
    m1,m2=X.size(0),Y.size(0)   # 特征向量的个数m1,m2
    XX=torch.pow(X,2).sum(1,keepdim=True).expand(m1,m2)     # X*X
    YY=torch.pow(Y,2).sum(1,keepdim=True).expand(m2,m1).t() # Y*Y
    dist=XX+YY  # X*X+Y*Y
    # X*X-2*X*Y+Y*Y 计算欧式距离的平方
    dist=torch.addmm(input=dist,beta=1,mat1=X,mat2=Y.t(),alpha=-2)
    dist=dist.clamp(min=1e-12).sqrt()   # 计算欧式距离
    return dist     # 返回m1*m2的代价矩阵

# 余弦距离函数
def cosine_dist(X,Y):
    '''
    输入X: type为Tensor,shape为m1*n
    表示有m1个n维的向量
    输入Y: type为Tensor,shape为m2*n
    表示有m2个n维的向量
    输出: m1*m2的二维余弦距离矩阵(代价矩阵)
    '''
    # 向量X做L2范数归一化
    X_normed=X/(torch.norm(X,p=2,dim=1,keepdim=True).expand_as(X)+1e-12)
    # 向量Y做L2范数归一化
    Y_normed=Y/(torch.norm(Y,p=2,dim=1,keepdim=True).expand_as(Y)+1e-12)
    # m1*m2的代价矩阵
    dist=1.0-torch.mm(X_normed,Y_normed.t())
    return dist

# Hard Negative Mining
def hard_example_mining(dist_mat,labels):
    '''
    输入dist_mat: type为Tensor,shape为m*m
    表示m*m的二维代价矩阵
    输入labels: type为Tensor,shape为(m,)
    表示长度为m的一维行人标签
    输出dist_AP: type为Tensor,shape为(m,)
    表示长度为m的一维AP距离(尽可能大)
    输出dist_AN: type为Tensor,shape为(m,)
    表示长度为m的一维AN距离(尽可能小)
    '''
    assert len(dist_mat.size())==2              # 代价矩阵二维
    assert dist_mat.size(0)==dist_mat.size(1)   # 代价矩阵长宽相等
    N=dist_mat.size(0)  # 特征向量个数
    is_P=labels.expand(N,N).eq(labels.expand(N,N).t())  # 正样本标记
    is_N=labels.expand(N,N).ne(labels.expand(N,N).t())  # 负样本标记
    dist_AP,_=torch.max(dist_mat[is_P].contiguous().view(N,-1),1,keepdim=True)  # 尽可能大的dist_AP
    dist_AN,_=torch.min(dist_mat[is_N].contiguous().view(N,-1),1,keepdim=True)  # 尽可能小的dist_AN
    dist_AP=dist_AP.squeeze(1)  # 二维到一维
    dist_AN=dist_AN.squeeze(1)  # 二维到一维
    return dist_AP,dist_AN

# Triplet Loss(含Hard Negative Mining)
class TripletHardLoss(object):
    def __init__(self,margin=0.3,metric="euclidean"):
        self.margin=margin      # 超参数a,表示距离差距
        self.metric=metric      # 欧式距离or余弦距离
        self.ranking_loss=nn.MarginRankingLoss(margin=margin)   # 三元组损失函数
    def __call__(self,feat1,labels,normalize_feature=False):
        '''
        输入feat1: type为Tensor,shape为8*2048
        表示网络输出的特征向量
        输入labels: type为Tensor,shape为(8,)
        表示batch=8的行人标号
        输入normalize_feature: 是否对feat1做L2范数归一化
        输出loss: type为Tensor,一个数值
        表示三元组的损失值
        '''
        if normalize_feature:
            feat1=normalize(feat1)      # L2范数归一化
        if self.metric=="euclidean":
            dist_mat=euclidean_dist(feat1,feat1)    # 基于欧式距离的代价矩阵
        elif self.metric=="cosine":
            dist_mat=cosine_dist(feat1,feat1)       # 基于余弦距离的代价矩阵
        else:
            raise NameError
        # 计算尽可能大的dist(A,P)和尽可能小的dist(A,N)
        dist_AP,dist_AN=hard_example_mining(dist_mat,labels)
        # 权重 [1,1,1,1,1,1,1,1]
        y=dist_AN.new().resize_as_(dist_AN).fill_(1)
        # 计算三元组损失
        loss=self.ranking_loss(dist_AN,dist_AP,y)
        return loss