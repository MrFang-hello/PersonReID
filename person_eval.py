import torch 
import torchvision.transforms as transforms
import torch.nn.functional as F 
from collections import OrderedDict
import time
import numpy as np
from sklearn.metrics import precision_recall_curve,average_precision_score 

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

# 模型前向传播
def inference_feat(model,inputs):
    '''
    输入:
    model:  网络模型;
    inputs: type为Tensor,shape为8*3*256*128; 
    输出:
    [feat1,feat2],feat1和feat2的shape为8*2048
    '''
    model.eval()
    model_out=model(inputs,training=False)
    return [model_out[0].data.cpu(),model_out[1].data.cpu()]
# 模型提取特征
def extract_feat(model,data_loader,normlizer,feat_type=["feat1","feat2"],print_freq=1):
    '''
    输入:
    model:网络模型;
    data_loader:数据集;
    normlizer:数据增强—标准化;
    feat_type:[feat1,feat2];
    print_freq:打印频率;
    输出:
    features={"feat1":[(name1,feat1_1),(name2,feat1_2),...,(namen,feat1_n)],
                "feat2":[(name1,feat2_1),(name2,feat2_2),...,(namen,feat2_n)]}
    labels={"feat1":[(name1,label1),(name2,label2),...,(namen,labeln)],
                "feat2":[(name1,label1),(name2,label2),...,(namen,labeln)]}
    '''
    model.eval()    # 验证模式
    data_time=AverageMeter()    # 读取一个batch的时间记录器
    batch_time=AverageMeter()   # 处理一个batch的时间记录器
    features={}     # 保存特征向量
    labels={}       # 保存标签
    # 初始化features和labels 
    for feat_name in feat_type:
        features[feat_name]=OrderedDict()
        labels[feat_name]=OrderedDict()
    end=time.time()     # 当前时间
    # 读取每一个batch(8)
    for i,(imgs,fnames,pids,_) in enumerate(data_loader):
        data_time.update(time.time()-end)   # 记录读取一个batch的时间
        in_size=imgs.size()     # 8*3*256*128
        # 数据增强—标准化
        for j in range(in_size[0]):
            imgs[j,:,:,:]=normlizer(imgs[j,:,:,:])
        # 一个batch的数据做前向传播
        outputs=inference_feat(model,imgs)  # [feat1,feat2]
        '''
        features={"feat1":[(name1,feat1_1),(name2,feat1_2),...,(namen,feat1_n)],
                    "feat2":[(name1,feat2_1),(name2,feat2_2),...,(namen,feat2_n)]}
        labels={"feat1":[(name1,label1),(name2,label2),...,(namen,labeln)],
                    "feat2":[(name1,label1),(name2,label2),...,(namen,labeln)]}
        '''
        for ii,feat_name in enumerate(feat_type):
            for fname,output,pid in zip(fnames,outputs[ii],pids):
                features[feat_name][fname]=output 
                labels[feat_name][fname]=pid
        # 记录处理一个batch的时间
        batch_time.update(time.time()-end)
        end=time.time()     # 更新当前时间
        # 每处理一个batch数据就做打印
        if (i+1)%print_freq==0:
            print("Extract Features:[{}/{}]\t"
                    "Data_time:{:.3f}({:.3f})\t"
                    "Batch_time:{:.3f}({:.3f})\t"
                    .format(i+1,len(data_loader),
                            data_time.val,data_time.avg,
                            batch_time.val,batch_time.avg))
    return features,labels

# mAP计算
def mean_ap(distmat,query_ids=None,gallery_ids=None,query_cams=None,gallery_cams=None):
    '''
    输入:
    distmat:      type为Tensor,shape为1400*5328,表示代价矩阵;
    query_ids:    type为list,长度为1400,记录着query集行人的标签;
    gallery_ids:  type为list,长度为5328,记录着gallery集行人的标签;
    query_cams:   type为list,长度为1400,记录着query集摄像头的标号(1 or 2);
    gallery_cams: type为list,长度为5328,记录着gallery集摄像头的标号(1 or 2);
    输出:
    mAP值;
    '''
    distmat=distmat.cpu().numpy()   # Tensor转numpy 
    m,n=distmat.shape   # 1400,5328 
    query_ids=np.asarray(query_ids)         # numpy
    gallery_ids=np.asarray(gallery_ids)     # numpy
    query_cams=np.asarray(query_cams)       # numpy
    gallery_cams=np.asarray(gallery_cams)   # numpy 
    indices=np.argsort(distmat,axis=1)      # 按距离由小到大排序,得到索引
    # 根据行人的标签判断是否匹配成功
    matches=(gallery_ids[indices]==query_ids[:,np.newaxis])
    aps=[]  # 记录每一个AP值
    for i in range(m):  # 遍历每一个query样本
        # 从5328个距离中找出与query样本行人标签不一样,摄像头标号不一样(跨境匹配)的gallery数据
        valid=((gallery_ids[indices[i]]!=query_ids[i])|(gallery_cams[indices[i]]!=query_cams[i]))
        # 筛选出有效的样本(True--正样本;False--负样本)
        y_true=matches[i,valid]
        # 将负距离作为有效样本的置信度
        y_score=(-1)*distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        # 基于有效样本和置信度计算AP
        aps.append(average_precision_score(y_true,y_score))
    return np.mean(aps)     # 所有AP取平均,得到mAP;

# CMC计算
def cmc(distmat,query_ids=None,gallery_ids=None,query_cams=None,gallery_cams=None,topk=100):
    '''
    输入:
    topk:         前k个匹配成功率;
    distmat:      type为Tensor,shape为1400*5328,表示代价矩阵;
    query_ids:    type为list,长度为1400,记录着query集行人的标签;
    gallery_ids:  type为list,长度为5328,记录着gallery集行人的标签;
    query_cams:   type为list,长度为1400,记录着query集摄像头的标号(1 or 2);
    gallery_cams: type为list,长度为5328,记录着gallery集摄像头的标号(1 or 2);
    输出:
    CMC=[top1,top2,...,top100];
    '''
    distmat=distmat.cpu().numpy()   # Tensor转numpy 
    m,n=distmat.shape   # 1400,5328  
    query_ids=np.asarray(query_ids)         # numpy
    gallery_ids=np.asarray(gallery_ids)     # numpy
    query_cams=np.asarray(query_cams)       # numpy
    gallery_cams=np.asarray(gallery_cams)   # numpy
    indices=np.argsort(distmat,axis=1)      # 按距离从小到大排序,得到索引
    # 根据行人的标签判断是否匹配成功
    matches=(gallery_ids[indices]==query_ids[:,np.newaxis])
    ret=np.zeros(topk)      # 记录topk匹配的个数
    num_valid_queries=0     # query个数,1400
    for i in range(m):  # 遍历每一个query
        # 从5328个距离中找出与query样本行人标签不一样,摄像头标号不一样(跨境匹配)的gallery数据
        valid=((gallery_ids[indices[i]]!=query_ids[i])|(gallery_cams[indices[i]]!=query_cams[i]))
        if not np.any(matches[i,valid]):
            continue
        # 获取有效样本中正样本的索引
        index=np.nonzero(matches[i,valid])[0]
        if index[0]<topk:
            ret[index[0]]+=1    # 记录最小的正样本索引
        num_valid_queries+=1    # 查询次数累加
    return ret.cumsum()/num_valid_queries   # CMC=[top1,top2,..,top100] 

# 验证模块
class ImgEvaluator():
    def __init__(self,model):
        # 模型
        self.model=model
        # 数据增强-标准化
        self.normlizer=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def do_eval(self,query_loader,gallery_loader,query,gallery,metric=["cosine"],feat_type=["feat1","feat2"],cmc_topk=(1,5,10)):
        '''
        输入:
        query_loader:   query集加载器;
        gallery_loader: gallery集加载器;
        query:          type为list,长度为1400,表示query数据集;
        gallery:        type为list,长度为5328,表示gallery数据集;
        metric:         "euclidean"--欧式距离;"cosine"--余弦距离;
        feat_type:      ["feat1","feat2"]
        cmc_topk:       topk匹配成功率
        '''
        # 提取query集样本的特征
        query_feat_list,_=extract_feat(self.model,query_loader,self.normlizer,feat_type)
        # 提取gallery集样本的特征
        gallery_feat_list,_=extract_feat(self.model,gallery_loader,self.normlizer,feat_type)
        query_features={}
        gallery_features={}
        for feat_name in feat_type:
            # 1400*2048
            x_q=torch.cat([query_feat_list[feat_name][f].unsqueeze(0) for f,_,_ in query],0)
            # 1400*2048
            x_q=x_q.view(x_q.size(0),-1)
            '''
            query_features={"feat1":1400*2048,
                            "feat2":1400*2048}
            '''
            query_features[feat_name]=x_q
            # 5328*2048
            x_g=torch.cat([gallery_feat_list[feat_name][f].unsqueeze(0) for f,_,_ in gallery],0)
            # 5328*2048
            x_g=x_g.view(x_g.size(0),-1)
            '''
            gallery_features={"feat1":5328*2048,
                                "feat2":5328*2048}
            '''
            gallery_features[feat_name]=x_g
        # 记录query集行人的标签
        query_ids=[pid for _,pid,_ in query]
        # 记录gallery集行人的标签
        gallery_ids=[pid for _,pid,_ in gallery]
        # 记录query集摄像头的标号(1 or 2)
        query_cams=[cam for _,_,cam in query]
        # 记录gallery集摄像头的标号(1 or 2)
        gallery_cams=[cam for _,_,cam in gallery]
        for feat_name in feat_type:
            for dist_type in metric:
                print("Evaluated with '{}' features and '{}' metric:".format(feat_name,dist_type))
                x=query_features[feat_name]     # 1400*2048
                y=gallery_features[feat_name]   # 5328*2048
                m,n=x.size(0),y.size(0)     # 1400,5328 
                if dist_type=="euclidean":  # 基于欧式距离计算代价矩阵
                    XX=torch.pow(x,2).sum(1,keepdim=True).expand(m,n)     # X*X
                    YY=torch.pow(y,2).sum(1,keepdim=True).expand(n,m).t() # Y*Y
                    dist=XX+YY  # X*X+Y*Y
                    # X*X-2*X*Y+Y*Y 计算欧式距离的平方
                    dist=torch.addmm(input=dist,beta=1,mat1=x,mat2=y.t(),alpha=-2)
                    dist=dist.clamp(min=1e-12).sqrt()   # 计算欧式距离
                elif dist_type=="cosine":   # 基于余弦距离计算代价矩阵
                    x=F.normalize(x,p=2,dim=1)
                    y=F.normalize(y,p=2,dim=1)
                    dist=1-torch.mm(x,y.t())
                else:
                    raise NameError
                # mAP计算
                mAP=mean_ap(dist,query_ids,gallery_ids,query_cams,gallery_cams)
                print("Mean AP:{:.3f}".format(mAP))
                # CMC计算
                cmc_scores=cmc(dist,query_ids,gallery_ids,query_cams,gallery_cams)
                print("CMC Scores:")
                for k in cmc_topk:
                    print("top-{}:{:.3f}".format(k,cmc_scores[k-1]))
        return 