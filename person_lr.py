# 学习率调节器
class LRScheduler(object):
    def __init__(self,base_lr=0.0008,step=[80,120,160,200,240,280,320,360],factor=0.5,
                warmup_epoch=20,warmup_begin_lr=0.000008,warmup_mode="linear"):
                self.base_lr=base_lr        # 基本的学习率
                self.learning_rate=base_lr  # 当前的学习率
                self.step=step              # 迭代的次数
                self.factor=factor          # 超参数
                self.warmup_epoch=warmup_epoch         # warmup的迭代次数
                self.warmup_begin_lr=warmup_begin_lr   # warmup开始的学习率
                self.warmup_final_lr=base_lr           # warmup结束的学习率
                self.warmup_mode=warmup_mode           # warmup的模式
    def update(self,epoch):
        '''
        输入epoch:当前迭代的次数;
        输出lr:当前迭代的学习率;
        '''
        # 0-19次迭代,进行warmup
        if self.warmup_epoch>epoch:
            if self.warmup_mode=="linear":      # 学习率线性增加
                self.learning_rate=self.warmup_begin_lr+(self.warmup_final_lr-self.warmup_begin_lr)*(epoch/self.warmup_epoch)
            elif self.warmup_mode=="constant":  # 学习率保持不变
                self.learning_rate=self.warmup_begin_lr
            else:
                raise NameError
        else:   # 20次迭代以后,学习率指数下降
            count=sum([1 for s in self.step if s<=epoch])
            self.learning_rate=self.base_lr*pow(self.factor,count)
        return self.learning_rate   # 根据当前的epoch,计算学习率并返回