from person_data import * 
from person_model import *
from person_loss import * 
from person_lr import * 
from person_train import *
import os 
from datetime import datetime 
from torch.utils.tensorboard import SummaryWriter
if __name__=="__main__":
    # 读取数据集
    dataset=CUHK03("D:\\MrFang\\TX2\\ReID\\PersonReID2/data/cuhk03/splits_new_labeled.json")
    # 加载训练集
    train_loader=DataLoader(
        Preprocessor(dataset.train,transform=train_transformer),
        batch_size=8,num_workers=0,
        sampler=RandomIdentitySampler(dataset.train,num_instances=4),
        pin_memory=True,drop_last=True
    )
    # 创建行人重识别网络
    model=Resnet50_RGA_Model()
    model = nn.DataParallel(model).cuda()
    # 损失函数:包括交叉熵损失和三元组损失
    criterion_cls = CrossEntropyLabelSmoothLoss().cuda()
    criterion_tri = TripletHardLoss()
    criterion = [criterion_cls, criterion_tri]
    # 学习率调节器
    lr_scheduler=LRScheduler()
    # 优化器optimizer
    s="adam"    # 优化器类型:SGD-Adam
    if hasattr(model.module,"backbone"):
        # backbone参数
        base_param_ids=set(map(id,model.module.backbone.parameters()))
        # 其他参数
        new_params=[p for p in model.parameters() if id(p) not in base_param_ids]
        # backbone参数+其他参数
        param_groups=[
            {"params":filter(lambda p:p.requires_grad,model.module.backbone.parameters()),
            "lr_mult":1.0},
            {"params":filter(lambda p:p.requires_grad,new_params),
            "lr_mult":1.0}
        ]
    else:
        param_groups=model.parameters()     # 若没有backbone,则参数放在一起
    if s=="sgd":    # SGD优化器
        optimizer=torch.optim.SGD(param_groups,lr=0.1,weight_decay=0.0005)
    elif s=="adam": # Adam优化器
        optimizer=torch.optim.Adam(param_groups,lr=0.1,weight_decay=0.0005)
    else:
        raise NameError
    # Tensorboard
    TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
    summary_writer = SummaryWriter(os.path.join("D:\\MrFang\\TX2\\ReID\\PersonReID2\\logs", 'tensorboard_log'+TIMESTAMP))
    # 训练模块
    trainer=ImgTrainer(model,criterion,summary_writer)
    # 开始迭代训练
    for epoch in range(0,600):
        # 每个epoch学习率做更新
        lr=lr_scheduler.update(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr
        print("Epoch[{}] learning rate update to {:.3e}".format(epoch,lr))
        # 每个epoch的训练
        trainer.train(epoch,train_loader,optimizer)