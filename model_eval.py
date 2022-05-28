from person_data import * 
from person_model import *
from person_eval import *
import torch 
if __name__=="__main__":
    # 读取数据集
    dataset=CUHK03("./data/cuhk03/splits_new_labeled.json")
    # 加载query集
    query_loader = DataLoader(
		Preprocessor(dataset.query,transform=test_transformer),
		batch_size=8, num_workers=0,
		shuffle=False, pin_memory=True)
    # 加载gallery集
    gallery_loader = DataLoader(
		Preprocessor(dataset.gallery, transform=test_transformer),
		batch_size=8, num_workers=0,
		shuffle=False, pin_memory=True)
    # 创建行人重识别网络
    model=Resnet50_RGA_Model()
    model.load_state_dict(torch.load("./weights/epoch299_resnet50_rga.pt"))
    model = nn.DataParallel(model).cuda()
    # 验证模块
    evaluator=ImgEvaluator(model)
    evaluator.do_eval(query_loader,gallery_loader,dataset.query,dataset.gallery,metric=["cosine"])