import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from torch.cuda.amp import autocast,GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_TSCP_Vit_fc import TSCP_Vit_fc
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import warnings

# 训练过程
def train_the_model(model,train_loader,optimizer,mixup,scaler,epoch):
    all_predicts_train = []
    all_targets_train = []
    model.train()
    total_loss=0
    loss_fn =SoftTargetCrossEntropy()
    for batch_idx, (data, target) in enumerate(train_loader):        
        data, target = data.to(device), target.to(device)            
        data, target =mixup(data, target)
        optimizer.zero_grad()                                                                                  
        with autocast():
            output = model(data)
            loss=loss_fn(output, target)
            total_loss = total_loss + loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    avg_loss = total_loss / len(train_loader)
    print(f'第{epoch}轮训练。 本轮各批在训练集上平均损失率为: {avg_loss}')
    
    
# 训练主函数
if __name__ == "__main__":
    learning_rate = 1e-4  # 学习率
    learning_rate_min = 1e-5 
    epochs = 300  # 训练轮数
    batchsize=128 #批大小128
    img_size =224
    num_ops= 2
    magnitude=9
    in_channel = 3  
    mixup_alpha = 1
    cutmix_alpha = 1
    label_smoothing=0.3
    
    patch_size=(16,16)
    embed_dim =360
    num_heads =12 
    num_layers =12  
    ffn_dim=1440  
    num_classes =101  

    warnings.simplefilter('ignore')
    # 数据预处理和转换
    transform_train = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # 标准化
    ])
    # 加载训练数据集，指定 root 为当前项目根目录
    train_dataset = datasets.Food101(
        root='./',
        split="train",
        download=False,  # 假设数据集已经手动放在根目录下，不需要再下载
        transform=transform_train  # 应用的预处理
    )
   
    # 使用 DataLoader 加载数据
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,num_workers=8,pin_memory=True)
    # 实例化模型
    
    model=TSCP_Vit_fc(
        img_size=img_size,
        patch_size=patch_size,
        in_channel=in_channel,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        num_classes=num_classes,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mix_up = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=None, prob=1, switch_prob=0.5, label_smoothing=label_smoothing,mode="batch", num_classes=num_classes)
    scheduler=CosineAnnealingLR(optimizer,T_max=epochs-1,eta_min=learning_rate_min)
    device = torch.device("cuda")
    model.to(device)
    scaler = GradScaler()
    for epoch in range(1, epochs + 1):
        train_the_model(model,train_loader,optimizer,mix_up, scaler,epoch)
        print(f'第{epoch}轮训练。 本轮实际应用的学习率为: {optimizer.param_groups[0]['lr']}')
        scheduler.step()
        torch.save(model,f"TSCP_Vit_fc_{epoch}.pth")
        if(epoch>1):
            os.remove(f"TSCP_Vit_fc_{epoch-1}.pth")
    print("Training complete and model saved.")