import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_TSCP_Vit import TSCP_Vit
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import warnings

def train_the_model(model,train_loader,optimizer,mixup,scaler,epoch):
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
    print(f'epochs:{epoch} avg_loss: {avg_loss}')
    
    

if __name__ == "__main__":
    learning_rate = 1e-4
    learning_rate_min = 1e-5 
    epochs = 300
    batchsize=128
    img_size =224
    num_ops= 2
    magnitude=9
    in_channel = 3  
    num_classes = 100  
    mixup_alpha = 1
    cutmix_alpha = 1
    label_smoothing=0.3
    
    patch_size=(16,16)
    squeeze_rate=(4,4) 
    embed_channel =72
    num_heads =12 
    num_layers =12  
    mlp_channel=288  

    warnings.simplefilter('ignore')

    transform_train = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761]), 
    ])

    train_dataset = datasets.CIFAR100(
        root='./',
        train=True,
        download=False,
        transform=transform_train
    )
   

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,num_workers=8,pin_memory=True)

    
    model=TSCP_Vit(
        img_size=img_size,
        patch_size=patch_size,
        squeeze_rate=squeeze_rate,
        in_channel=in_channel,
        embed_channel=embed_channel,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_channel=mlp_channel,
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
        scheduler.step()
    torch.save(model,f"TSCP_Vit_base_{epochs}_CIFAR100.pth")
    print("Training complete and model saved.")