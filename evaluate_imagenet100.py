import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def evaluate_the_model(model, device, test_loader):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_size = target.size(0)
            total += batch_size
            top1 = output.argmax(dim=1)            
            correct_top1 += (top1 == target).sum().item()
            _, topk_idx = output.topk(k=5, dim=1, largest=True, sorted=True)
            in_topk = topk_idx.eq(target.view(-1, 1))  
            correct_top5 += in_topk.any(dim=1).sum().item()

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    print(f"Top-1 Accuracy: {top1_acc}")
    print(f"Top-5 Accuracy: {top5_acc}")
    

if __name__ == "__main__":
    
    batchsize=128 
    img_size =224

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_root = './imagenet-100/val'
    val_dataset = datasets.ImageFolder(
        root=val_root,
        transform=transform_val
    )

    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False,num_workers=8,pin_memory=True)
  
    model =torch.load("TSCP_Vit_base_300_imagenet100.pth",weights_only=False)
    
    device = torch.device("cuda")
    model.to(device)
    evaluate_the_model(model, device,val_loader)
     