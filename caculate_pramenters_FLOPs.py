import numpy as np
import torch
import timm
from model_TSCP_Vit import TSCP_Vit
from transformers import CvtConfig, CvtForImageClassification
from torch.utils.flop_counter import FlopCounterMode
if __name__ == "__main__": 
    img_size =224
    in_channel = 3  
    num_classes = 100  

    patch_size=(16,16)
    squeeze_rate=(4,4)
    embed_channel =96
    num_heads =16 
    num_layers =12  
    mlp_channel=384  

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
    
    device = torch.device("cuda")
    model.to(device)
    total = 0
    for p in model.parameters():
        n = p.numel()
        total=total+n
    print(f'prameter count:{total/1e6}M')
    x = torch.zeros(1,in_channel, img_size, img_size,device=device)
    model.eval()
    with FlopCounterMode(mods=model, display=False, depth=None) as flop_counter:
        _ = model(x)

    FLOPs = flop_counter.get_total_flops()
    print(f'FLOPs:{FLOPs/1e9} G')