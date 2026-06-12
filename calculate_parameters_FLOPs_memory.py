import torch
from model_TSCP_Vit import TSCP_Vit
from model_TSCP_Vit_fc import TSCP_Vit_fc
import timm
from transformers import CvtConfig, CvtForImageClassification
from torch.utils.flop_counter import FlopCounterMode

def bytes_to_mib(num_bytes):
    """Convert bytes to MiB."""
    return num_bytes / 1024 ** 2


if __name__ == "__main__":
    img_size = 224
    in_channel = 3
    num_classes = 100

    patch_size = (16, 16)
    squeeze_rate = (4, 4)
    embed_channel = 72
    num_heads = 12
    num_layers = 12
    ffn_channel = 288

    model = TSCP_Vit(
        img_size=img_size,
        patch_size=patch_size,
        squeeze_rate=squeeze_rate,
        in_channel=in_channel,
        embed_channel=embed_channel,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_channel=ffn_channel,
        num_classes=num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires a CUDA device to measure GPU memory.")

    model.to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"parameter counts are {total / 1e6}M")

   
    x = torch.zeros(1, in_channel, img_size, img_size, device=device)

    
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize(device)
    del y

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize(device)

    peak_memory = torch.cuda.max_memory_allocated(device)
    print(f"peak memory is {bytes_to_mib(peak_memory)}MiB")

    del y
    torch.cuda.empty_cache()

   
   
    with FlopCounterMode(display=False, depth=None) as flop_counter:
        _ = model(x)

    FLOPs = flop_counter.get_total_flops()
    print(f"FLOPs are {FLOPs / 1e9} G")
