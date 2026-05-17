import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from model_TSCP_Vit import TSCP_Vit


class BlockEncoder(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, X):
        batch_size, N, _, _, _ = X.size()

        ln1 = self.block.layernorm_1(X)
        X = self.block.TSCP_Vit_attention(ln1, ln1, ln1) + X

        ln2 = self.block.layernorm_2(X)
        ffn_input = ln2.reshape(
            batch_size * N,
            self.block.embed_channel,
            self.block.squeezed_patch_size[0],
            self.block.squeezed_patch_size[1],
        )
        ffn_output = self.block.mlp(ffn_input)
        ffn_output = ffn_output.reshape(
            batch_size,
            N,
            self.block.embed_channel,
            self.block.squeezed_patch_size[0],
            self.block.squeezed_patch_size[1],
        )
        output = ffn_output + X
        return output


class BlockDecoder(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, output, cls):
        update_cls_mean = output.mean(dim=1, keepdim=True)
        update_cls_std = output.std(dim=1, keepdim=True)  
        update_cls = torch.cat([update_cls_mean, update_cls_std], dim=1)

        cls = cls + update_cls
        ln3 = self.block.layernorm_3(cls)
        cls = self.block.TSCP_Vit_attention(ln3, ln3, ln3) + cls

        cls = cls + update_cls
        ln4 = self.block.layernorm_4(cls)
        cls = self.block.TSCP_Vit_attention(ln4, output, output) + cls

        return cls


if __name__ == "__main__":
    img_size = 224
    in_channel = 3
    num_classes = 100

    patch_size = (16, 16)
    squeeze_rate = (4, 4)
    embed_channel =96
    num_heads = 16
    num_layers = 12
    mlp_channel =384

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TSCP_Vit(
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
    
    model.to(device)
    model.eval()
    block0 = model.TSCP_Vit_block_list[0]
    encoder = BlockEncoder(block0).to(device).eval()
    decoder = BlockDecoder(block0).to(device).eval()

    N = model.N
    C = model.embed_channel
    th, tw = model.squeezed_patch_size

    X = torch.zeros(1, N, C, th, tw, device=device)
    cls = torch.zeros(1, 2, C, th, tw, device=device)

    with torch.no_grad():
        with FlopCounterMode(mods=encoder, display=False, depth=None) as fc_enc:
            out = encoder(X)
        enc_flops = fc_enc.get_total_flops()

        with FlopCounterMode(mods=decoder, display=False, depth=None) as fc_dec:
            _ = decoder(out, cls)
        dec_flops = fc_dec.get_total_flops()

    print(f"[Single Block] Encoder FLOPs: {enc_flops/1e9} G")
    print(f"[Single Block] Decoder FLOPs: {dec_flops/1e9} G")
