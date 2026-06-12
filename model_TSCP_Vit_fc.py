import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _to_2tuple(x, name: str):
    if isinstance(x, (tuple, list)):
        if len(x) != 2:
            raise ValueError(f"{name} must be a tuple/list of length 2, got {x}.")
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))
    

class NRG_GELU_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        z = (x - a) * (b**2)
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        y = x * cdf
        ctx.save_for_backward(x, a, b)  
        return y
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, a, b = ctx.saved_tensors
        z = (x - a)*(b**2)
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 *math.pi)
        grad_x = grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_out * (cdf + (b**2)*x*pdf)
        if ctx.needs_input_grad[1]:
            ga = grad_out * (-(b**2)*x*pdf)
            grad_a =ga.sum().to(a).reshape_as(a)
        if ctx.needs_input_grad[2]:
            gb = grad_out * (2*b*x*(x-a)*pdf)
            grad_b = gb.sum().to(b).reshape_as(b)
        return grad_x, grad_a, grad_b

class NRG_GELU(torch.nn.Module):
    def __init__(self):
        super(NRG_GELU,self).__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.ones(1))
    def forward(self, x):
        return NRG_GELU_Function.apply(x, self.a, self.b)



class PatchEmbedding(nn.Module):
    def __init__(self, patch_size,IN_channel,embed_dim):
        super(PatchEmbedding, self).__init__()
        self.embed_dim=embed_dim
        self.W_embedding = nn.Conv2d(in_channels=IN_channel, out_channels=embed_dim,kernel_size=patch_size, stride=patch_size, padding=0)
       
    def forward(self, X):
        batch_size= X.size(0)
        X=self.W_embedding(X) 
        X = X.permute(0, 2, 3, 1).contiguous()         
        output=X.reshape(batch_size,X.size(1)*X.size(2),self.embed_dim)
        return output




class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention,self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim// num_heads
        
        self.Wq= nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.Wk = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.Wv = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.Wo= nn.Linear(in_features=embed_dim, out_features=embed_dim)
        
    def forward(self,X,Y1,Y2):
        batch_size, N_X, _ = X.size()
        batch_size, N_Y1, _ = Y1.size() 
        batch_size, N_Y2, _ = Y2.size() 
        
        if N_Y1!=N_Y2:
            raise ValueError("token number Y1 and token number Y2 must be same.")
            
        N_Y=N_Y1
        Q=self.Wq(X)
        K=self.Wk(Y1)
        V=self.Wv(Y2)
          
        Q=Q.reshape(batch_size, N_X,self.num_heads,self.head_dim)
        K=K.reshape(batch_size, N_Y, self.num_heads, self.head_dim)
        V=V.reshape(batch_size, N_Y, self.num_heads, self.head_dim)
        Q=torch.transpose(Q,1,2)
        K=torch.transpose(K, 1, 2)
        V=torch.transpose(V,1,2)

        
        output = F.scaled_dot_product_attention(Q, K, V,attn_mask=None,dropout_p=0.0,is_causal=False)
        output=torch.transpose(output,1,2)
        output = output.reshape(batch_size,N_X,self.num_heads*self.head_dim)
        output = self.Wo(output)
        
        return output


class TSCP_Vit_fc_block(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super(TSCP_Vit_fc_block, self).__init__()
        self.attention =MultiheadAttention(embed_dim, num_heads)
        self.layernorm_1=nn.LayerNorm(normalized_shape=embed_dim)
        self.layernorm_2=nn.LayerNorm(normalized_shape=embed_dim)
        self.layernorm_3=nn.LayerNorm(normalized_shape=embed_dim)
        self.layernorm_4=nn.LayerNorm(normalized_shape=embed_dim)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ffn_dim),
            NRG_GELU(),
            nn.Linear(in_features=ffn_dim, out_features=embed_dim),
        )

    def forward(self, X,cls):
        batch_size, N, _ = X.size()
        ln1=self.layernorm_1(X)
        X=self.attention(ln1,ln1,ln1)+X
        
        ln2=self.layernorm_2(X)
        ffn_output=self.FFN(ln2)
        output=ffn_output+X

        
        update_cls_mean=output.mean(dim=1, keepdim=True)  
        update_cls_std=output.std(dim=1, keepdim=True)
        update_cls=torch.cat([update_cls_mean,update_cls_std],dim=1)
        
        cls=cls+update_cls
        ln3=self.layernorm_3(cls)
        cls=self.attention(ln3,ln3,ln3)+cls
        
        cls=cls+update_cls
        ln4=self.layernorm_4(cls)
        cls=self.attention(ln4,output,output)+cls

        return output,cls




class TSCP_Vit_fc(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_channel=3, embed_dim=360, num_heads=12, num_layers=12,ffn_dim=1440,num_classes=1000):
        super(TSCP_Vit_fc, self).__init__()
        self.img_size = _to_2tuple(img_size, "img_size")           
        self.patch_size = _to_2tuple(patch_size, "patch_size")     
        self.in_channel=in_channel
        self.embed_dim=embed_dim

        H, W = self.img_size
        ph, pw = self.patch_size
       
        if H % ph != 0 or W % pw != 0:
            raise ValueError("img_size must be divisible by patch_size (both H and W).")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

            
        self.N = (H // ph) * (W // pw)

        
        self.embedding=PatchEmbedding(self.patch_size,self.in_channel, self.embed_dim)
        self.initial_cls=nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.N, self.embed_dim))

        
        self.TSCP_Vit_fc_block_list = nn.ModuleList()
        for _ in range(num_layers):
            self.TSCP_Vit_fc_block_list.append(TSCP_Vit_fc_block(self.embed_dim, num_heads, ffn_dim))
            
        self.classifier = nn.Linear(2*self.embed_dim, num_classes)
       
    def forward(self, X):
        if X.dim() != 4:
            raise ValueError(f"Input must be 4D tensor (B,C,H,W), got shape {tuple(X.shape)}.")
        batch_size, c, h, w = X.shape
        if (h, w) != self.img_size:
            raise ValueError(f"Input image size {(h, w)} does not match model img_size {self.img_size}.")
        if c != self.in_channel:
            raise ValueError(f"Input channel {c} does not match model in_channel {self.in_channel}.")
            
        X = self.embedding(X)
        positional_encoding=self.positional_encoding
        positional_encoding=positional_encoding.expand(batch_size,-1,-1)    
        X=X+positional_encoding
        
        cls=self.initial_cls
        cls=cls.expand(batch_size,-1,-1)
        
        for block in self.TSCP_Vit_fc_block_list:
            X,cls = block(X,cls)
            
        cls=cls.reshape(batch_size,2*self.embed_dim)
        output=self.classifier(cls)
        return output