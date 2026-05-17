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
    

class adaptive_GELU_Function(torch.autograd.Function):
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

class adaptive_GELU(torch.nn.Module):
    def __init__(self):
        super(adaptive_GELU,self).__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.ones(1))
    def forward(self, x):
        return adaptive_GELU_Function.apply(x, self.a, self.b)



class TSCP_Vit_PatchEmbedding(nn.Module):
    def __init__(self, patch_size,squeeze_rate,IN_channel, embed_channel):
        super(TSCP_Vit_PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_channel = embed_channel
        self.squeeze_rate=squeeze_rate
        self.W_embedding = nn.Conv2d(in_channels=IN_channel, out_channels=embed_channel,kernel_size=5, stride=1, padding=2)
        self.squeeze = nn.Conv2d(in_channels=embed_channel,
                                 out_channels=embed_channel,
                                 kernel_size=squeeze_rate,
                                 stride=squeeze_rate,
                                 padding=0,
                                 groups=embed_channel)

    def forward(self, X):
        batch_size, IN_channel, height, width = X.size()
        X = X.unfold(2, self.patch_size[0],self.patch_size[0]).unfold(3, self.patch_size[1],self.patch_size[1])  
        X = X.permute(0, 2, 3, 1, 4, 5).contiguous()                                   
        N=X.size(1)*X.size(2)
        X=X.reshape(batch_size,X.size(1)*X.size(2),IN_channel,self.patch_size[0],self.patch_size[1])                    
        X=X.reshape(batch_size*N,IN_channel,self.patch_size[0],self.patch_size[1])
        X=self.W_embedding(X)
        X=self.squeeze(X)
        output = X.reshape(batch_size, N, *X.shape[1:] )
        return output


class TSCP_Vit_MultiheadAttention(nn.Module):
    def __init__(self, embed_channel, num_heads):
        super(TSCP_Vit_MultiheadAttention,self).__init__()
        self.num_heads = num_heads
        self.embed_channel= embed_channel
        self.head_channel = embed_channel// num_heads
        self.Wq= nn.Conv2d(in_channels=embed_channel, out_channels=embed_channel,kernel_size=5, stride=1, padding=2)
        self.Wk = nn.Conv2d(in_channels=embed_channel, out_channels=embed_channel,kernel_size=5, stride=1, padding=2)
        self.Wv = nn.Conv2d(in_channels=embed_channel, out_channels=embed_channel,kernel_size=5, stride=1, padding=2)
        self.Wo= nn.Conv2d(in_channels=embed_channel, out_channels=embed_channel,kernel_size=5, stride=1, padding=2)
    def forward(self,X,Y1,Y2):
        batch_size, N_X, embed_channel,token_h,token_w = X.size()
        batch_size, N_Y1, embed_channel,token_h,token_w = Y1.size() 
        batch_size, N_Y2, embed_channel,token_h,token_w = Y2.size() 
        
        if N_Y1!=N_Y2:
            raise ValueError("token number Y1 and token number Y2 must be same.")
            
        N_Y=N_Y1
        X=X.reshape(batch_size*N_X,embed_channel,token_h,token_w) 
        Y1=Y1.reshape(batch_size*N_Y,embed_channel,token_h,token_w)  
        Y2=Y2.reshape(batch_size*N_Y,embed_channel,token_h,token_w) 
        Q=self.Wq(X)
        K=self.Wk(Y1)
        V=self.Wv(Y2)
        Q=Q.reshape(batch_size, N_X, embed_channel,token_h,token_w)
        K=K.reshape(batch_size, N_Y, embed_channel,token_h,token_w)
        V=V.reshape(batch_size, N_Y, embed_channel,token_h,token_w)  
        Q=Q.reshape(batch_size, N_X,self.num_heads,self.head_channel,token_h,token_w)
        K=K.reshape(batch_size, N_Y, self.num_heads, self.head_channel, token_h, token_w)
        V=V.reshape(batch_size, N_Y, self.num_heads, self.head_channel, token_h, token_w)
        Q=torch.transpose(Q,1,2)
        K=torch.transpose(K, 1, 2)
        V=torch.transpose(V,1,2)
        Q = Q.reshape(batch_size, self.num_heads,N_X,self.head_channel*token_h*token_w)
        K = K.reshape(batch_size, self.num_heads, N_Y, self.head_channel*token_h*token_w)
        V=V.reshape(batch_size, self.num_heads, N_Y, self.head_channel*token_h*token_w)
        output = F.scaled_dot_product_attention(Q, K, V,attn_mask=None,dropout_p=0.0,is_causal=False)
        output=output.reshape(batch_size, self.num_heads,N_X,self.head_channel,token_h,token_w)
        output=torch.transpose(output,1,2)
        output = output.reshape(batch_size,N_X,self.num_heads*self.head_channel, token_h, token_w)
        output=output.reshape(batch_size*N_X,embed_channel,token_h,token_w)
        output = self.Wo(output)
        output = output.reshape(batch_size,N_X, embed_channel, token_h, token_w)
        return output


class TSCP_Vit_block(nn.Module):
    def __init__(self, squeezed_patch_size,embed_channel, num_heads, mlp_channel):
        super(TSCP_Vit_block, self).__init__()
        self.squeezed_patch_size=squeezed_patch_size
        self.embed_channel=embed_channel
        self.TSCP_Vit_attention =TSCP_Vit_MultiheadAttention(self.embed_channel, num_heads)
        self.layernorm_1=nn.LayerNorm(normalized_shape=[self.embed_channel, self.squeezed_patch_size[0], self.squeezed_patch_size[1]])
        self.layernorm_2=nn.LayerNorm(normalized_shape=[self.embed_channel, self.squeezed_patch_size[0], self.squeezed_patch_size[1]])
        self.layernorm_3=nn.LayerNorm(normalized_shape=[self.embed_channel, self.squeezed_patch_size[0], self.squeezed_patch_size[1]])
        self.layernorm_4=nn.LayerNorm(normalized_shape=[self.embed_channel, self.squeezed_patch_size[0], self.squeezed_patch_size[1]])
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_channel, out_channels=mlp_channel, kernel_size=5, stride=1, padding=2),
            adaptive_GELU(),
            nn.Conv2d(in_channels=mlp_channel, out_channels=self.embed_channel, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, X,cls):
        batch_size, N, _, _, _ = X.size()
        ln1=self.layernorm_1(X)
        X=self.TSCP_Vit_attention(ln1,ln1,ln1)+X
        
        ln2=self.layernorm_2(X)
        ffn_input=ln2.reshape(batch_size*N, self.embed_channel, self.squeezed_patch_size[0], self.squeezed_patch_size[1])
        ffn_output=self.mlp(ffn_input)
        ffn_output=ffn_output.reshape(batch_size,N, self.embed_channel, self.squeezed_patch_size[0], self.squeezed_patch_size[1])
        output=ffn_output+X

        
        update_cls_mean=output.mean(dim=1, keepdim=True)  
        update_cls_std=output.std(dim=1, keepdim=True)
        update_cls=torch.cat([update_cls_mean,update_cls_std],dim=1)
        
        cls=cls+update_cls
        ln3=self.layernorm_3(cls)
        cls=self.TSCP_Vit_attention(ln3,ln3,ln3)+cls
        
        cls=cls+update_cls
        ln4=self.layernorm_4(cls)
        cls=self.TSCP_Vit_attention(ln4,output,output)+cls

        return output,cls




class TSCP_Vit(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=(16,16), squeeze_rate=(4,4),in_channel=3, embed_channel=72, num_heads=12, num_layers=12, mlp_channel=288,num_classes=1000):
        super(TSCP_Vit, self).__init__()
        self.img_size = _to_2tuple(img_size, "img_size")           
        self.patch_size = _to_2tuple(patch_size, "patch_size")     
        self.squeeze_rate = _to_2tuple(squeeze_rate, "squeeze_rate") 
        self.in_channel=in_channel
        self.embed_channel=embed_channel

        H, W = self.img_size
        ph, pw = self.patch_size
        sh, sw = self.squeeze_rate
        if H % ph != 0 or W % pw != 0:
            raise ValueError("img_size must be divisible by patch_size (both H and W).")
        if embed_channel % num_heads != 0:
            raise ValueError("embed_channel must be divisible by num_heads.")
        if sh <= 0 or sw <= 0:
            raise ValueError("squeeze_rate must be positive.")
        if sh > ph or sw > pw:
            raise ValueError("squeeze_rate must be <= patch_size (both dims).")
        if ph % sh != 0 or pw % sw != 0:
            raise ValueError("patch_size must be divisible by squeeze_rate (both dims).")

        self.N = (H // ph) * (W // pw)
        self.squeezed_patch_size = (ph // sh, pw // sw)
        self.TSCP_Vit_embedding=TSCP_Vit_PatchEmbedding(self.patch_size,self.squeeze_rate, self.in_channel, self.embed_channel)
        self.initial_cls=nn.Parameter(torch.zeros(1, 2, self.embed_channel, self.squeezed_patch_size[0],self.squeezed_patch_size[1]))
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.N, self.embed_channel,self.squeezed_patch_size[0],self.squeezed_patch_size[1]))
        
        self.TSCP_Vit_block_list = nn.ModuleList()
        for _ in range(num_layers):
            self.TSCP_Vit_block_list.append(TSCP_Vit_block(self.squeezed_patch_size,self.embed_channel, num_heads, mlp_channel))
            
        self.classifier = nn.Linear(2*self.embed_channel*self.squeezed_patch_size[0]*self.squeezed_patch_size[1] , num_classes)
       
    def forward(self, X):
        if X.dim() != 4:
            raise ValueError(f"Input must be 4D tensor (B,C,H,W), got shape {tuple(X.shape)}.")
        batch_size, c, h, w = X.shape
        if (h, w) != self.img_size:
            raise ValueError(f"Input image size {(h, w)} does not match model img_size {self.img_size}.")
        if c != self.in_channel:
            raise ValueError(f"Input channel {c} does not match model in_channel {self.in_channel}.")
            
        X = self.TSCP_Vit_embedding(X)
        positional_encoding=self.positional_encoding
        positional_encoding=positional_encoding.expand(batch_size,-1,-1,-1,-1)    
        X=X+positional_encoding
        
        cls=self.initial_cls
        cls=cls.expand(batch_size,-1,-1,-1,-1)
        
        for block in self.TSCP_Vit_block_list:
            X,cls = block(X,cls)
            
        cls=cls.reshape(batch_size,2*self.embed_channel*self.squeezed_patch_size[0]*self.squeezed_patch_size[1])
        output=self.classifier(cls)
        return output