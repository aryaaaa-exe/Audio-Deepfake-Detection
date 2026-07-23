import torch
import torch.nn as nn
import torch.nn.functional as F

#Preparing the patches for input in the transformer
class EncoderInput(nn.Module):
  def __init__(self,num_patches=160,d_model=768,patch_dim=256):
    super().__init__()

    self.patch_embed=nn.Linear( #Creates a Fully Connected layer where right now the weights are random
        in_features=patch_dim, #Each flattened patch contains 256 values
        out_features=d_model #every patch is converted into a 768 dimensional embedding
    )

    #returns a tensor filled with random numbers
    self.pos_embed=nn.Parameter(
        torch.randn(num_patches,d_model)
    )

  def forward(self,x):#x-> patches
    embedded_patches=self.patch_embed(x)
    output=embedded_patches+self.pos_embed
    return output



#Transformer Encoder Block as exactly same as used in msm-mae model
from timm.models.vision_transformer import DropPath, Mlp

class AttentionKBiasZero(nn.Module):
  #dim=embeddings=768,num_heads=nuber of attention heads; query and value have bias; dropout applied to attention weights and after the final projecttion.
  #Query and Value will be updated using learnable parameters, while intentionally leaving the Key projection without a learnable bias to match the pretrained BEiT/MAE architecture used in PS3DT.
    def __init__(self,dim,num_heads=8,qkv_bias=True,attn_drop=0.,proj_drop=0.):
        super().__init__()
        assert dim % num_heads==0 #dim should be divisible by num_heads
        self.num_heads=num_heads #stores number of heads
        head_dim=dim//num_heads #dimension of head
        self.scale=head_dim**-0.5 #scaling factor used to prevent unstable gradient

        self.qkv=nn.Linear(dim,dim*3,bias=False) #one large Linear layer computes all query,key and value together; bias is false because we will manually create the bias
        #creating the learnable bias vectors. these bias vectors will later be added to the Query and Value projections.
        if qkv_bias:
            self.q_bias=nn.Parameter(torch.zeros(dim)) #creates 768 learnable parameters initialised to zero
            self.v_bias=nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias=None
            self.v_bias=None
        #dropouts to prevent overfitting
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)

    def forward(self,x,return_attention=False):
        B,N,C=x.shape #input shape: batch size,no. of patches,embedding dim
        qkv_bias=None
        if self.q_bias is not None:         #(below)creates key bias=0
            qkv_bias=torch.cat((self.q_bias,torch.zeros_like(self.v_bias,requires_grad=False),self.v_bias)) #concatenate query,key and value bias
        qkv=nn.functional.linear(input=x,weight=self.qkv.weight,bias=qkv_bias) #linear projection
        qkv=qkv.reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2, 0, 3, 1, 4) #transforms(reshapes) and permutes
        q,k,v=qkv.unbind(0) #splits and now q,k,v are separate tensors

        #attention scores
        attn=(q@k.transpose(-2,-1))*self.scale #Q.K^T and scale is (1/sqrt(dim))^2 to prevent unstable gradient
        attn=attn.softmax(dim=-1) #converts scores into probabilities
        attn=self.attn_drop(attn) #Randomly removes some attention weights during training.
        x=(attn@v).transpose(1,2).reshape(B,N,C) #weighted sum
        x=self.proj(x) #output projection
        x=self.proj_drop(x)
        if return_attention:
          return x,attn
        return x


class BlockKBiasZero(nn.Module):

    def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=True,drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1=norm_layer(dim) #Layer normalization before attention.
        self.attn=AttentionKBiasZero(dim,num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop)
        #Implementing stochastic depth:similar to Dropout, but instead of dropping individual neurons, it drops an entire layer during training.
        self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2=norm_layer(dim) #LayerNorm before MLP.
        mlp_hidden_dim=int(dim*mlp_ratio)
        #creating feed forward network
        self.mlp=Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

    def forward(self,x,return_attention=False):
        if return_attention:
            #layernorm->multihead-self attention
            x_att,attn=self.attn(self.norm1(x),return_attention=True)
            x=x+self.drop_path(x_att)
            x=x+self.drop_path(self.mlp(self.norm2(x)))
            return x,attn

        #layernorm->attention->droppath
        x=x+self.drop_path(self.attn(self.norm1(x)))
        #layernorm->mlp->droppath
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x
    
# we want to stack 12 encoder blocks as done in the paper
from functools import partial

class Encoder(nn.Module):
  def __init__(self,num_layers=12,d_model=768,num_heads=12):
    super().__init__() # Initialize nn.Module
    #When loading pretrained weights, PyTorch matches layer names and our dataset has "blocks.0","b;ocks.1", etc so renamed "layers" to "blocks"
    self.blocks=nn.ModuleList() #creates an empty list to store PyTorch layers
    for i in range(num_layers): #each iteration creates one transformer block
      self.blocks.append(
          BlockKBiasZero( #creates transformer block
              dim=d_model,
              num_heads=num_heads,
              mlp_ratio=4,
              qkv_bias=True,
              norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
      )

  def forward(self,x):
    for blk in self.blocks: #Passes output of one encoder into next encoder.
      x=blk(x)
    return x
  
def concatenate(X):
    #X shape:(batch_size,160,768)

    B=X.shape[0] #BATCH Size
    #patches in a grid of 5*32
    num_freq_patch=5
    num_frame=32

    #5patch -> 1frames
    #(B,160,768)->(B,5,32,768) (reshaping)
    X_grid=X.view(B,num_freq_patch,num_frame,768)

    #(B,5,32,768)->(B,32,5,768)
    X_grid=X_grid.permute(0,2,1,3)

    #(B,32,5,768)->(B,32,3840)
    frame_vector=X_grid.reshape(B,num_frame,num_freq_patch*768)

    return frame_vector
  
class OutputBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential_1=nn.Sequential(
            nn.Linear(3840,768)
        )
        self.sequential_2=nn.Sequential(
            nn.Linear(768,256), #768 input and 256 neurrons
            nn.ReLU(),
            nn.Linear(256,2),
            #nn.Sigmoid() #no need of this anymore because our loss function has in built softmax func
        )

    def forward(self,x):
        x=concatenate(x) # 160 patches converted to 32frames
        x=self.sequential_1(x)# 3840 dim conveerted to 768 dim
        x=x.mean(dim=1)  # mean pooling across frames  # x has 768 dim
        x=self.sequential_2(x) # gives bonafied or synthetic prob
        return x

class PS3DT(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder_input=EncoderInput()
    self.encoder=Encoder()
    self.norm=nn.LayerNorm(768, eps=1e-6)
    self.output_block=OutputBlock()

  def forward(self,patches):
    x=self.encoder_input(patches)
    x=self.encoder(x)
    x=self.norm(x)

    probs=self.output_block(x)
    return probs