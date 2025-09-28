import torch
from torch import nn


       
 

class MemoryUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
           
        self.norm_token = nn.LayerNorm(dim)
        self.proj_1 =  nn.Linear(dim,dim)
        self.proj_2 =  nn.Linear(dim,dim)
        self.proj_3 = nn.Linear(dim,dim)  
        self.gelu = nn.GELU()
             	   
    def forward(self, x):
    
    	x = self.norm_token(x)    	
    	u, v = x, x 
    	u = self.proj_1(u)
    	u = self.gelu(u)
    	v = self.proj_2(v)
    	g = u * v
    	x = self.proj_3(g)
    	
    	
    	return x

class InteractionUnit(nn.Module):
    def __init__(self,dim,score_dim):
        super().__init__()
            
        self.norm_token = nn.LayerNorm(dim)       
        self.gelu = nn.GELU()
             	   
    def forward(self, x):
    
    	x = self.norm_token(x)
    	dim0 = x.shape[0]
    	dim1 = x.shape[1]
    	dim2 = x.shape[2]
    	x = x.reshape([dim0,dim1*dim2])
    	x = self.gelu(x)
    	x = x.reshape([dim0,dim1,dim2])
    	
    	
    	
    	return x

class InteractorBlock(nn.Module):
    def __init__(self, d_model, num_tokens):
        super().__init__()
       
         
        self.memory = MemoryUnit(d_model)
        self.interaction = InteractionUnit(d_model,num_tokens)
        
    def forward(self, x):
                  
        residual = x
        
        x = self.interaction(x)
    
        x = x + residual
        
        residual = x
        
        x = self.memory(x)
        
                                          
        out = x + residual
        
        
        return out



class Interactor(nn.Module):
    def __init__(self, d_model,num_tokens, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[InteractorBlock(d_model,num_tokens) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)















