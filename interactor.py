import torch
from torch import nn


       
 

class MemoryUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
           
        self.norm_token = nn.LayerNorm(dim)
        self.proj_1 =  nn.Linear(dim,dim)
        self.proj_2 =  nn.Linear(dim,dim)
        self.proj_3 = nn.Linear(dim,dim)  
             	   
    def forward(self, x):
    
    	x = self.norm_token(x)    	
    	u, v = x, x 
    	u = self.proj_1(u)
    	u = self.norm_token(u)
    	v = self.proj_2(v)
    	g = u * v
    	x = self.proj_3(g)
    	x = self.norm_token(x)
    	
    	return x

class InteractionUnit(nn.Module):
    def __init__(self,dim,score_dim):
        super().__init__()
        
             
        self.norm_token = nn.LayerNorm(dim)       
        self.norm_score = nn.LayerNorm(score_dim)
             	   
    def forward(self, x):
    
    	x = self.norm_token(x)
    	q,k,v = x,x,x
    	score = torch.matmul(q, k.transpose(-1, -2))
    	interaction = self.norm_score(score)    
    	x = torch.matmul(interaction,v)
    	x = self.norm_token(x)
    	
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








