import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_gpt = True 
        self.pretrain = True
        self.freeze = True
        self.patch_size = configs.patch_len
        self.pretrain = True
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if self.is_gpt:
            if self.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('openai-community/gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.e_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(self.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        if self.freeze and self.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.train()
        
        self.cnt = 0
        self.split_token = torch.nn.Parameter(torch.zeros(1, 1, configs.d_model))
        torch.nn.init.uniform_(self.split_token)


    def forward(self, prompt, x):
        B, L, M = x.shape
        prompt_list = []
        if prompt is not None:
            for i in range(prompt.shape[0]):
                means = prompt[i].mean(1, keepdim=True).detach()
                prompt[i] = prompt[i] - means
                stdev = torch.sqrt(torch.var(prompt[i], dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
                prompt[i] /= stdev
            
            if prompt.shape[0]>0:
                prompt = prompt.permute(0,1,3,2)
                prompt = prompt.unfold(dimension=-1, size=self.patch_size, step=self.stride) # [topk x bs x nvars x patch_num x patch_len]
                prompt = self.in_layer(prompt)   # prompt: [topk x bs x nvars x patch_num x d_model] 
    
            if prompt.shape[0]>0:
                for i in range(prompt.shape[0]):
                    data = prompt[i]
                    data = torch.reshape(data,(data.shape[0]*data.shape[1],data.shape[2],data.shape[3])) # [bs*nvars x patch_num x d_model]
                    split_token = self.split_token.expand(data.shape[0],-1,-1)
                    prompt_list.append(data)
                    prompt_list.append(split_token)


        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        input_data = self.in_layer(x)
        prompt_list.append(input_data)
        outputs = torch.cat(prompt_list, dim=1)
        
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = outputs[:, -self.patch_num:, :]
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
