import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
# from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from utils.rev_in import RevIn

class GPT4TS_multi(nn.Module):
    
    def __init__(self, device):
        super(GPT4TS_multi, self).__init__()
        self.is_gpt = True
        self.patch_size = 16
        self.pretrain = 1
        self.stride = 8
        self.patch_num = (512 - self.patch_size) // self.stride + 1
        self.gpt_layers = 6
        self.d_model = 768

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if self.is_gpt:
            if self.pretrain:
                self.gpt2_trend = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                self.gpt2_season = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                self.gpt2_noise = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2_trend = GPT2Model(GPT2Config())
                self.gpt2_season = GPT2Model(GPT2Config())
                self.gpt2_noise = GPT2Model(GPT2Config())
            self.gpt2_trend.h = self.gpt2_trend.h[:self.gpt_layers]
            self.gpt2_season.h = self.gpt2_season.h[:self.gpt_layers]
            self.gpt2_noise.h = self.gpt2_noise.h[:self.gpt_layers]
            #print("gpt2 = {}".format(self.gpt2))
            self.prompt = 1
            # 
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_trend_token = self.tokenizer(text="Predice the future time step given the trend", return_tensors="pt").to(device)
            self.gpt2_season_token = self.tokenizer(text="Predice the future time step given the season", return_tensors="pt").to(device)
            self.gpt2_residual_token = self.tokenizer(text="Predice the future time step given the residual", return_tensors="pt").to(device)
        

        self.in_layer_trend = nn.Linear(self.patch_size, self.d_model)

        # self.out_layer_trend = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        self.in_layer_season = nn.Linear(configs.patch_size, configs.d_model)
        # self.out_layer_season = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        self.in_layer_noise = nn.Linear(configs.patch_size, configs.d_model)
        # self.out_layer_noise = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if self.prompt == 1:
            self.out_layer_trend = nn.Linear(configs.d_model * (self.patch_num+9), configs.pred_len)
            self.out_layer_season = nn.Linear(configs.d_model * (self.patch_num+9), configs.pred_len)
            self.out_layer_noise = nn.Linear(configs.d_model * (self.patch_num+9), configs.pred_len)
        else:
            self.out_layer_trend = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
            self.out_layer_season = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
            self.out_layer_noise = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)


        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2_trend.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for i, (name, param) in enumerate(self.gpt2_season.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for i, (name, param) in enumerate(self.gpt2_noise.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2_trend, self.gpt2_season, self.gpt2_noise, self.in_layer_trend, self.out_layer_trend, \
                      self.in_layer_season, self.out_layer_season, self.in_layer_noise, self.out_layer_noise):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

    def get_norm(self, x, d = 'norm'):
        # if d == 'norm':
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        return x, means, stdev
    
    def get_patch(self, x):
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # 4, 1, 420
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #4,1, 64, 16
        x = rearrange(x, 'b m n p -> (b m) n p') # 4, 64, 16

        return x
    
    def get_emb(self, x, tokens=None):
        if tokens is None:
            x = self.gpt2_trend(inputs_embeds =x).last_hidden_state
            return x
        else:
            [a,b,c] = x.shape
          
            prompt_x = self.gpt2_trend.wte(tokens)
            prompt_x = prompt_x.repeat(a,1,1)
            # print(prompt_x.shape)
            x_all = torch.cat((prompt_x, x), dim=1)
            #trend = self.gpt2_trend(inputs_embeds=trend).last_hidden_state # 4, 64, 768
            x = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state #[:,:,:] # 4, 64, 768
            return x


    def forward(self, x, itr, trend, season, noise):
        B, L, M = x.shape # 4, 512, 1

        # print(x.shape)
        # print(trend.shape)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        trend, means_trend, stdev_trend = self.get_norm(trend)
        season, means_season, stdev_season = self.get_norm(season)
        noise, means_noise, stdev_noise = self.get_norm(noise)



        # means_trend = trend.mean(1, keepdim=True).detach()
        # trend = trend - means_trend
        # stdev_trend = torch.sqrt(torch.var(trend, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        # trend /= stdev_trend

        # means_season = season.mean(1, keepdim=True).detach()
        # season = season - means_season
        # stdev_season = torch.sqrt(torch.var(season, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
        # season /= stdev_season

        # means_noise = noise.mean(1, keepdim=True).detach()
        # noise = noise - means_noise
        # stdev_noise = torch.sqrt(torch.var(noise, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
        # noise /= stdev_noise

        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # 4, 1, 420
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #4,1, 64, 16
        x = rearrange(x, 'b m n p -> (b m) n p') # 4, 64, 16


        trend = self.get_patch(trend)
        season = self.get_patch(season)
        noise = self.get_patch(noise)

        
        # if self.is_gpt and self.prompt == 1:
        #     M += 9

        trend = self.in_layer_trend(trend) # 4, 64, 768
        if self.is_gpt and self.prompt == 1:
            trend = self.get_emb(trend, self.gpt2_trend_token['input_ids'])
        else:
            trend = self.get_emb(trend)
            
            
        # print(trend.shape)
        # print(trend.reshape(B*M, -1).shape)
        trend = self.out_layer_trend(trend.reshape(B*M, -1)) # 4, 96
        # print(trend.shape)
        # trend = self.out_layer_trend(trend.reshape(B*M, -1)) # 4, 96
        trend = rearrange(trend, '(b m) l -> b l m', b=B) # 4, 96, 1
        trend = trend * stdev_trend + means_trend

        
        season = self.in_layer_season(season) # 4, 64, 768
        if self.is_gpt and self.prompt == 1:
            season = self.get_emb(season, self.gpt2_season_token['input_ids'])
        else:
            season = self.get_emb(season)
        # print(season.shape)
        # print(season.reshape(B*M, -1).shape)
        season = self.out_layer_season(season.reshape(B*M, -1)) # 4, 96
        # print(season.shape)
        season = rearrange(season, '(b m) l -> b l m', b=B) # 4, 96, 1
        season = season * stdev_season + means_season

        noise = self.in_layer_noise(noise)
        if self.is_gpt and self.prompt == 1:
            noise = self.get_emb(noise, self.gpt2_residual_token['input_ids'])
        else:
            noise = self.get_emb(noise)
        noise = self.out_layer_noise(noise.reshape(B*M, -1)) # 4, 96
        noise = rearrange(noise, '(b m) l -> b l m', b=B)
        noise = noise * stdev_noise + means_noise
        
        
        outputs = trend + season + noise

        return outputs