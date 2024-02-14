import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2Tokenizer
from peft import get_peft_model, LoraConfig

class ComplexLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        self.fc_real = nn.Linear(input_dim, output_dim)
        self.fc_imag = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_real = torch.real(x)
        x_imag = torch.imag(x)
        out_real = self.fc_real(x_real) - self.fc_imag(x_imag)
        out_imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        return torch.complex(out_real, out_imag)




def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class MultiFourier(torch.nn.Module):
    def __init__(self, N, P):
        super(MultiFourier, self).__init__()
        self.N = N
        self.P = P
        self.a = torch.nn.Parameter(torch.randn(max(N), len(N)), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(max(N), len(N)), requires_grad=True)
    
    def forward(self, t):
        output = torch.zeros_like(t)
        t = t.unsqueeze(-1).repeat(1, 1, max(self.N))  # shape: [batch_size, seq_len, max(N)]
        n = torch.arange(max(self.N)).unsqueeze(0).unsqueeze(0).to(t.device)  # shape: [1, 1, max(N)]
        for j in range(len(self.N)):  # loop over seasonal components
            # import ipdb; ipdb.set_trace() 
            cos_terms = torch.cos(2 * np.pi * (n[..., :self.N[j]]+1) * t[..., :self.N[j]] / self.P[j])  # shape: [batch_size, seq_len, N[j]]
            sin_terms = torch.sin(2 * np.pi * (n[..., :self.N[j]]+1) * t[..., :self.N[j]] / self.P[j])  # shape: [batch_size, seq_len, N[j]]
            output += torch.matmul(cos_terms, self.a[:self.N[j], j]) + torch.matmul(sin_terms, self.b[:self.N[j], j])
        return output

class GPT4TS_multi(nn.Module):
    
    def __init__(self, device, pred_len=336):
        super(GPT4TS_multi, self).__init__()
        self.is_gpt = 1
        self.patch_size = 16
        self.pretrain = 1
        self.stride = 8
        self.seq_len = 336
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.mul_season = MultiFourier([2], [24*4]) #, [ 24, 24*4])

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.gpt_layers = 6
        self.d_model = 768
        self.pred_len = pred_len
        self.freeze = 1
        
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
           
            self.prompt = 1
            # 
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_trend_token = self.tokenizer(text="Predict the future time step given the trend", return_tensors="pt").to(device)
            self.gpt2_season_token = self.tokenizer(text="Predict the future time step given the season", return_tensors="pt").to(device)
            self.gpt2_residual_token = self.tokenizer(text="Predict the future time step given the residual", return_tensors="pt").to(device)

            self.token_len = len(self.gpt2_trend_token['input_ids'][0])

            self.prompt_key_dict = nn.ParameterDict({})
            self.prompt_value_dict = nn.ParameterDict({})
            # self.summary_map = nn.Linear(self.token_len, 1)
            self.summary_map = nn.Linear(self.patch_num, 1)
            self.pool_size = 30
            self.top_k = 3
            self.prompt_len = 3
            for i in range(self.pool_size):
                prompt_shape = (self.prompt_len, 768)
                key_shape = (768)
                self.prompt_value_dict[f"prompt_value_{i}"] = nn.Parameter(torch.randn(prompt_shape))
                self.prompt_key_dict[f"prompt_key_{i}"] = nn.Parameter(torch.randn(key_shape))
        
        self.prompt_record = {f"id_{i}": 0 for i in range(self.pool_size)}
        self.diversify = True

        self.in_layer_trend = nn.Linear(self.patch_size, self.d_model)

        self.in_layer_season = nn.Linear(self.patch_size, self.d_model)

        self.in_layer_noise = nn.Linear(self.patch_size, self.d_model)

        if self.prompt == 1:
            need_token = False
            if need_token:
                    self.out_layer_trend = nn.Linear(self.d_model * (self.patch_num+self.token_len), self.pred_len)
                    self.out_layer_season = nn.Linear(self.d_model * (self.patch_num+self.token_len), self.pred_len)
                    self.out_layer_noise = nn.Linear(self.d_model * (self.patch_num+self.token_len), self.pred_len)
            else:
                self.out_layer_trend = nn.Linear(self.d_model * self.patch_num, self.pred_len)
                self.out_layer_season = nn.Linear(self.d_model * self.patch_num, self.pred_len)
                self.out_layer_noise = nn.Linear(self.d_model * self.patch_num, self.pred_len)
                self.fre_len = self.seq_len # // 2 + 1
                self.out_layer_noise_fre = ComplexLinear(self.fre_len, self.pred_len)
                self.pred_len = self.pred_len
                self.seq_len = self.seq_len


            self.prompt_layer_trend = nn.Linear(self.d_model, self.d_model)
            self.prompt_layer_season = nn.Linear(self.d_model, self.d_model)
            self.prompt_layer_noise = nn.Linear(self.d_model, self.d_model)

            for layer in (self.prompt_layer_trend, self.prompt_layer_season, self.prompt_layer_noise):
                layer.to(device=device)
                layer.train()


        else:
            self.out_layer_trend = nn.Linear(self.d_model * self.patch_num, self.pred_len)
            self.out_layer_season = nn.Linear(self.d_model * self.patch_num, self.pred_len)
            self.out_layer_noise = nn.Linear(self.d_model * self.patch_num, self.pred_len)


        
        if self.freeze and self.pretrain:
            for i, (name, param) in enumerate(self.gpt2_trend.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM, # causal language model
            r=16,
            lora_alpha=16,
            # target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",               # bias, set to only lora layers to train
            # modules_to_save=["classifier"],
        )
        
        # for i, (name, param) in enumerate(self.gpt2_season.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # for i, (name, param) in enumerate(self.gpt2_noise.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        
        self.gpt2_trend = get_peft_model(self.gpt2_trend, config)
       
        print_trainable_parameters(self.gpt2_trend)

        for layer in (self.gpt2_trend, self.in_layer_trend, self.out_layer_trend, \
                      self.in_layer_season, self.out_layer_season, self.in_layer_noise, self.out_layer_noise):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def select_prompt(self, summary, prompt_mask=None):
        prompt_key_matrix = torch.stack(tuple([self.prompt_key_dict[i] for i in self.prompt_key_dict.keys()]))
        # print("prompt_key_matrix: ", prompt_key_matrix.shape) # [15, 768]
        prompt_norm = self.l2_normalize(prompt_key_matrix, dim=1) # Pool_size, C
        # print("prompt_norm: ", prompt_norm.shape) # [10, 768]
        # print("summary: ", summary.shape) # [16, 15, 768]
        
        summary_reshaped = summary.view(-1, self.patch_num)
        # print('summay_reshaped:', summary_reshaped.shape)
        # print("summary_reshaped: ", summary_reshaped.shape) # [16 * 768, 15]
        summary_mapped = self.summary_map(summary_reshaped)
        
        summary = summary_mapped.view(-1, 768)
        # print("after map, summary: ", summary.shape) # [16, 768]
        summary_embed_norm = self.l2_normalize(summary, dim=1)
        # print("summary_embed_norm: ", summary_embed_norm.shape) # [16, 768]
        similarity = torch.matmul(summary_embed_norm, prompt_norm.t())
        # print("similarity: ", similarity.shape) # [16, 15]
        if not prompt_mask==None:
            idx = prompt_mask
        else:
            topk_sim, idx = torch.topk(similarity, k=self.top_k, dim=1)
        # Count
        # print(idx)
        if prompt_mask==None:
            count_of_keys = torch.bincount(torch.flatten(idx), minlength=15)
            # print(count_of_keys)
            for i in range(len(count_of_keys)):
                # print("i: ", i, count_of_keys[i].item())
                self.prompt_record[f"id_{i}"] += count_of_keys[i].item()
        # print("topk_sim: ", topk_sim.shape) # [16, 1]
        # print("idx: ", idx.shape) # [16, 1]

        prompt_value_matrix = torch.stack(tuple([self.prompt_value_dict[i] for i in self.prompt_value_dict.keys()]))
        # print("prompt_value_matrix: ", prompt_value_matrix.shape) # [15, 5 ,768]
        batched_prompt_raw = prompt_value_matrix[idx].squeeze(1)
        # print("batched_prompt_raw: ", batched_prompt_raw.shape) 
        batch_size, top_k, length, c = batched_prompt_raw.shape # [16, 3, 5, 768]
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) 
        # print("batched_prompt: ", batched_prompt.shape) # [16, 15, 768]

        batched_key_norm = prompt_norm[idx]
        summary_embed_norm = summary_embed_norm.unsqueeze(1)
        sim = batched_key_norm * summary_embed_norm
        reduce_sim = torch.sum(sim) / summary.shape[0]
        # print("reduce_sim: ", reduce_sim)

        return batched_prompt, reduce_sim


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
    
    def get_emb(self, x, tokens=None, type = 'Trend'):
        if tokens is None:
            if type == 'Trend':
                x = self.gpt2_trend(inputs_embeds =x).last_hidden_state
            elif type == 'Season':
                x = self.gpt2_trend(inputs_embeds =x).last_hidden_state
            elif type == 'Residual':
                x = self.gpt2_trend(inputs_embeds =x).last_hidden_state
            return x
        else:
            [a,b,c] = x.shape
          
            
            if type == 'Trend':
                prompt_x = self.gpt2_trend.wte(tokens)
                prompt_x = prompt_x.repeat(a,1,1)
                prompt_x = self.prompt_layer_trend(prompt_x)
                prompt_x, reduce_sim_trend = self.select_prompt(x, prompt_mask=None)

               
                x = torch.cat((prompt_x, x), dim=1)
                # x = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state

            elif type == 'Season':
                prompt_x = self.gpt2_trend.wte(tokens)
                prompt_x = prompt_x.repeat(a,1,1)
                prompt_x = self.prompt_layer_season(prompt_x)
                prompt_x, reduce_sim_trend = self.select_prompt(x, prompt_mask=None)

                
                x = torch.cat((prompt_x, x), dim=1)
                # x = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state
                
            elif type == 'Residual':
                prompt_x = self.gpt2_trend.wte(tokens)
                prompt_x = prompt_x.repeat(a,1,1)
                prompt_x = self.prompt_layer_noise(prompt_x)
                prompt_x, reduce_sim_trend = self.select_prompt(x, prompt_mask=None)
                
                x = torch.cat((prompt_x, x), dim=1)
                # x_residual = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state
            # # prompt_x = self.prompt_layer_trend(prompt_x)
            # # print(prompt_x.shape)
            # x_all = torch.cat((prompt_x, x), dim=1)
            # #trend = self.gpt2_trend(inputs_embeds=trend).last_hidden_state # 4, 64, 768
            # x = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state #[:,:,:] # 4, 64, 768
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

        # season = self.mul_season(season.squeeze()).unsqueeze(-1) +season

        


        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # 4, 1, 420
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #4,1, 64, 16
        x = rearrange(x, 'b m n p -> (b m) n p') # 4, 64, 16


        trend = self.get_patch(trend)
        season = self.get_patch(season)
        noise = self.get_patch(noise)


        # # TODO: add prompt
        # if eval:
        #     prompt_mask = None
        # else:
        #     most_frequent_event = torch.bincount(event_root_type).argmax().item()
        #     # print("most_frequent_event: ", most_frequent_event)
        #     single_prompt_mask = torch.tensor(event2maskId[most_frequent_event]).to(self.device)
        #     # print("single_prompt_mask: ", single_prompt_mask)
        #     prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
        # # print("prompt_mask: ", prompt_mask)
            
        # prompt_selected_trend, reduce_sim_trend = self.select_prompt(self.gpt2_trend_token['input_ids'], prompt_mask=None)
        # prompt_selected_season, reduce_sim_season = self.select_prompt(self.gpt2_season_token['input_ids'], prompt_mask=None)
        # prompt_selected_noise, reduce_sim_noise = self.select_prompt(self.gpt2_residual_token['input_ids'], prompt_mask=None)

    
        trend = self.in_layer_trend(trend) # 4, 64, 768
        if self.is_gpt and self.prompt == 1:
            trend = self.get_emb(trend, self.gpt2_trend_token['input_ids'], 'Trend')
        else:
            trend = self.get_emb(trend)

        season = self.in_layer_season(season) # 4, 64, 768
        if self.is_gpt and self.prompt == 1:
            season = self.get_emb(season, self.gpt2_season_token['input_ids'], 'Season')
        else:
            season = self.get_emb(season)

        noise = self.in_layer_noise(noise)
        if self.is_gpt and self.prompt == 1:
            noise = self.get_emb(noise, self.gpt2_residual_token['input_ids'], 'Residual')
        else:
            noise = self.get_emb(noise)

        x_all = torch.cat((trend, season, noise), dim=1)

        x = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state 
        # trend = self.gpt2_trend(inputs_embeds =trend).last_hidden_state 
        # season = self.gpt2_season(inputs_embeds =season).last_hidden_state 

        # import ipdb; ipdb.set_trace()
            
        trend  = x[:, :self.token_len+self.patch_num, :]  
        season  = x[:, self.token_len+self.patch_num:2*self.token_len+2*self.patch_num, :]  
        noise = x[:, 2*self.token_len+2*self.patch_num:, :]

        trend = trend[:, self.token_len:, :]
        season = season[:, self.token_len:, :]
        noise = noise[:, self.token_len:, :]    
            
        # print(trend.shape)
        # print(trend.reshape(B*M, -1).shape)
        trend = self.out_layer_trend(trend.reshape(B*M, -1)) # 4, 96
        # print(trend.shape)
        # trend = self.out_layer_trend(trend.reshape(B*M, -1)) # 4, 96
        trend = rearrange(trend, '(b m) l -> b l m', b=B) # 4, 96, 1
        trend = trend * stdev_trend + means_trend

        
       
        # print(season.shape)
        # print(season.reshape(B*M, -1).shape)
        season = self.out_layer_season(season.reshape(B*M, -1)) # 4, 96
        # print(season.shape)
        season = rearrange(season, '(b m) l -> b l m', b=B) # 4, 96, 1
        season = season * stdev_season + means_season

        # seq_last_noise = noise[:, -1:, :]#.detech()
        # noise = noise - seq_last_noise
        # fft_result = torch.fft.fft(noise.squeeze())
        # noise_new = self.out_layer_noise_fre(fft_result)
        # noise_new = torch.fft.ifft(noise_new).unsqueeze(-1)
        # noise = torch.real(noise_new)[:,-self.pred_len:,:] #+ noise
        # noise = noise + seq_last_noise
        noise = self.out_layer_noise(noise.reshape(B*M, -1)) # 4, 96
        noise = rearrange(noise, '(b m) l -> b l m', b=B)
        noise = noise * stdev_noise + means_noise
        
        
        outputs = trend + season + noise

        return outputs