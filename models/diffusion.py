import torch
from torch.nn import Module
import torch.nn as nn

from .egnn import *


class Diffuser_ped_inter_geometric_cond_w_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.tau=config.tau1 / config.tau2
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        self.egnn = NetEGNN_hid2(hid_dim=config.egnn_hid_dim, n_layers=config.egnn_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        self.concat_ped1 = nn.Linear(config.spatial_emsize+config.context_dim+3, config.spatial_emsize//2)
        self.relu2 = nn.ReLU()
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO**
        hist_embedded = self.lstm_output(hist_embedded) # bs,N,embsize_out
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        acce_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        context_emb = torch.cat((acce_emb, hist_embedded), dim=-1)

        spatial_input_embedded = self.concat_ped1(torch.cat((context_emb, spatial_input_embedded, time_emb), dim=-1))
        output_ped = self.decode_ped1(self.relu2(spatial_input_embedded))

        return output_ped + pred_acc_dest
    
    
class Diffuser_ped_inter_geometric_cond_w_obs_w_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.tau=config.tau1 / config.tau2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        
        # config.context_dim = 2
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        self.egnn = NetEGNN_hid2(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        self.has_obstacles = config.has_obstacles
        if config.has_obstacles == True:
            self.egnn_obs = NetEGNN_hid_obs2(hid_dim = config.egnn_hid_dim_obs, n_layers = config.egnn_layers_obs)
            config.context_dim = config.egnn_hid_dim + config.egnn_hid_dim_obs + config.history_lstm_out
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)

        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = nn.Linear(config.spatial_emsize+config.context_dim+3, config.spatial_emsize//2)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        self.relu2 = nn.ReLU()
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO**
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        if self.has_obstacles:
          obs_features = context[5]
          near_obstacle_idx = context[6]
          neigh_obs_mask = context[7]
          obs_emb = self.egnn_obs([ped_features, obs_features, neigh_obs_mask, near_obstacle_idx], time_emb)
          ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb), dim=-1)
        else:
          ctx_emb = torch.cat((hist_embedded, ped_emb), dim=-1)


        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))

        spatial_input_embedded = self.concat_ped1(torch.cat((ctx_emb, spatial_input_embedded, time_emb), dim=-1))
        output_ped = self.decode_ped1(self.relu2(spatial_input_embedded))
        
        
        return output_ped + pred_acc_dest


class ped_inter_geometric_cond_w_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tau = config.tau1 / config.tau2
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        self.egnn = NetEGNN_hid2(hid_dim=config.egnn_hid_dim, n_layers=config.egnn_layers)

        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)

        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)

        self.concat_ped1 = nn.Linear(config.spatial_emsize + config.context_dim + 3, config.spatial_emsize // 2)
        self.relu2 = nn.ReLU()
        self.decode_ped1 = nn.Linear(config.spatial_emsize // 2, 2)

    def forward(self, x, beta, context: tuple, nei_list, t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature)))  # bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0, end_dim=1)  # bs*N,his_len,embsize
        _, (hist_embedded, _) = self.history_LSTM(hist_embedded)  # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2], -1)  # bs,N,embsize # TODO**
        hist_embedded = self.lstm_output(hist_embedded)  # bs,N,embsize_out

        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_  # des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1, x.shape[-2], 1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)

        acce_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)

        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))

        context_emb = torch.cat((acce_emb, hist_embedded), dim=-1)

        spatial_input_embedded = self.concat_ped1(torch.cat((context_emb, spatial_input_embedded, time_emb), dim=-1))
        output_ped = self.decode_ped1(self.relu2(spatial_input_embedded))

        return output_ped + pred_acc_dest, None


class ped_inter_geometric_cond_w_obs_w_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tau = config.tau1 / config.tau2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out

        # config.context_dim = 2
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        self.egnn = NetEGNN_hid2(hid_dim=config.egnn_hid_dim, n_layers=config.egnn_layers)
        self.has_obstacles = config.has_obstacles
        if config.has_obstacles == True:
            self.egnn_obs = NetEGNN_hid_obs2(hid_dim=config.egnn_hid_dim_obs, n_layers=config.egnn_layers_obs)
            config.context_dim = config.egnn_hid_dim + config.egnn_hid_dim_obs + config.history_lstm_out
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)

        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = nn.Linear(config.spatial_emsize + config.context_dim + 3, config.spatial_emsize // 2)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        self.relu2 = nn.ReLU()
        self.decode_ped1 = nn.Linear(config.spatial_emsize // 2, 2)

    def forward(self, x, beta, context: tuple, nei_list, t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature)))  # bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0, end_dim=1)  # bs*N,his_len,embsize
        _, (hist_embedded, _) = self.history_LSTM(hist_embedded)  # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2], -1)  # bs,N,embsize # TODO**
        hist_embedded = self.lstm_output(hist_embedded)

        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_  # des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1, x.shape[-2], 1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)

        ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)

        if self.has_obstacles:
            obs_features = context[5]
            near_obstacle_idx = context[6]
            neigh_obs_mask = context[7]
            obs_emb = self.egnn_obs([ped_features, obs_features, neigh_obs_mask, near_obstacle_idx], time_emb)
            ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb), dim=-1)
        else:
            ctx_emb = torch.cat((hist_embedded, ped_emb), dim=-1)

        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))

        spatial_input_embedded = self.concat_ped1(torch.cat((ctx_emb, spatial_input_embedded, time_emb), dim=-1))
        output_ped = self.decode_ped1(self.relu2(spatial_input_embedded))

        return output_ped + pred_acc_dest, None
