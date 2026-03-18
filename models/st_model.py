import os

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from tqdm import tqdm

import models.diffusion as diffusion
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from data.data import Pedestrians, RawData

def clear_nan(tensor:torch.Tensor):
    tensor[tensor.isnan()]=0
    return tensor

class PhysicalVelocityNet(nn.Module):
    """预测物理守恒速度场"""

    def __init__(self, grid_rows, grid_cols, embed_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 输入密度场
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)  # 输出v_x, v_y场
        )

    def forward(self, rho):
        v_field = self.conv(rho.unsqueeze(1))  # [B, 2, H, W]
        return v_field.permute(0, 2, 3, 1)  # [B, H, W, 2]

class VelocityFusion(nn.Module):
    """动态融合权重网络"""

    def __init__(self, embed_dim=8):
        super().__init__()
        self.rho_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.mlp = nn.Linear(embed_dim, 1)

    def forward(self, rho):
        B, H, W = rho.shape
        alpha = torch.sigmoid(
            self.mlp(self.rho_proj(rho.view(B, H * W, 1)))
        ).view(B, H, W, 1)
        return alpha * 0.9 + 0.05  # 限制在[0.05,0.95]避免完全偏向

class DensityODE(nn.Module, Pedestrians):
    def __init__(self, model, t_len, grid_size, min_x, min_y, grid_rows, grid_cols, topk_ped,
                 sight_angle_ped, dist_threshold_ped, topk_obs, sight_angle_obs, dist_threshold_obs, esti_goal,
                 pred_collision, node_embed_dim, phys_embed_dim, fusion_embed_dim,fused_v ):
        super().__init__()
        self.t_len = t_len
        
        nodes = grid_rows * grid_cols
        self.nodes = nodes + 1

        self.model = model
        self.topk_ped = topk_ped
        self.sight_angle_ped = sight_angle_ped
        self.dist_threshold_ped = dist_threshold_ped
        self.topk_obs = topk_obs
        self.sight_angle_obs = sight_angle_obs
        self.dist_threshold_obs = dist_threshold_obs
        self.esti_goal = esti_goal

        self.embed_dim = node_embed_dim  
        self.node_w = nn.Parameter(torch.randn(self.nodes, self.embed_dim) * 0.01)
        self.node_b = nn.Parameter(torch.randn(self.nodes, self.embed_dim) * 0.01)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.min_x = min_x
        self.min_y = min_y
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_size = grid_size

        self.register_buffer('grid_centers', self._precompute_grid_centers())
        self.fused_v = fused_v
        if self.fused_v:
            self.phys_net = PhysicalVelocityNet(
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                embed_dim=phys_embed_dim
            )
            self.fusion_net = VelocityFusion(embed_dim=fusion_embed_dim)
        self.pred_collision = pred_collision

    def compute_fused_velocity(self, positions, rho, v_traj):
        """解耦速度计算核心方法"""
        B, N, _ = positions.shape

        # 2. 物理速度场 (宏观)
        v_phys_field = self.phys_net(rho)  # [B, grid_rows, grid_cols, 2]

        # 3. 物理场插值到个体位置
        nan_mask = positions.isnan().any(dim=-1, keepdim=True)  # [B, N, 1]
        positions_valid = positions.clone()
        positions_valid = torch.where(nan_mask.expand_as(positions_valid),
                                      torch.full_like(positions_valid, -10.0),
                                      positions_valid)  # 用 -10 替换 NaN

        grid_coords = (positions_valid - torch.tensor([self.min_x, self.min_y], device=positions.device))
        grid_coords = (grid_coords / self.grid_size).clamp(0, 1) * 2 - 1  # 归一化到[-1,1]

        v_phys = F.grid_sample(
            v_phys_field.permute(0, 3, 1, 2),  # [B, 2, H, W]
            grid_coords.unsqueeze(2),  # [B, N, 1, 2]
            align_corners=False,
            mode='bilinear'
        ).squeeze(3).permute(0, 2, 1)  # [B, N, 2]

        # 4. 动态融合 (基于局部密度)
        alpha = self.fusion_net(rho)  # [B, grid_rows, grid_cols, 1]
        alpha_sampled = F.grid_sample(
            alpha.permute(0, 3, 1, 2),
            grid_coords.unsqueeze(2),
            align_corners=False
        ).squeeze([2, 3])  # [B, N, 1]
        alpha_sampled = alpha_sampled.permute(0, 2, 1)

        
        v_phys = torch.where(nan_mask.expand_as(v_phys), torch.zeros_like(v_phys), v_phys)
        v_traj = torch.where(nan_mask.expand_as(v_traj), torch.zeros_like(v_traj), v_traj)
        alpha_sampled = torch.where(nan_mask, torch.zeros_like(alpha_sampled), alpha_sampled)

        return alpha_sampled * v_phys + (1 - alpha_sampled) * v_traj

    def _precompute_grid_centers(self):
        """预计算所有真实格子的中心坐标"""
        cols = torch.arange(self.grid_cols, dtype=torch.float32, device=self.node_w.device) + 0.5
        rows = torch.arange(self.grid_rows, dtype=torch.float32, device=self.node_w.device) + 0.5
        grid_x = cols * self.grid_size + self.min_x
        grid_y = rows * self.grid_size + self.min_y
        centers = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)  # [grid_rows, grid_cols, 2]
        return centers

    def set_data(self, data: RawData):
        self.position = []
        self.velocity = []
        self.acceleration = []

        if self.pred_collision:
            self.collision = []


        self.data = data
        self.ped_features = self.data.ped_features[..., 0, :, :, :]
        self.obs_features = self.data.obs_features[..., 0, :, :, :]
        self.self_features = self.data.self_features[..., 0, :, :]
        self.position.append(self.data.position[..., 0, :, :])
        self.velocity.append(self.data.velocity[..., 0, :, :])
        self.acceleration.append(self.data.acceleration[..., 0, :, :])
        self.dest_idx_cur = self.data.dest_idx[..., 0, :]
        self.dest_num = self.data.dest_num
        self.waypoints = self.data.waypoints
        # self.obstacles = self.data.obstacles
        self.new_peds_flag = (self.data.mask_p - self.data.mask_p_pred).long()
        self.mask_p_ = self.data.mask_p_pred.clone().long()
        self.mask_a_ = self.data.mask_a_pred.clone().long()
        self.desired_speed = self.data.self_features[..., 0, :, -1].unsqueeze(-1)
        self.history_features = self.data.self_hist_features[..., 0, :, :, :]
        self.history_features[self.history_features.isnan()] = 0
        self.near_ped_idx = self.data.near_ped_idx[..., 0, :, :]
        self.neigh_ped_mask = data.neigh_ped_mask[..., 0, :, :]
        self.near_obstacle_idx = data.near_obstacle_idx[..., 0, :, :]
        self.neigh_obs_mask = data.neigh_obs_mask[..., 0, :, :]
        self.curr = torch.cat((self.position[0], self.velocity[0], self.acceleration[0]), dim=-1)  # *c, n, 3
        self.dest_cur = self.data.destination[..., 0, :, :]  # *c, N, 2
        self.dest_idx_cur = self.data.dest_idx[..., 0, :]  # *c, N
        self.obstacles = self.data.obstacles.unsqueeze(0).repeat(self.near_obstacle_idx.shape[0], 1, 1)
        self.time_unit = data.time_unit
    def position_to_grid_probs(self, coords: torch.Tensor) -> torch.Tensor:
        """可微分的格子分配，返回每个坐标属于每个格子的概率"""

        B, ped = coords.shape[:2]
        device = coords.device

        # 分离坐标并检测NaN
        x, y = coords[..., 0], coords[..., 1]
        nan_mask = torch.isnan(x) | torch.isnan(y)
        valid_mask = ~nan_mask

        # 初始化概率张量 (包含虚拟格子)
        total_grids = self.grid_rows * self.grid_cols + 1
        probs = torch.zeros(B, ped, total_grids, device=device)

        # 处理有效坐标 (软分配)
        if valid_mask.any():
            valid_coords = coords[valid_mask]  # [N_valid, 2]

            # 计算到所有真实格子的距离 (向量化)
            dist = valid_coords.unsqueeze(1) - self.grid_centers.view(-1, 2)  # [N_valid, grid_rows*grid_cols, 2]
            dist_norm = torch.norm(dist, dim=-1)  # [N_valid, grid_rows*grid_cols]

            # 稳定的概率计算
            logits = -dist_norm.pow(2) * torch.clamp(self.temperature, min=1e-3, max=1e3)
            valid_probs = torch.softmax(logits, dim=-1)  # [N_valid, grid_rows*grid_cols]

            # 填充到结果张量 (不填充虚拟格子部分)
            batch_idx, ped_idx = torch.where(valid_mask)
            probs[batch_idx, ped_idx, :-1] = valid_probs[torch.arange(len(batch_idx))]

        # 处理NaN坐标 (硬分配)
        batch_idx, ped_idx = torch.where(nan_mask)
        probs[batch_idx, ped_idx, -1] = 1.0

        # 数值稳定性检查
        assert not torch.isnan(probs).any(), "Probability contains NaN!"
        return probs

    def grid_to_rho(self, grid_probs: torch.Tensor) -> torch.Tensor:
        """可微分的密度计算"""
        real_grid_probs = grid_probs[..., :-1]  # [B, ped, grid_rows*grid_cols]
        rho = real_grid_probs.sum(dim=1)  # [B, grid_rows*grid_cols]
        return rho.view(-1, self.grid_rows, self.grid_cols)

    def get_crossing_weights(self, pos1, pos2):
        """可微分的网格跨越权重计算"""
        # 1. 获取概率分布（移除虚拟格子）
        probs1 = self.position_to_grid_probs(pos1)[..., :-1]  # [B, N, grids]
        probs2 = self.position_to_grid_probs(pos2)[..., :-1]

        # 2. 概率平滑与归一化
        probs1 = (probs1 + 1e-10) / (probs1.sum(dim=-1, keepdim=True) + 1e-10 * probs1.shape[-1])
        probs2 = (probs2 + 1e-10) / (probs2.sum(dim=-1, keepdim=True) + 1e-10 * probs2.shape[-1])

        # 3. 计算JS散度（Jensen-Shannon Divergence）
        m = 0.5 * (probs1 + probs2)
        log_p1 = torch.log(probs1)
        log_p2 = torch.log(probs2)
        log_m = torch.log(m)

        # KL(P1 || M)
        kl_p1_m = (probs1 * (log_p1 - log_m)).sum(dim=-1)
        # KL(P2 || M)
        kl_p2_m = (probs2 * (log_p2 - log_m)).sum(dim=-1)

        js_div = 0.5 * (kl_p1_m + kl_p2_m)  # [B, N]

        # 4. 生成可微分权重
        crossing_weight = torch.sigmoid(self.temperature * (js_div - 0.1))
        return crossing_weight.clamp(0.01, 0.99)  # 限制范围保持梯度稳定性

    def forward(self, t, rho):
        B, row, col = rho.shape
        assert row == self.grid_rows and col == self.grid_cols

        t_index = (t).long()
        self.pred_next(t_index)

        # 使用可微分的位置到格子分配
        grid_probs_next = self.position_to_grid_probs(self.position[t_index + 1])
        pred_rho = self.grid_to_rho(grid_probs_next)

        # 1. 数据准备
        B, ped = self.velocity[t_index].shape[:2]
        device = self.velocity[t_index].device

        # 获取当前和下一时刻的格子概率分布
        senders_in_probs = self.position_to_grid_probs(self.position[t_index])  # [B, ped, nodes]
        receivers_in_probs = grid_probs_next  # [B, ped, nodes]

        # 计算位置last的格子分布
        position_last = self.position[t_index + 1] + self.velocity[t_index + 1] * self.time_unit
        senders_out_probs = receivers_in_probs
        receivers_out_probs = self.position_to_grid_probs(position_last)

        # 获取嵌入向量
        w_emb = self.node_w[:-1]  # [nodes, embed_dim]
        b_emb = self.node_b[:-1] # [nodes, embed_dim]

        # 计算加权平均的嵌入向量
        w_in_senders = torch.einsum('bpn,nd->bpd', senders_in_probs[..., :-1], w_emb)  # [B, ped, embed_dim]
        w_in_receivers = torch.einsum('bpn,nd->bpd', receivers_in_probs[..., :-1], w_emb)
        b_in_senders = torch.einsum('bpn,nd->bpd', senders_in_probs[..., :-1], b_emb)
        b_in_receivers = torch.einsum('bpn,nd->bpd', receivers_in_probs[..., :-1], b_emb)

        v_in = torch.norm(self.velocity[t_index], dim=-1)  # [B, ped]
        crossing_weight_in = self.get_crossing_weights(self.position[t_index], self.position[t_index])


        # 计算流入量
        w_in = torch.einsum('bpd,bpd->bp', w_in_senders, w_in_receivers)  # [B, ped]
        b_in = torch.einsum('bpd,bpd->bp', b_in_senders, b_in_receivers)  # [B, ped]
        inflow = (w_in * v_in * crossing_weight_in + b_in)  # [B, ped]

        # 计算rho的加权聚合
        rho_ext = torch.cat([rho.view(B, -1), torch.zeros((B, 1), device=rho.device)], dim=-1)  # [B, nodes]
        rho_senders = torch.einsum('bpn,bn->bp', senders_in_probs, rho_ext)  # [B, ped]

        weighted_inflow = inflow * rho_senders  # [B, ped]

        # 使用概率分布进行聚合
        inflows = torch.einsum('bp,bpn->bn', weighted_inflow, receivers_in_probs[..., :-1])  # [B, nodes]

        # 计算流出量
        w_senders_out = torch.einsum('bpn,nd->bpd', senders_out_probs[..., :-1], w_emb)
        w_receivers_out = torch.einsum('bpn,nd->bpd', receivers_out_probs[..., :-1], w_emb)
        b_senders_out = torch.einsum('bpn,nd->bpd', senders_out_probs[..., :-1], b_emb)
        b_receivers_out = torch.einsum('bpn,nd->bpd', receivers_out_probs[..., :-1], b_emb)

        v_out = torch.norm(self.velocity[t_index + 1], dim=-1)  # [B, ped]
        crossing_weight_out = self.get_crossing_weights(self.position[t_index + 1], position_last)


        w_out = torch.einsum('bpd,bpd->bp', w_senders_out, w_receivers_out)
        b_out = torch.einsum('bpd,bpd->bp', b_senders_out, b_receivers_out)
        outflow = (w_out * v_out * crossing_weight_out + b_out)  # [B, ped]

        # 使用概率分布进行聚合
        outflows = torch.einsum('bp,bpn->bn', outflow, senders_out_probs[..., :-1])  # [B, nodes]
        pred_rho = pred_rho.view(B, -1)  # [B, nodes]
        outflows = outflows * pred_rho
        # 计算净流量
        flows = inflows - outflows # 去掉虚拟格子

        return flows.view_as(rho)

    def pred_next(self, t):
        beta = 0.  # 归一化时间
        beta = torch.full((self.acceleration[t].shape[0],), beta, device=self.position[0].device)
        a_next, pred_collision = self.model(x=self.acceleration[t] , #c, n, 2
                            beta=beta,
                            context=(self.curr,
                                    self.neigh_ped_mask.detach(),
                                    self.self_features.detach(),
                                    self.near_ped_idx.detach(),
                                    self.history_features.detach(),
                                    self.obstacles.detach(),
                                    self.near_obstacle_idx.detach(),
                                    self.neigh_obs_mask.detach()),
                            nei_list=None,
                            t=None)  #chec**
        if self.pred_collision:
            self.collision.append(pred_collision)
        if self.fused_v:
            current_rho = self.grid_to_rho(
                self.position_to_grid_probs(self.position[t])
            )
            fused_v = self.compute_fused_velocity(self.position[t], current_rho, self.velocity[t])
        else:
            fused_v = self.velocity[t]
        self.acceleration.append(a_next)
        self.velocity.append(fused_v + self.acceleration[t] * self.time_unit)  # *c, n, 2
        self.position.append(self.position[t] + self.velocity[t] * self.time_unit)

        # update destination & mask_p
        # out_of_bound = torch.tensor(float('nan'), device=self.position[t].device)
        dis_to_dest = torch.norm(self.position[t].detach() - self.dest_cur, p=2, dim=-1)

        # 更新 dest_idx_cur（不改原始变量 inplace）
        dest_idx_cur = self.dest_idx_cur.clone()
        mask_arrived = dis_to_dest < 0.5
        dest_idx_cur[mask_arrived] += 1

        # clip 越界
        mask_oob = dest_idx_cur > self.dest_num - 1
        pos_next = self.position[t + 1].clone()
        # pos_next[mask_oob] = out_of_bound
        self.position[t + 1] = pos_next

        dest_idx_cur[mask_oob] -= 1
        self.dest_idx_cur = dest_idx_cur  # 覆盖回去

        # 获取当前目的地
        dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
        dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
        self.dest_cur = torch.gather(self.waypoints, -3, dest_idx_cur_).squeeze(1)  # *c, n, 2

        # update everyone's state
        p_cur = self.position[t + 1]
        v_cur = self.velocity[t + 1]
        a_cur = self.acceleration[t + 1]

        # update hist_v
        # hist_v = self.state_features[2][..., :, 2:-3]  # *c, n, 2*x
        # hist_v_new = hist_v.clone()
        # hist_v_new[..., :, :-2] = hist_v_new[..., :, 2:]
        # hist_v_new[..., :, -2:] = self.velocity[t + 1]

        # 处理新加入的行人
        if t < self.t_len - 1:
            new_idx = self.new_peds_flag[..., t + 1, :]  # c, n
            if torch.sum(new_idx) > 0:
                pos_next = self.position[t + 1].clone()
                vel_next = self.velocity[t + 1].clone()
                acc_next = self.acceleration[t + 1].clone()
                dest_next = self.dest_cur.clone()
                # hist_v_next = hist_v_new.clone()
                dest_idx_next = dest_idx_cur.clone()

                pos_next[new_idx == 1] = self.data.position[..., t + 1, :, :][new_idx == 1]
                vel_next[new_idx == 1] = self.data.velocity[..., t + 1, :, :][new_idx == 1]
                acc_next[new_idx == 1] = self.data.acceleration[..., t + 1, :, :][new_idx == 1]
                dest_next[new_idx == 1] = self.data.destination[..., t + 1, :, :][new_idx == 1]
                dest_idx_next[new_idx == 1] = self.data.dest_idx[..., t + 1, :][new_idx == 1]
                # hist_v_next[new_idx == 1] = self.data.self_features[..., t + 1, :, 2:-3][new_idx == 1]

                self.position[t + 1] = pos_next
                self.velocity[t + 1] = vel_next
                self.acceleration[t + 1] = acc_next
                self.dest_cur = dest_next
                self.dest_idx_cur = dest_idx_next
                # hist_v_new = hist_v_next
        self.curr = torch.cat((self.position[t + 1], self.velocity[t + 1], self.acceleration[t + 1]), dim=-1).detach()

        if self.esti_goal == 'acce':
            new_traj = self.curr.clone().unsqueeze(-2)
            self.history_features = torch.cat([self.history_features[..., 1:, :], new_traj], dim=-2)
            # self.history_features[self.history_features.isnan()] = 0
            history_features = self.history_features.clone()
            history_features[history_features.isnan()] = 0
            self.history_features = history_features
        elif self.esti_goal=='pos':
            if self.config.history_dim == 2:
                self.history_features[..., :-1, :] = self.history_features[:, 1:, :].clone()  # history_features: n, len, 6
                self.history_features[..., -1, :2] = p_cur.clone()

        # calculate features
        ped_features, obs_features, dest_features, near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask = self.get_relative_features(
            p_cur.unsqueeze(-3).detach(), v_cur.unsqueeze(-3).detach(), a_cur.unsqueeze(-3).detach(),
            self.dest_cur.unsqueeze(-3).detach(), self.obstacles, self.topk_ped, self.sight_angle_ped,
            self.dist_threshold_ped, self.topk_obs, self.sight_angle_obs, self.dist_threshold_obs)

        self.ped_features = ped_features.squeeze(1)
        self.obs_features = obs_features.squeeze(1)
        dest_features = dest_features.squeeze(1)
        self.near_ped_idx = near_ped_idx.squeeze(1)
        self.neigh_ped_mask = neigh_ped_mask.squeeze(1)
        self.near_obstacle_idx = near_obstacle_idx.squeeze(1)
        self.neigh_obs_mask = neigh_obs_mask.squeeze(1)

        self.self_features = torch.cat((dest_features, v_cur, a_cur, self.desired_speed), dim=-1)

class DiffeqSolver(nn.Module):
    def __init__(self, odefunc, method, odeint_rtol=1e-4, odeint_atol=1e-5):
        nn.Module.__init__(self)

        self.ode_method = method
        self.odefunc = odefunc

        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        ode_pred_rho = odeint(self.odefunc,
                              first_point,
                              time_steps_to_predict,
                              rtol=self.rtol,
                              atol=self.atol,
                              method=self.ode_method)

        nn_pred_a = self.odefunc.acceleration
        nn_pred_p = self.odefunc.position
        pred_collision = self.odefunc.collision if self.odefunc.pred_collision else None

        nn_pred_a = torch.stack(nn_pred_a, dim=1)  # 将 nn_pred_v 转为 tensor
        nn_pred_p = torch.stack(nn_pred_p, dim=1)  # 将 position 转为 tensor
        nn_pred_collision = torch.stack(pred_collision, dim=1) if pred_collision is not None else None
        ode_pred_rho = ode_pred_rho.permute(1, 0, 2, 3)  # 调整维度顺序为 (batch_size, time_steps, rows, cols)


        mask_a = self.odefunc.mask_a_.unsqueeze(-1)  # 确保维度匹配
        nn_pred_a = torch.where(mask_a != 0, nn_pred_a, torch.zeros_like(nn_pred_a))

        # 处理 nn_pred_p（替换 NaN 为 0）
        mask_p = self.odefunc.mask_p_.unsqueeze(-1)  # 确保维度匹配
        nn_pred_p = torch.where(mask_p != 0, nn_pred_p, torch.zeros_like(nn_pred_p))
        return nn_pred_a[:, 1:, ...], ode_pred_rho[:, 1:, ...], nn_pred_p[:, 1:, ...], nn_pred_collision

class CoupledModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        ST_net = getattr(diffusion, config.diffnet)
        self.pre_model = ST_net(config)
        # self.load_model(config.model_dir)
        self.horizon = config.valid_steps - 1
        self.config = config

        # 时间参数
        self.ode_method = config.ode_method
        self.atol = config.odeint_atol
        self.rtol = config.odeint_rtol
        self.grid_size = config.grid_size
        self.min_x = config.min_x
        self.min_y = config.min_y
        self.grid_rows = config.grid_rows
        self.grid_cols = config.grid_cols
        self.pred_collision = config.pred_collision


        self.tau = config.tau1 / config.tau2  

        self.odefunc = DensityODE(self.pre_model, config.valid_steps, self.grid_size,
                                  self.min_x, self.min_y, self.grid_rows, self.grid_cols, self.config.topk_ped, self.config.sight_angle_ped,
                             self.config.dist_threshold_ped, self.config.topk_obs,
                             self.config.sight_angle_obs, self.config.dist_threshold_obs, self.config.esti_goal, self.pred_collision,
                            self.config.node_embed_dim, self.config.phys_embed_dim, self.config.fusion_embed_dim, self.config.fused_v)

        self.diffeq_solver = DiffeqSolver(self.odefunc,
                                          self.ode_method,
                                          odeint_rtol=self.rtol,
                                          odeint_atol=self.atol)


    def forward(self, batch, lam_1=1.0, lam_2=1.0):
        time_steps_to_predict = torch.arange(start=0, end=self.horizon + 1, step=1).float().to(batch.labels.device)
        # time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
        self.odefunc.set_data(batch)
        position_start = batch.labels[..., 0, :, :2]
        grid = self.odefunc.position_to_grid_probs(position_start)
        first_point = self.odefunc.grid_to_rho(grid)
        nn_pred_a, ode_pred_rho, position, pred_collision = self.diffeq_solver(first_point, time_steps_to_predict)

        batch.labels[self.odefunc.mask_a_ == 0] = 0.
        rho_lable = self.odefunc.position_to_grid_probs(batch.labels[..., 1:, :, :2].reshape(-1, *position_start.shape[1:]))
        rho_lable = self.odefunc.grid_to_rho(rho_lable)
        rho_lable = rho_lable.view(position_start.shape[0], self.horizon, self.grid_rows, self.grid_cols)
        collision_label = batch.labels[:, 1:, :, 6:] if self.pred_collision else None

        true_a = batch.labels[:, 1:, :, 4:6]
        true_p = batch.labels[:, 1:, :, :2]
        loss_nn_a = F.mse_loss(nn_pred_a, true_a)
        loss_ode_rho = self.physics_loss(ode_pred_rho, rho_lable)
        loss_nn_position = F.mse_loss(position, true_p)
    
        if self.pred_collision:
            collision_pred_loss = F.binary_cross_entropy(pred_collision, collision_label, reduction='sum')
            loss = lam_1 * (loss_nn_a + loss_nn_position + collision_pred_loss) + lam_2 * (loss_ode_rho)
        else:
            collision_pred_loss = torch.tensor(0.0, device=batch.labels.device)
            loss = lam_1 * (loss_nn_a + loss_nn_position) + lam_2 * (loss_ode_rho)


        return {
            'loss': loss,
            'loss_nn_a': loss_nn_a,
            'loss_nn_position': loss_nn_position,
            'loss_ode_rho': loss_ode_rho,
            'loss_nn_collision': collision_pred_loss,
        }

    def physics_loss(self, ode_pred, label):
        # 创建有效掩码 (排除虚拟格子)
        valid_mask = (label > 0).float()

        # 密度加权
        weights = 1 + label * 20  # 放大高密度区域权重

        # 只计算有效区域的损失
        loss = (weights * valid_mask * (ode_pred - label) ** 2).sum()
        loss /= (valid_mask.sum() + 1e-6)

        return loss
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        data = torch.load(str(f'{model_path}/model-best.pt'))
        self.pre_model.load_state_dict(data['model'])
        print(f"prediction Model loaded from {model_path} successfully.")

    def simulate(self, data, t_start=0):
        args = self.config
        # if load_model:
        #     self.load_model(args, set_model=False, finetune_flag=self.finetune_flag)
        ped = Pedestrians()
        destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n

        desired_speed = data.self_features[..., t_start, :, -1].unsqueeze(-1)  # *c, n, 1

        if self.config.esti_goal == 'acce':
            history_features = data.self_hist_features[..., t_start, :, :, :]
            history_features = clear_nan(history_features)
        elif self.config.esti_goal == 'pos':
            raise NotImplementedError
            # hist_pos = data.self_hist_features[...,t_start, :, :, :2]
            # hist_vel = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_acce = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_vel[:,1:,:] = data.self_hist_features[...,t_start, :, :-1, 2:4]
            # hist_acce[:,2:,:] = data.self_hist_features[...,t_start, :, :-2, 4:6]
            # history_features = torch.cat((hist_pos, hist_vel, hist_acce), dim=-1)
            # history_features = clear_nan(history_features)
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        self_feature = data.self_features[..., t_start, :, :]

        near_ped_idx = data.near_ped_idx[..., t_start, :, :]
        neigh_ped_mask = data.neigh_ped_mask[..., t_start, :, :]
        near_obstacle_idx = data.near_obstacle_idx[..., t_start, :, :]
        neigh_obs_mask = data.neigh_obs_mask[..., t_start, :, :]

        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)  # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        p_res = torch.zeros(data.position.shape, device=args.device)  # *c, t, n, 2
        v_res = torch.zeros(data.velocity.shape, device=args.device)  # *c, t, n, 2
        a_res = torch.zeros(data.acceleration.shape, device=args.device)  # *c, t, n, 2
        dest_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        ped_force_res = torch.zeros(data.acceleration.shape, device=args.device)

        p_res[..., :t_start + 1, :, :] = data.position[..., :t_start + 1, :, :]
        v_res[..., :t_start + 1, :, :] = data.velocity[..., :t_start + 1, :, :]
        a_res[..., :t_start + 1, :, :] = data.acceleration[..., :t_start + 1, :, :]

        mask_p_new = torch.zeros(mask_p_.shape, device=mask_p_.device)
        mask_p_new[..., :t_start + 1, :] = data.mask_p[..., :t_start + 1, :].long()

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        for t in tqdm(range(t_start, data.num_frames)):
            p_res[..., t, :, :] = p_cur
            v_res[..., t, :, :] = v_cur
            a_res[..., t, :, :] = a_cur
            # mask_p_new[..., t, ~p_cur[:, 0].isnan()] = 1
            mask_p_new[..., t, :][~p_cur[..., 0].isnan()] = 1

            # a_next = self.diffusion.sample(*state_features)[0]
            if self.config.esti_goal == 'acce':
                beta = 0.  # 归一化时间
                beta = torch.full((p_cur.shape[0],), beta, device=p_cur.device)
                a_next, _ = self.pre_model(x=a_cur,  # c, n, 2
                                    beta=beta,
                                    context=(curr.detach(),
                                             neigh_ped_mask.detach(),
                                             self_feature.detach(),
                                             near_ped_idx.detach(),
                                             history_features.detach(),
                                             obstacles.detach(),
                                             near_obstacle_idx.detach(),
                                             neigh_obs_mask.detach()),
                                    nei_list=None,
                                    t=None)  # chec**

                # print(a_next.sum())
                # if a_next[mask_p_[t]].max()>15 or a_next[mask_p_[t]].isnan().any():
                #     pdb.set_trace()
                # dest force part
                self_feature = self_feature
                desired_speed = self_feature[..., -1].unsqueeze(-1)
                temp = torch.norm(self_feature[..., :2], p=2, dim=-1, keepdim=True)
                temp_ = temp.clone()
                temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
                dest_direction = self_feature[..., :2] / temp_  # des,direction
                pred_acc_dest = (desired_speed * dest_direction - self_feature[..., 2:4]) / self.tau
                pred_acc_ped = a_next - pred_acc_dest
                if t < data.num_frames - 1:
                    dest_force_res[..., t + 1, :, :] = pred_acc_dest
                    ped_force_res[..., t + 1, :, :] = pred_acc_ped

                v_next = v_cur + a_cur * data.time_unit
                p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            elif self.config.esti_goal == 'pos':
                raise NotImplementedError
                # if self.config.history_dim==6:
                #     p_next = self.diffusion.sample(history_features.unsqueeze(0), dest_features.unsqueeze(0))
                # elif self.config.history_dim==2:
                #     p_next = self.diffusion.sample(history_features[...,:2].unsqueeze(0), dest_features.unsqueeze(0))
                #     v_next = v_cur
                #     a_next = a_cur

            # update destination & mask_p
            out_of_bound = torch.tensor(float('nan'), device=args.device)
            dis_to_dest = torch.norm(p_cur - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] += 1  # *c, n
            # TODO: currently don't delete?
            p_next[dest_idx_cur > dest_num - 1, :] = out_of_bound  # destination arrived

            dest_idx_cur[dest_idx_cur > dest_num - 1] -= 1
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze(1)  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            v_cur = v_next
            a_cur = a_next
            # curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)
            # update hist_v
            hist_v = self_feature[..., :, 2:-3]  # *c, n, 2*x
            hist_v[..., :, :-2] = hist_v[..., :, 2:]
            hist_v[..., :, -2:] = v_cur

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]

                    # update hist_v
                    hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]

            curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)  # *c, N, 6

            # update hist_features
            if self.config.esti_goal == 'acce':
                history_features[..., :-1, :] = history_features[..., 1:, :].clone()  # history_features: n, len, 6
                history_features[..., -1, :2] = p_cur.clone()
                history_features[..., -1, 2:4] = v_cur.clone()
                history_features[..., -1, 4:6] = a_cur.clone()
                history_features = clear_nan(history_features)

            elif self.config.esti_goal == 'pos':
                if self.config.history_dim == 2:
                    history_features[..., :-1, :] = history_features[:, 1:, :].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur.clone()

            # calculate features
            if self.config.esti_goal == 'acce':
                ped_features, obs_features, dest_features, \
                    near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask = ped.get_relative_features(
                    p_cur.unsqueeze(-3), v_cur.unsqueeze(-3), a_cur.unsqueeze(-3),
                    dest_cur.unsqueeze(-3), obstacles, args.topk_ped, args.sight_angle_ped,
                    args.dist_threshold_ped, args.topk_obs,
                    args.sight_angle_obs, args.dist_threshold_obs)
                ped_features = ped_features.squeeze(1)
                obs_features = obs_features.squeeze(1)
                dest_features = dest_features.squeeze(1)
                near_ped_idx = near_ped_idx.squeeze(1)
                neigh_ped_mask = neigh_ped_mask.squeeze(1)
                near_obstacle_idx = near_obstacle_idx.squeeze(1)
                neigh_obs_mask = neigh_obs_mask.squeeze(1)

                self_feature = torch.cat((dest_features, hist_v, a_cur, desired_speed), dim=-1)
            elif self.config.esti_goal == 'pos':
                raise NotImplementedError
                dest_features = dest_cur - p_cur
                dest_features[dest_features.isnan()] = 0.

        output = RawData(p_res.squeeze(0), v_res.squeeze(0), a_res.squeeze(0), destination.squeeze(0), destination.squeeze(0), obstacles.squeeze(0),
                              mask_p_new.squeeze(0), meta_data=data.meta_data)
        return output, dest_force_res, ped_force_res

    def init_optimizers(self, main_optimizer):
        """初始化解耦训练的优化器"""
        # 轨迹网络优化器 (继承主优化器配置)
        self.opt_traj = type(main_optimizer)(
            [
                {'params': self.pre_model.parameters()},  # 原有加速度网络

            ],
            **{k: v for k, v in main_optimizer.defaults.items()}
        )

        # 物理网络优化器 (更小的学习率)
        if self.config.fused_v:
            self.opt_physics = type(main_optimizer)(
                [
                    {'params': self.odefunc.phys_net.parameters()},
                    {'params': self.odefunc.fusion_net.parameters()},
                    {'params': [self.odefunc.temperature]},  # 温度参数
                    {'params': self.odefunc.node_w},  # 动态图参数
                    {'params': self.odefunc.node_b}
                ],
                lr=main_optimizer.defaults['lr'] * 0.5,
                **{k: v for k, v in main_optimizer.defaults.items() if k != 'lr'}
            )
        else:
            self.opt_physics = type(main_optimizer)(
                [
                    {'params': [self.odefunc.temperature]},  # 温度参数
                    {'params': self.odefunc.node_w},  # 动态图参数
                    {'params': self.odefunc.node_b}
                ],
                lr=main_optimizer.defaults['lr'] * 0.5,
                **{k: v for k, v in main_optimizer.defaults.items() if k != 'lr'}
            )

class DensityODE_post(nn.Module, Pedestrians):
    def __init__(self, model, t_len, grid_size, min_x, min_y, grid_rows, grid_cols, topk_ped,
                 sight_angle_ped, dist_threshold_ped, topk_obs, sight_angle_obs, dist_threshold_obs, esti_goal):
        super().__init__()
        self.t_len = t_len

        nodes = grid_rows * grid_cols
        self.nodes = nodes + 1

        self.model = model
        self.topk_ped = topk_ped
        self.sight_angle_ped = sight_angle_ped
        self.dist_threshold_ped = dist_threshold_ped
        self.topk_obs = topk_obs
        self.sight_angle_obs = sight_angle_obs
        self.dist_threshold_obs = dist_threshold_obs
        self.esti_goal = esti_goal

        self.embed_dim = 10  # 可以调整这个维度
        self.node_w = nn.Parameter(torch.randn(self.nodes, self.embed_dim) * 0.01)
        self.node_b = nn.Parameter(torch.randn(self.nodes, self.embed_dim) * 0.01)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.min_x = min_x
        self.min_y = min_y
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_size = grid_size

        self.register_buffer('grid_centers', self._precompute_grid_centers())

        self.phys_net = PhysicalVelocityNet(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            embed_dim=16
        )
        self.fusion_net = VelocityFusion(embed_dim=8)

    def compute_fused_velocity(self, positions, rho, v_traj):
        """解耦速度计算核心方法"""
        B, N, _ = positions.shape

        # 2. 物理速度场 (宏观)
        v_phys_field = self.phys_net(rho)  # [B, grid_rows, grid_cols, 2]

        # 3. 物理场插值到个体位置
        nan_mask = positions.isnan().any(dim=-1, keepdim=True)  # [B, N, 1]
        positions_valid = positions.clone()
        positions_valid = torch.where(nan_mask.expand_as(positions_valid),
                                      torch.full_like(positions_valid, -10.0),
                                      positions_valid)  # 用 -10 替换 NaN

        grid_coords = (positions_valid - torch.tensor([self.min_x, self.min_y], device=positions.device))
        grid_coords = (grid_coords / self.grid_size).clamp(0, 1) * 2 - 1  # 归一化到[-1,1]

        v_phys = F.grid_sample(
            v_phys_field.permute(0, 3, 1, 2),  # [B, 2, H, W]
            grid_coords.unsqueeze(2),  # [B, N, 1, 2]
            align_corners=False,
            mode='bilinear'
        ).squeeze(3).permute(0, 2, 1)  # [B, N, 2]

        # 4. 动态融合 (基于局部密度)
        alpha = self.fusion_net(rho)  # [B, grid_rows, grid_cols, 1]
        alpha_sampled = F.grid_sample(
            alpha.permute(0, 3, 1, 2),
            grid_coords.unsqueeze(2),
            align_corners=False
        ).squeeze([2, 3])  # [B, N, 1]
        alpha_sampled = alpha_sampled.permute(0, 2, 1)

        # 用 torch.where 替代直接赋值，保持梯度
        v_phys = torch.where(nan_mask.expand_as(v_phys), torch.zeros_like(v_phys), v_phys)
        v_traj = torch.where(nan_mask.expand_as(v_traj), torch.zeros_like(v_traj), v_traj)
        alpha_sampled = torch.where(nan_mask, torch.zeros_like(alpha_sampled), alpha_sampled)

        return alpha_sampled * v_phys + (1 - alpha_sampled) * v_traj

    def _precompute_grid_centers(self):
        """预计算所有真实格子的中心坐标"""
        cols = torch.arange(self.grid_cols, dtype=torch.float32, device=self.node_w.device) + 0.5
        rows = torch.arange(self.grid_rows, dtype=torch.float32, device=self.node_w.device) + 0.5
        grid_x = cols * self.grid_size + self.min_x
        grid_y = rows * self.grid_size + self.min_y
        centers = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)  # [grid_rows, grid_cols, 2]
        return centers

    def set_data(self, data: RawData):

        self.rho_pred = []
        self.data = data

        self.position = self.data.position
        self.velocity = self.data.velocity
        self.acceleration = self.data.acceleration
        self.rho_pred = torch.zeros_like(self.position)


        self.time_unit = data.time_unit
    def position_to_grid_probs(self, coords: torch.Tensor) -> torch.Tensor:
        """可微分的格子分配，返回每个坐标属于每个格子的概率"""

        B, ped = coords.shape[:2]
        device = coords.device

        # 分离坐标并检测NaN
        x, y = coords[..., 0], coords[..., 1]
        nan_mask = torch.isnan(x) | torch.isnan(y)
        valid_mask = ~nan_mask

        # 初始化概率张量 (包含虚拟格子)
        total_grids = self.grid_rows * self.grid_cols + 1
        probs = torch.zeros(B, ped, total_grids, device=device)

        # 处理有效坐标 (软分配)
        if valid_mask.any():
            valid_coords = coords[valid_mask]  # [N_valid, 2]

            # 计算到所有真实格子的距离 (向量化)
            dist = valid_coords.unsqueeze(1) - self.grid_centers.view(-1, 2)  # [N_valid, grid_rows*grid_cols, 2]
            dist_norm = torch.norm(dist, dim=-1)  # [N_valid, grid_rows*grid_cols]

            # 稳定的概率计算
            logits = -dist_norm.pow(2) * torch.clamp(self.temperature, min=1e-3, max=1e3)
            valid_probs = torch.softmax(logits, dim=-1)  # [N_valid, grid_rows*grid_cols]

            # 填充到结果张量 (不填充虚拟格子部分)
            batch_idx, ped_idx = torch.where(valid_mask)
            probs[batch_idx, ped_idx, :-1] = valid_probs[torch.arange(len(batch_idx))]

        # 处理NaN坐标 (硬分配)
        batch_idx, ped_idx = torch.where(nan_mask)
        probs[batch_idx, ped_idx, -1] = 1.0

        # 数值稳定性检查
        assert not torch.isnan(probs).any(), "Probability contains NaN!"
        return probs

    def grid_to_rho(self, grid_probs: torch.Tensor) -> torch.Tensor:
        """可微分的密度计算"""
        # grid_probs: [B, ped, grid_rows*grid_cols+1]
        # 对每个batch，将行人概率分布聚合到格子上
        # rho = grid_probs.sum(dim=1)  # [B, grid_rows*grid_cols+1]
        # rho = rho[..., :-1]  # 去掉虚拟格子
        # return rho.view(-1, self.grid_rows, self.grid_cols)
        real_grid_probs = grid_probs[..., :-1]  # [B, ped, grid_rows*grid_cols]
        rho = real_grid_probs.sum(dim=1)  # [B, grid_rows*grid_cols]
        return rho.view(-1, self.grid_rows, self.grid_cols)


    def forward(self, t, rho):
        B, row, col = rho.shape
        assert row == self.grid_rows and col == self.grid_cols

        t_index = (t).long()

        # 使用可微分的位置到格子分配
        grid_probs_next = self.position_to_grid_probs(self.position[..., t_index + 1, :, :])
        pred_rho = self.grid_to_rho(grid_probs_next)


        # 1. 数据准备
        B, ped = self.velocity[..., t_index, :, :].shape[:2]

        # 获取当前和下一时刻的格子概率分布
        senders_in_probs = self.position_to_grid_probs(self.position[..., t_index, :, :])  # [B, ped, nodes]
        receivers_in_probs = grid_probs_next  # [B, ped, nodes]

        # 计算位置last的格子分布
        position_last = self.position[..., t_index + 1, :, :] + self.velocity[..., t_index + 1, :, :] * self.time_unit
        senders_out_probs = receivers_in_probs
        receivers_out_probs = self.position_to_grid_probs(position_last)

        # 获取嵌入向量
        w_emb = self.node_w[:-1]  # [nodes, embed_dim]
        b_emb = self.node_b[:-1] # [nodes, embed_dim]

        # 计算加权平均的嵌入向量
        w_in_senders = torch.einsum('bpn,nd->bpd', senders_in_probs[..., :-1], w_emb)  # [B, ped, embed_dim]
        w_in_receivers = torch.einsum('bpn,nd->bpd', receivers_in_probs[..., :-1], w_emb)
        b_in_senders = torch.einsum('bpn,nd->bpd', senders_in_probs[..., :-1], b_emb)
        b_in_receivers = torch.einsum('bpn,nd->bpd', receivers_in_probs[..., :-1], b_emb)

        v_in = torch.norm(self.velocity[..., t_index, :, :], dim=-1)  # [B, ped]

        # 计算流入量
        w_in = torch.einsum('bpd,bpd->bp', w_in_senders, w_in_receivers)  # [B, ped]
        b_in = torch.einsum('bpd,bpd->bp', b_in_senders, b_in_receivers)  # [B, ped]
        inflow = (w_in * v_in + b_in)  # [B, ped]

        # 计算rho的加权聚合
        rho_ext = torch.cat([rho.view(B, -1), torch.zeros((B, 1), device=rho.device)], dim=-1)  # [B, nodes]
        rho_senders = torch.einsum('bpn,bn->bp', senders_in_probs, rho_ext)  # [B, ped]

        weighted_inflow = inflow * rho_senders  # [B, ped]

        # 使用概率分布进行聚合
        inflows = torch.einsum('bp,bpn->bn', weighted_inflow, receivers_in_probs[..., :-1])  # [B, nodes]

        # 计算流出量
        w_senders_out = torch.einsum('bpn,nd->bpd', senders_out_probs[..., :-1], w_emb)
        w_receivers_out = torch.einsum('bpn,nd->bpd', receivers_out_probs[..., :-1], w_emb)
        b_senders_out = torch.einsum('bpn,nd->bpd', senders_out_probs[..., :-1], b_emb)
        b_receivers_out = torch.einsum('bpn,nd->bpd', receivers_out_probs[..., :-1], b_emb)

        v_out = torch.norm(self.velocity[..., t_index + 1, :, :], dim=-1)  # [B, ped]

        w_out = torch.einsum('bpd,bpd->bp', w_senders_out, w_receivers_out)
        b_out = torch.einsum('bpd,bpd->bp', b_senders_out, b_receivers_out)
        outflow = (w_out * v_out + b_out)  # [B, ped]

        # 使用概率分布进行聚合
        outflows = torch.einsum('bp,bpn->bn', outflow, senders_out_probs[..., :-1])  # [B, nodes]
        pred_rho = pred_rho.view(B, -1)  # [B, nodes]
        outflows = outflows * pred_rho
        # 计算净流量
        flows = inflows - outflows # 去掉虚拟格子

        return flows.view_as(rho)

class DiffeqSolve_post(nn.Module):
    def __init__(self, odefunc, method, odeint_rtol=1e-4, odeint_atol=1e-5):
        nn.Module.__init__(self)

        self.ode_method = method
        self.odefunc = odefunc

        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        ode_pred_rho = odeint(self.odefunc,
                              first_point,
                              time_steps_to_predict,
                              rtol=self.rtol,
                              atol=self.atol,
                              method=self.ode_method)

        ode_pred_rho = ode_pred_rho.permute(1, 0, 2, 3)  # 调整维度顺序为 (batch_size, time_steps, rows, cols)

        return ode_pred_rho[:, 1:, ...]

class CoupledModel_post(nn.Module):
    def __init__(self, config):
        super().__init__()

        ST_net = getattr(diffusion, config.diffnet)
        self.pre_model = ST_net(config)
        # self.load_model(config.model_dir)
        self.horizon = config.valid_steps - 1
        self.config = config

        # 时间参数
        self.ode_method = config.ode_method
        self.atol = config.odeint_atol
        self.rtol = config.odeint_rtol
        self.grid_size = config.grid_size
        self.min_x = config.min_x
        self.min_y = config.min_y
        self.grid_rows = config.grid_rows
        self.grid_cols = config.grid_cols

        if 'ucy' in config.data_dict_path:
            self.tau = 5 / 6
        else:
            self.tau = 2

        self.odefunc = DensityODE_post(self.pre_model, config.valid_steps, self.grid_size,
                                  self.min_x, self.min_y, self.grid_rows, self.grid_cols, self.config.topk_ped,
                                  self.config.sight_angle_ped,
                                  self.config.dist_threshold_ped, self.config.topk_obs,
                                  self.config.sight_angle_obs, self.config.dist_threshold_obs, self.config.esti_goal)

        self.diffeq_solver = DiffeqSolve_post(self.odefunc,
                                          self.ode_method,
                                          odeint_rtol=self.rtol,
                                          odeint_atol=self.atol)

    def forward(self, batch, lam_1=1.0, lam_2=1.0):
        time_steps_to_predict = torch.arange(start=0, end=self.horizon + 1, step=1).float().to(batch.labels.device)
        # time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
        self.odefunc.set_data(batch)
        position_start = batch.labels[..., 0, :, :2]
        grid = self.odefunc.position_to_grid_probs(position_start)
        first_point = self.odefunc.grid_to_rho(grid)
        ode_pred_rho = self.diffeq_solver(first_point, time_steps_to_predict)

        rho_lable = self.odefunc.position_to_grid_probs(
            batch.labels[..., 1:, :, :2].reshape(-1, *position_start.shape[1:]))
        rho_lable = self.odefunc.grid_to_rho(rho_lable)
        rho_lable = rho_lable.view(position_start.shape[0], self.horizon, self.grid_rows, self.grid_cols)

        loss_ode_rho = F.mse_loss(ode_pred_rho, rho_lable)
        loss = torch.tensor(0.0, device=batch.labels.device)  # 初始化损失为0
        loss_nn_rho = torch.tensor(0.0, device=batch.labels.device)
        loss_nn_a = torch.tensor(0.0, device=batch.labels.device)
        loss_nn_position = torch.tensor(0.0, device=batch.labels.device)
        loss_ode_nn_rho = torch.tensor(0.0, device=batch.labels.device)
        return {
            'loss': loss,
            'loss_nn_rho': loss_nn_rho,
            'loss_nn_a': loss_nn_a,
            'loss_nn_position': loss_nn_position,
            'loss_ode_rho': loss_ode_rho,
            'loss_ode_nn_rho': loss_ode_nn_rho
        }

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        data = torch.load(str(f'{model_path}/model-best.pt'))
        self.pre_model.load_state_dict(data['model'])
        print(f"prediction Model loaded from {model_path} successfully.")


    def init_optimizers(self, main_optimizer):
        """初始化解耦训练的优化器"""
        # 轨迹网络优化器 (继承主优化器配置)
        self.opt_traj = type(main_optimizer)(
            [
                {'params': self.pre_model.parameters()},  # 原有加速度网络

            ],
            **{k: v for k, v in main_optimizer.defaults.items()}
        )

        # 物理网络优化器 (更小的学习率)
        self.opt_physics = type(main_optimizer)(
            [
                {'params': self.odefunc.phys_net.parameters()},
                {'params': self.odefunc.fusion_net.parameters()},
                {'params': [self.odefunc.temperature]},  # 温度参数
                {'params': self.odefunc.node_w},  # 动态图参数
                {'params': self.odefunc.node_b}
            ],
            lr=main_optimizer.defaults['lr'] * 0.5,
            **{k: v for k, v in main_optimizer.defaults.items() if k != 'lr'}
        )

class PreModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ST_net = getattr(diffusion, config.diffnet)
        self.pre_model = ST_net(config)
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2


    def forward(self, batch):
        a_cur = batch.acceleration[..., 0, :, :]
        p_cur = batch.position[..., 0, :, :]
        v_cur = batch.velocity[..., 0, :, :]
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)
        neigh_ped_mask = batch.neigh_ped_mask[..., 0, :, :]
        self_features = batch.self_features[..., 0, :, :]
        near_ped_idx = batch.near_ped_idx[..., 0, :, :]
        history_features = batch.self_hist_features[..., 0, :, :, :]
        history_features = clear_nan(history_features)
        near_obstacle_idx = batch.near_obstacle_idx[..., 0, :, :]
        obstacles = batch.obstacles.unsqueeze(0).repeat(near_obstacle_idx.shape[0], 1, 1)

        neigh_obs_mask = batch.neigh_obs_mask[..., 0, :, :]
        beta = 0.  # 归一化时间
        device = p_cur.device
        beta = torch.full((a_cur.shape[0],), beta, device=device)
        a_next = self.pre_model(x=a_cur,  # c, n, 2
                            beta=beta,
                            context=(curr.detach(),
                                     neigh_ped_mask.detach(),
                                     self_features.detach(),
                                     near_ped_idx.detach(),
                                     history_features.detach(),
                                     obstacles.detach(),
                                     near_obstacle_idx.detach(),
                                     neigh_obs_mask.detach()),
                            nei_list=None,
                            t=None)  # chec**

        a_true = batch.acceleration[..., 1, :, :]  # *c, n, 2
        a_next[batch.mask_a_pred[..., 1, :] == 0] = 0.  # *c, n, 2
        loss = F.mse_loss(a_next, a_true)
        return loss


    def simulate(self, data, t_start=0):
        args = self.config
        # if load_model:
        #     self.load_model(args, set_model=False, finetune_flag=self.finetune_flag)
        ped = Pedestrians()
        destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n

        desired_speed = data.self_features[..., t_start, :, -1].unsqueeze(-1)  # *c, n, 1

        if self.config.esti_goal == 'acce':
            history_features = data.self_hist_features[..., t_start, :, :, :]
            history_features = clear_nan(history_features)
        elif self.config.esti_goal == 'pos':
            raise NotImplementedError
            # hist_pos = data.self_hist_features[...,t_start, :, :, :2]
            # hist_vel = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_acce = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_vel[:,1:,:] = data.self_hist_features[...,t_start, :, :-1, 2:4]
            # hist_acce[:,2:,:] = data.self_hist_features[...,t_start, :, :-2, 4:6]
            # history_features = torch.cat((hist_pos, hist_vel, hist_acce), dim=-1)
            # history_features = clear_nan(history_features)
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        self_feature = data.self_features[..., t_start, :, :]

        near_ped_idx = data.near_ped_idx[..., t_start, :, :]
        neigh_ped_mask = data.neigh_ped_mask[..., t_start, :, :]
        near_obstacle_idx = data.near_obstacle_idx[..., t_start, :, :]
        neigh_obs_mask = data.neigh_obs_mask[..., t_start, :, :]

        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)  # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        p_res = torch.zeros(data.position.shape, device=args.device)  # *c, t, n, 2
        v_res = torch.zeros(data.velocity.shape, device=args.device)  # *c, t, n, 2
        a_res = torch.zeros(data.acceleration.shape, device=args.device)  # *c, t, n, 2
        dest_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        ped_force_res = torch.zeros(data.acceleration.shape, device=args.device)

        p_res[..., :t_start + 1, :, :] = data.position[..., :t_start + 1, :, :]
        v_res[..., :t_start + 1, :, :] = data.velocity[..., :t_start + 1, :, :]
        a_res[..., :t_start + 1, :, :] = data.acceleration[..., :t_start + 1, :, :]

        mask_p_new = torch.zeros(mask_p_.shape, device=mask_p_.device)
        mask_p_new[..., :t_start + 1, :] = data.mask_p[..., :t_start + 1, :].long()

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        for t in tqdm(range(t_start, data.num_frames)):
            p_res[..., t, :, :] = p_cur
            v_res[..., t, :, :] = v_cur
            a_res[..., t, :, :] = a_cur
            # mask_p_new[..., t, ~p_cur[:, 0].isnan()] = 1
            mask_p_new[..., t, :][~p_cur[..., 0].isnan()] = 1

            # a_next = self.diffusion.sample(*state_features)[0]
            if self.config.esti_goal == 'acce':
                beta = 0.  # 归一化时间
                beta = torch.full((p_cur.shape[0],), beta, device=p_cur.device)
                a_next = self.pre_model(x=a_cur,  # c, n, 2
                                    beta=beta,
                                    context=(curr.detach(),
                                             neigh_ped_mask.detach(),
                                             self_feature.detach(),
                                             near_ped_idx.detach(),
                                             history_features.detach(),
                                             obstacles.detach(),
                                             near_obstacle_idx.detach(),
                                             neigh_obs_mask.detach()),
                                    nei_list=None,
                                    t=None)  # chec**

                # print(a_next.sum())
                # if a_next[mask_p_[t]].max()>15 or a_next[mask_p_[t]].isnan().any():
                #     pdb.set_trace()
                # dest force part
                self_feature = self_feature
                desired_speed = self_feature[..., -1].unsqueeze(-1)
                temp = torch.norm(self_feature[..., :2], p=2, dim=-1, keepdim=True)
                temp_ = temp.clone()
                temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
                dest_direction = self_feature[..., :2] / temp_  # des,direction
                pred_acc_dest = (desired_speed * dest_direction - self_feature[..., 2:4]) / self.tau
                pred_acc_ped = a_next - pred_acc_dest
                if t < data.num_frames - 1:
                    dest_force_res[..., t + 1, :, :] = pred_acc_dest
                    ped_force_res[..., t + 1, :, :] = pred_acc_ped

                v_next = v_cur + a_cur * data.time_unit
                p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            elif self.config.esti_goal == 'pos':
                raise NotImplementedError
                # if self.config.history_dim==6:
                #     p_next = self.diffusion.sample(history_features.unsqueeze(0), dest_features.unsqueeze(0))
                # elif self.config.history_dim==2:
                #     p_next = self.diffusion.sample(history_features[...,:2].unsqueeze(0), dest_features.unsqueeze(0))
                #     v_next = v_cur
                #     a_next = a_cur

            # update destination & mask_p
            out_of_bound = torch.tensor(float('nan'), device=args.device)
            dis_to_dest = torch.norm(p_cur - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] += 1  # *c, n
            # TODO: currently don't delete?
            p_next[dest_idx_cur > dest_num - 1, :] = out_of_bound  # destination arrived

            dest_idx_cur[dest_idx_cur > dest_num - 1] -= 1
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze(1)  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            v_cur = v_next
            a_cur = a_next
            # curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)
            # update hist_v
            hist_v = self_feature[..., :, 2:-3]  # *c, n, 2*x
            hist_v[..., :, :-2] = hist_v[..., :, 2:]
            hist_v[..., :, -2:] = v_cur

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]

                    # update hist_v
                    hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]

            curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)  # *c, N, 6

            # update hist_features
            if self.config.esti_goal == 'acce':
                history_features[..., :-1, :] = history_features[..., 1:, :].clone()  # history_features: n, len, 6
                history_features[..., -1, :2] = p_cur.clone()
                history_features[..., -1, 2:4] = v_cur.clone()
                history_features[..., -1, 4:6] = a_cur.clone()
                history_features = clear_nan(history_features)

            elif self.config.esti_goal == 'pos':
                if self.config.history_dim == 2:
                    history_features[..., :-1, :] = history_features[:, 1:, :].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur.clone()

            # calculate features
            if self.config.esti_goal == 'acce':
                ped_features, obs_features, dest_features, \
                    near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask = ped.get_relative_features(
                    p_cur.unsqueeze(-3), v_cur.unsqueeze(-3), a_cur.unsqueeze(-3),
                    dest_cur.unsqueeze(-3), obstacles, args.topk_ped, args.sight_angle_ped,
                    args.dist_threshold_ped, args.topk_obs,
                    args.sight_angle_obs, args.dist_threshold_obs)
                ped_features = ped_features.squeeze(1)
                obs_features = obs_features.squeeze(1)
                dest_features = dest_features.squeeze(1)
                near_ped_idx = near_ped_idx.squeeze(1)
                neigh_ped_mask = neigh_ped_mask.squeeze(1)
                near_obstacle_idx = near_obstacle_idx.squeeze(1)
                neigh_obs_mask = neigh_obs_mask.squeeze(1)

                self_feature = torch.cat((dest_features, hist_v, a_cur, desired_speed), dim=-1)
            elif self.config.esti_goal == 'pos':
                raise NotImplementedError
                dest_features = dest_cur - p_cur
                dest_features[dest_features.isnan()] = 0.

        output = RawData(p_res.squeeze(0), v_res.squeeze(0), a_res.squeeze(0), destination.squeeze(0), destination.squeeze(0), obstacles.squeeze(0),
                              mask_p_new.squeeze(0), meta_data=data.meta_data)
        return output, dest_force_res, ped_force_res