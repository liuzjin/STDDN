import os
import argparse
import torch
import dill
# import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
from tensorboardX import SummaryWriter

from utils import data_loader as LOADER
from utils.utils import mask_mse_func, post_process
import data.data as DATA
import data.dataset as DATASET
import torch.nn.functional as F
import utils.metrics as METRIC
from models.st_model import CoupledModel, PreModel
from torch.utils.data import Dataset, DataLoader


class Trainer():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True  
        self._build()

        self.patience = 4

    def set_requires_grad(self, model, component, requires):
        """设置特定组件的梯度要求"""
        if component == 'trajectory':
            for param in model.pre_model.parameters():
                param.requires_grad = requires
        elif component == 'physics':
            if self.config.fused_v:
                for param in model.odefunc.phys_net.parameters():
                    param.requires_grad = requires
                for param in model.odefunc.fusion_net.parameters():
                    param.requires_grad = requires
            model.odefunc.node_w.requires_grad = requires
            model.odefunc.node_b.requires_grad = requires
            model.odefunc.temperature.requires_grad = requires

    def train(self):

        train_loaders = self.finetune_train_loaders
        val_list = self.finetune_valid_list
        optimizer = self.ft_optimizer
        scheduler = self.ft_scheduler
        total_epochs = self.config.total_epochs
        finetune_flag = self.config.finetune

        loss_epoch = {}
        self.model.init_optimizers(optimizer)
        wait = 0
        accumulation_steps = 1
        if self.stage == 'trajectory':
            lam_1, lam_2 = 1.0, 0.1
            print(f"=======================当前阶段: {self.stage}=============================")
        elif self.stage == 'physics':
            lam_1, lam_2 = 0.1, 1.0
            print(f"=======================当前阶段: {self.stage}=============================")
        elif self.stage == 'joint':
            lam_1, lam_2 = 1.0, 1.0
            print(f"=======================当前阶段: {self.stage}, lam_1:{lam_1}, lam_2:{lam_2}============================")
        for epoch in range(self.epoch, total_epochs + 1):
            if epoch != -1:
                self.model.train()
                loss_epoch[epoch] = []

                nn_a = []
                nn_position = []
                ode_rho = []
                nn_collision = []
                for i, train_loader in enumerate(train_loaders):
                    if self.config.finetune:

                        assert type(train_loader) == DATA.ChanneledTimeIndexedPedData
                        if self.stage in ['trajectory']:
                            self.set_requires_grad(self.model, 'trajectory', True)
                            self.set_requires_grad(self.model, 'physics', False)
                            if i % accumulation_steps == 0:
                                self.model.opt_traj.zero_grad()
                        if self.stage in ['physics']:
                            self.set_requires_grad(self.model, 'trajectory', False)
                            self.set_requires_grad(self.model, 'physics', True)
                            if i % accumulation_steps == 0:
                                self.model.opt_physics.zero_grad()
                        if self.stage in ['joint']:
                            self.set_requires_grad(self.model, 'trajectory', True)
                            self.set_requires_grad(self.model, 'physics', True)
                            if i % accumulation_steps == 0:
                                self.model.opt_traj.zero_grad()
                                self.model.opt_physics.zero_grad()
                        output = self.model(train_loader, lam_1=lam_1, lam_2=lam_2)
                        loss = output['loss']
                        loss_nn_a = output['loss_nn_a']
                        loss_nn_position = output['loss_nn_position']
                        loss_ode_rho = output['loss_ode_rho']
                        loss_nn_collision = output['loss_nn_collision']

                        loss_epoch[epoch].append(loss.item())
                        nn_a.append(loss_nn_a.item())
                        nn_position.append(loss_nn_position.item())
                        ode_rho.append(loss_ode_rho.item())
                        nn_collision.append(loss_nn_collision.item())

                        # loss.backward()
                        # optimizer.step()
                        if self.stage == 'trajectory':
                            ((output['loss_nn_a'] + output['loss_nn_position']) / accumulation_steps).backward()
                            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loaders):
                                self.model.opt_traj.step()
                        elif self.stage == 'physics':
                            (output['loss_ode_rho'] / accumulation_steps).backward()
                            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loaders):
                                self.model.opt_physics.step()
                        else:  # joint
                            (output['loss'] / accumulation_steps).backward()
                            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loaders):
                                self.model.opt_traj.step()
                                self.model.opt_physics.step()

                        print('last lr:', scheduler.get_last_lr())
                        scheduler.step()
                        mse_mean = np.mean(loss_epoch[epoch])
                        nn_a_mean = np.mean(nn_a)
                        nn_position_mean = np.mean(nn_position)
                        ode_rho_mean = np.mean(ode_rho)
                        nn_collision_mean = np.mean(nn_collision)

                        print(f"Epoch {epoch} NO {i} MSE: {mse_mean} "
                              f"nn_a_mean: {nn_a_mean},nn_position_mean: {nn_position_mean}, "
                              f"ode_rho_mean: {ode_rho_mean}, nn_collision_mean: {nn_collision_mean}")
                        del output,  loss, loss_nn_a, loss_nn_position, loss_ode_rho
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        self.optimizer.zero_grad()
                        # torch.autograd.set_detect_anomaly(True)
                        assert type(train_loader) == DATA.ChanneledTimeIndexedPedData
                        # train_loader = self.to(train_loader)
                        loss = self.model(train_loader)
                        loss_epoch[epoch].append(loss.item())
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        print('last lr:', scheduler.get_last_lr())
                        mse_mean = np.mean(loss_epoch[epoch])

                        print(f"Epoch {epoch} NO {i} MSE: {mse_mean} ")
                        del loss

                self.log_writer.add_scalar('train_MSE', np.mean(loss_epoch[epoch]), epoch)

            if (epoch) % self.config.eval_every == 0:
                self.model.eval()
                mse_list = []
                mae_list = []
                ot_list = []
                FDE_list = []
                mmd_list = []
                collision_list = []
                dtw_list = []
                ipd_list = []
                ipd_mmd_list = []
                for i, val_data in enumerate(val_list):
                    with torch.no_grad():
                        val_data = self.to_batch(val_data)
                        traj_pred, dest_force, ped_force = self.model.simulate(val_data, t_start=self.config.skip_frames)
                        # traj_pred, dest_force, ped_force = self.model.generate_multistep(val_data, t_start=self.config.skip_frames)
                        val_data = self.batch_to_data(val_data)
                        # ipd_mmd = METRIC.get_nearby_distance_mmd(traj_pred.position, traj_pred.velocity,
                        #                                          val_data.labels[..., :2],
                        #                                          val_data.labels[..., 2:4],
                        #                                          val_data.mask_p_pred.long(),
                        #                                          self.config.dist_threshold_ped,
                        #                                          self.config.topk_ped * 2, reduction='mean')
                        # ipd_mmd_list.append(ipd_mmd)
                        p_pred = traj_pred.position
                        # p_pred_ = p_pred.clone()
                        # p_pred_[:-1, :, :] = p_pred_[1:, :, :].clone()
                        # p_pred = p_pred_
                        mask_p_pred = val_data.mask_p_pred.long()  # (*c) t, n
                        labels = val_data.labels[..., :2]

                        # torch.save(labels, self.logs_dir + f'/{epoch}_labels.pth')
                        # torch.save(p_pred, self.logs_dir + f'/{epoch}_p_pred.pth')
                        # torch.save(mask_p_pred, self.logs_dir + f'/{epoch}_mask_p_pred.pth')
                        # torch.save(dest_force, self.logs_dir + f'/{epoch}_dest_force.pth')
                        # torch.save(ped_force, self.logs_dir + f'/{epoch}_ped_force.pth')

                        # plot_trajectory(p_pred, labels, name=time.strftime('%Y-%m-%d-%H-%M'))
                        # collision = METRIC.collision_count(p_pred, 0.5, reduction='sum')
                        # FDE = METRIC.fde_at_label_end(p_pred, labels, reduction='mean')

                        p_pred = post_process(val_data, p_pred, traj_pred.mask_p, mask_p_pred)
                        dtw = METRIC.dtw_tensor(p_pred, labels, mask_p_pred, mask_p_pred, reduction='mean')
                        dtw_list.append(dtw)

                        # func = lambda x: x * torch.exp(-x)
                        # ipd = METRIC.inter_ped_dis(p_pred, labels, mask_p_pred, reduction='mean',
                        #                            applied_func=func)
                        # ipd_list.append(ipd)

                        loss = F.mse_loss(p_pred[mask_p_pred == 1], labels[mask_p_pred == 1],
                                          reduction='mean') * 2
                        loss = loss.item()
                        mse_list.append(loss)
                        mae = METRIC.mae_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                        mae_list.append(mae)

                    ot = METRIC.ot_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean',
                                                  dvs=self.config.device)
                    mmd = METRIC.mmd_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                    #
                    ot_list.append(ot)
                    mmd_list.append(mmd)
                    # FDE_list.append(FDE)
                    # collision_list.append(collision)
                    # ade,fde = evaluation.compute_batch_statistics2(traj_pred,gt,best_of=True)
                    # eval_ade_batch_errors.append(ade)
                    # eval_fde_batch_errors.append(fde)
                # ade = np.mean(eval_ade_batch_errors)
                # fde = np.mean(eval_fde_batch_errors)
                mse = np.mean(mse_list)
                mae = np.mean(mae_list)
                ot = np.mean(ot_list)
                # FDE = np.mean(FDE_list)
                mmd = np.mean(mmd_list)
                # collision = np.mean(collision)
                dtw = np.mean(dtw_list)
                # ipd = np.mean(ipd_list)
                # ipd_mmd = np.mean(ipd_mmd_list)
                # if self.config.dataset == "eth":
                #     ade = ade/0.6
                #     fde = fde/0.6
                # elif self.config.dataset == "sdd":
                #     ade = ade * 50
                #     fde = fde * 50
                # print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
                print(f"Epoch {epoch} MSE: {mse} MAE: {mae}, OT: {ot} MMD: {mmd}, DWT: {dtw}")
                # print(f"Epoch {epoch} OT: {ot} MMD: {mmd}")
                # print(f"Epoch {epoch} collision: {collision} FDE: {FDE}")
                # print(f"Epoch {epoch} dtw: {dtw} inter ped distance mmd: {ipd_mmd}")

                # self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")
                # self.log_writer.add_scalar('ADE', ade, epoch)
                # self.log_writer.add_scalar('FDE', fde, epoch)
                self.log.info(f"Epoch {epoch} MSE: {mse} MAE: {mae}, OT: {ot} MMD: {mmd}, DWT: {dtw}")
                # self.log.info(f"Epoch {epoch} OT: {ot} MMD: {mmd}")
                # self.log.info(f"Epoch {epoch} collision: {collision} FDE: {FDE}")
                # self.log.info(f"Epoch {epoch} dtw: {dtw} inter ped distance mmd: {ipd_mmd}")
                self.log.info(" ")
                self.log_writer.add_scalar('MSE', mse, epoch)
                self.log_writer.add_scalar('MAE', mae, epoch)
                # self.log_writer.add_scalar('OT', ot, epoch)
                # self.log_writer.add_scalar('MMD', mmd, epoch)
                # self.log_writer.add_scalar('Collision', collision, epoch)
                # self.log_writer.add_scalar('fde', FDE, epoch)
                # self.log_writer.add_scalar('dtw', dtw, epoch)
                # self.log_writer.add_scalar('ipd_mmd', ipd_mmd, epoch)
                if mae < self.min_val_loss:
                    wait = 0
                    if not self.config.finetune:
                        save_dir = os.path.join(self.model_dir, f'pre_train')
                    else:
                        save_dir = os.path.join(self.model_dir, f'train')
                    print('Val loss decrease from {:.4f} to {:.4f}, '
                        'epoch {}'.format(self.min_val_loss, mae, epoch))
                    self.min_val_loss = mae
                    os.makedirs(save_dir, exist_ok=True)
                    save_name = f'best_model.pth'
                    save_path = os.path.join(save_dir, save_name)
                    model_dict = self.model.state_dict() if self.config.finetune else self.model.pre_model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss_epoch': loss_epoch,
                        'config': self.config,
                        'min_val_loss': self.min_val_loss,
                        'stage': self.stage,
                    }, save_path)

                else:
                    wait += 1
                    if wait == self.patience and self.stage == 'trajectory':
                        wait = 0
                        print('stage change at epoch: %d' % epoch)
                        self.stage = 'physics'
                        lam_1, lam_2 = 0.1, 1.0  # 弱化轨迹损失
                        print(f"=======================当前阶段: {self.stage}===================")
                        continue
                    if wait == self.patience and self.stage == 'physics':
                        wait = 0
                        print('stage change at epoch: %d' % epoch)
                        self.stage = 'joint'
                        lam_1, lam_2 = 1.0, 1.0
                        print(f"======================当前阶段: {self.stage}===================")
                        continue
                    if wait == self.patience and self.stage == 'joint':
                        print(f'Early stopping at epoch: {epoch},best mae: {self.min_val_loss}' )
                        break

                del traj_pred
                # import gc
                # gc.collect()
                # torch.cuda.empty_cache()

                self.model.train()

            # self.train_dataset.augment = False

    def _build(self):
        # self.accelerator = Accelerator()
        # device = self.accelerator.device
        # self.config.device = device
        self._build_dir()

        self._build_data_loader()

        self._build_model()
        self._build_optimizer()
        if self.config.resume or self.config.mode == 'test':
            if self.config.finetune:
                checkpoint_path = f'{self.model_dir}/train/best_model.pth'
            else:
                checkpoint_path = f'{self.model_dir}/pre_train/best_model.pth'
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Resuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.ft_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.ft_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.epoch = checkpoint['epoch']
                self.min_val_loss = checkpoint['min_val_loss']
                self.stage = checkpoint['stage']
                print("> Checkpoint loaded!")
        else:
            self.epoch = 0
            self.stage = 'joint'
            self.min_val_loss = float('inf')
            print("Resume flag is set but checkpoint path is invalid, training from scratch.")

        self.model = self.model.cuda()

    def _build_dir(self):
        self.model_dir = osp.join("./experiments", self.config.exp_name)
        import sys
        debug_flag = 'run' if sys.gettrace() == None else 'debug'
        print('running in', debug_flag, 'mode')
        logs_dir = osp.join(self.model_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
        logs_dir += debug_flag
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=logs_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(logs_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)
        self.log.info(time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.log.info("Config:")
        for item in self.config.items():
            self.log.info(item)

        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        print("> Directory built!")

    def _build_optimizer(self):
        if 'ft_lr' not in self.config.keys():
            if 'ucy' in self.config.data_dict_path:
                self.config.ft_lr = self.config.lr / 100
            elif 'gc' in self.config.data_dict_path:
                self.config.ft_lr = self.config.lr / 1000
            elif 'eth' in self.config.data_dict_path:
                self.config.ft_lr = self.config.lr / 100
            elif 'hotel' in self.config.data_dict_path:
                self.config.ft_lr = self.config.lr / 100

        print('ft_lr:', self.config.ft_lr)
        self.ft_optimizer = optim.Adam([
            # {'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
            {'params': self.model.parameters()}
        ],
            lr=self.config.ft_lr,
            weight_decay=1e-5)
        if 'hotel' in self.config.data_dict_path:
            self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=10, gamma=0.5)
        else:
            self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=3, gamma=0.5)

        self.log.info(f'(\'ft_lr\', {self.config.ft_lr})')
        print("> Optimizer built!")

    def _build_model(self):
        """ Define Model """
        config = self.config
        # model = AutoEncoder(config, encoder=None)
        if self.config.finetune:
            model = CoupledModel(config)
        else:
            model = PreModel(config)
        self.model = model
        self.log.info("\n")
        print("> Model built!")

    def _build_data_loader(self):

        if self.config.rebuild_dataset == True:
            if self.config.dataset_type == 'timeindex':
                finetune_dataset = DATASET.TimeIndexedPedDataset2()
            else:
                raise NotImplementedError
            finetune_dataset.load_data(self.config.data_config, grid_size=self.config.grid_size)

            print('number of finetune training dataset: ', len(finetune_dataset.raw_data['train']))
            finetune_dataset.build_dataset(self.config, finetune_flag=(self.config.finetune_trainmode == 'multistep'))

            with open(self.config.data_dict_path, 'wb') as f:
                dill.dump(finetune_dataset, f, protocol=dill.HIGHEST_PROTOCOL)
        elif self.config.finetune == True:
            with open(self.config.data_dict_path, 'rb') as f:
                finetune_dataset = dill.load(f)
        elif self.config.finetune == False:

            finetune_dataset = DATASET.TimeIndexedPedDataset2()
            finetune_dataset.load_data(self.config.data_config, grid_size=self.config.grid_size)

            print('number of finetune training dataset: ', len(finetune_dataset.raw_data['train']))
            finetune_dataset.build_dataset(self.config, finetune_flag=(self.config.finetune_trainmode == 'multistep'))
        self.config.min_x = finetune_dataset.min_x
        self.config.min_y = finetune_dataset.min_y
        self.config.max_x = finetune_dataset.max_x
        self.config.max_y = finetune_dataset.max_y
        self.config.grid_cols = int(np.ceil((finetune_dataset.max_x - finetune_dataset.min_x) / self.config.grid_size))
        self.config.grid_rows = int(np.ceil((finetune_dataset.max_y - finetune_dataset.min_y) / self.config.grid_size))

        if self.config.finetune_flag == True:

            finetune_train_list = finetune_dataset.train_data
            if self.config.finetune_trainmode == 'singlestep':
                self.finetune_train_loaders = []
                assert type(finetune_train_list) == list
                for item in finetune_train_list:
                    self.finetune_train_loaders.append(DataLoader(
                        item,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        drop_last=True))
            elif self.config.finetune_trainmode == 'multistep':
                assert type(finetune_train_list[0]) == DATA.ChanneledTimeIndexedPedData
                # if self.config.val
                self.finetune_train_loaders = LOADER.data_loader(finetune_train_list, self.config.batch_size,
                                                                 self.config.seed, shuffle=False, drop_last=False)

                # dataset = CustomDataset(train_loaders)
                # self.finetune_train_loaders = DataLoader(dataset, batch_size=1, shuffle=False,
                #                                          num_workers=0,collate_fn=custom_collate_fn)


            self.finetune_valid_list = finetune_dataset.valid_data


        return

    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")

    def to_batch(self, data):
        data.ped_features = data.ped_features.unsqueeze(0)
        data.obs_features = data.obs_features.unsqueeze(0)
        data.self_features = data.self_features.unsqueeze(0)
        data.self_hist_features = data.self_hist_features.unsqueeze(0)
        data.near_ped_idx = data.near_ped_idx.unsqueeze(0)
        data.neigh_ped_mask = data.neigh_ped_mask.unsqueeze(0)
        data.near_obstacle_idx = data.near_obstacle_idx.unsqueeze(0)
        data.neigh_obs_mask = data.neigh_obs_mask.unsqueeze(0)
        data.labels = data.labels.unsqueeze(0)
        data.mask_p = data.mask_p.unsqueeze(0)
        data.mask_v = data.mask_v.unsqueeze(0)
        data.mask_a = data.mask_a.unsqueeze(0)
        data.mask_p_pred = data.mask_p_pred.unsqueeze(0)
        data.mask_v_pred = data.mask_v_pred.unsqueeze(0)
        data.mask_a_pred = data.mask_a_pred.unsqueeze(0)
        data.position = data.position.unsqueeze(0)
        data.velocity = data.velocity.unsqueeze(0)
        data.acceleration = data.acceleration.unsqueeze(0)
        data.destination = data.destination.unsqueeze(0)
        data.dest_idx = data.dest_idx.unsqueeze(0)
        data.waypoints = data.waypoints.unsqueeze(0)
        data.obstacles = data.obstacles.unsqueeze(0)
        # data.num_frames = data.num_frames.to(self.accelerator.device)
        # data.dataset_len = data.dataset_len.to(self.accelerator.device)

        return data

    def batch_to_data(self, data):
        data.ped_features = data.ped_features.squeeze(0)
        data.obs_features = data.obs_features.squeeze(0)
        data.self_features = data.self_features.squeeze(0)
        data.self_hist_features = data.self_hist_features.squeeze(0)
        data.near_ped_idx = data.near_ped_idx.squeeze(0)
        data.neigh_ped_mask = data.neigh_ped_mask.squeeze(0)
        data.near_obstacle_idx = data.near_obstacle_idx.squeeze(0)
        data.neigh_obs_mask = data.neigh_obs_mask.squeeze(0)
        data.labels = data.labels.squeeze(0)
        data.mask_p = data.mask_p.squeeze(0)
        data.mask_v = data.mask_v.squeeze(0)
        data.mask_a = data.mask_a.squeeze(0)
        data.mask_p_pred = data.mask_p_pred.squeeze(0)
        data.mask_v_pred = data.mask_v_pred.squeeze(0)
        data.mask_a_pred = data.mask_a_pred.squeeze(0)
        data.position = data.position.squeeze(0)
        data.velocity = data.velocity.squeeze(0)
        data.acceleration = data.acceleration.squeeze(0)
        data.destination = data.destination.squeeze(0)
        data.dest_idx = data.dest_idx.squeeze(0)
        data.waypoints = data.waypoints.squeeze(0)
        data.obstacles = data.obstacles.squeeze(0)
        return data


    def simulate(self):
        """
        Simulate the model on the given data.
        :param data: The input data for simulation.
        :param t_start: The starting time step for simulation.
        :return: The simulated trajectory.
        """
        self.model.eval()
        mse_list = []
        mae_list = []
        ot_list = []
        FDE_list = []
        mmd_list = []
        collision_list = []
        dtw_list = []
        ipd_list = []
        ipd_mmd_list = []
        for i, val_data in enumerate(self.finetune_valid_list):
            with torch.no_grad():
                val_data = self.to_batch(val_data)
                traj_pred, dest_force, ped_force = self.model.simulate(val_data, t_start=self.config.skip_frames)
                val_data = self.batch_to_data(val_data)
                ipd_mmd = METRIC.get_nearby_distance_mmd(traj_pred.position, traj_pred.velocity,
                                                         val_data.labels[..., :2],
                                                         val_data.labels[..., 2:4],
                                                         val_data.mask_p_pred.long(),
                                                         self.config.dist_threshold_ped,
                                                         self.config.topk_ped * 2, reduction='mean')
                ipd_mmd_list.append(ipd_mmd)
                p_pred = traj_pred.position
                # p_pred_ = p_pred.clone()
                # p_pred_[:-1, :, :] = p_pred_[1:, :, :].clone()
                # p_pred = p_pred_
                mask_p_pred = val_data.mask_p_pred.long()  # (*c) t, n
                labels = val_data.labels[..., :2]

                # torch.save(labels, self.logs_dir + f'/{epoch}_labels.pth')
                # torch.save(p_pred, self.logs_dir + f'/{epoch}_p_pred.pth')
                # torch.save(mask_p_pred, self.logs_dir + f'/{epoch}_mask_p_pred.pth')
                # torch.save(dest_force, self.logs_dir + f'/{epoch}_dest_force.pth')
                # torch.save(ped_force, self.logs_dir + f'/{epoch}_ped_force.pth')

                # plot_trajectory(p_pred, labels, name=time.strftime('%Y-%m-%d-%H-%M'))
                collision = METRIC.collision_count(p_pred, 0.5, reduction='sum')
                FDE = METRIC.fde_at_label_end(p_pred, labels, reduction='mean')

                p_pred = post_process(val_data, p_pred, traj_pred.mask_p, mask_p_pred)
                dtw = METRIC.dtw_tensor(p_pred, labels, mask_p_pred, mask_p_pred, reduction='mean')
                dtw_list.append(dtw)

                func = lambda x: x * torch.exp(-x)
                ipd = METRIC.inter_ped_dis(p_pred, labels, mask_p_pred, reduction='mean',
                                           applied_func=func)
                ipd_list.append(ipd)

                loss = F.mse_loss(p_pred[mask_p_pred == 1], labels[mask_p_pred == 1],
                                  reduction='mean') * 2
                loss = loss.item()
                mse_list.append(loss)
                mae = METRIC.mae_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                mae_list.append(mae)

            import pickle
            save_dict = {
                'p_pred': p_pred.cpu(),
                'labels': labels.cpu(),
                'mask_p_pred': mask_p_pred.cpu()
            }
            data = 'ucy' if 'ucy' in self.config.data_dict_path else 'gc'
            save_dir = os.path.join(self.model_dir, f'predict')
            os.makedirs(save_dir, exist_ok=True)
            with open(f'{save_dir}/{data}_{self.config.valid_steps}.pkl', 'wb') as f:
                pickle.dump(save_dict, f)

            ot = METRIC.ot_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean',
                                          dvs=self.config.device)
            mmd = METRIC.mmd_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')

            ot_list.append(ot)
            mmd_list.append(mmd)
            FDE_list.append(FDE)
            collision_list.append(collision)
            # ade,fde = evaluation.compute_batch_statistics2(traj_pred,gt,best_of=True)
            # eval_ade_batch_errors.append(ade)
            # eval_fde_batch_errors.append(fde)
        # ade = np.mean(eval_ade_batch_errors)
        # fde = np.mean(eval_fde_batch_errors)
        mse = np.mean(mse_list)
        mae = np.mean(mae_list)
        ot = np.mean(ot_list)
        FDE = np.mean(FDE_list)
        mmd = np.mean(mmd_list)
        collision = np.mean(collision)
        dtw = np.mean(dtw_list)
        ipd = np.mean(ipd_list)
        ipd_mmd = np.mean(ipd_mmd_list)
        # if self.config.dataset == "eth":
        #     ade = ade/0.6
        #     fde = fde/0.6
        # elif self.config.dataset == "sdd":
        #     ade = ade * 50
        #     fde = fde * 50
        # print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
        print(f" MSE: {mse} MAE: {mae}")
        print(f" OT: {ot} MMD: {mmd}")
        print(f" collision: {collision} FDE: {FDE}")
        print(f" dtw: {dtw} inter ped distance mmd: {ipd_mmd}")

        self.log.info(f" MSE: {mse} MAE: {mae} OT: {ot} MMD: {mmd} dtw: {dtw} collision: {collision} FDE: {FDE}")
