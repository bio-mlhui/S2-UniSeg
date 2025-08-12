import os
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.optimization.utils import get_total_grad_norm
from einops import repeat, rearrange, reduce
from functools import partial
import numpy as np
import logging
import torchvision.transforms.functional as Trans_F
import copy
from models.registry import register_model
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler 
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
import math
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from detectron2.modeling import BACKBONE_REGISTRY

from .utils import trunc_normal_
import torch.distributed as dist
from torchvision.transforms.functional import normalize as torchv_Normalize



class OptimizeModel(nn.Module):
    @property
    def device(self):
        return self.pixel_mean.device
    """
    optimize_setup:
        optimizer, scheduler都是标准类
        log_lr_idx随着训练不改变
        
    optimize:
        backward, optimzier_step, optimizer_zero_grad, scheduler_step
        
    """
    def __init__(self, ) -> None:
        super().__init__()
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None
        self.log_lr_group_idx: Dict = None

    def optimize_setup(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def sample(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')        

    def optimize_state_dict(self,):
        return {'optim': self.optimizer.state_dict(),}
    
    def load_optimize_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict=state_dict['optim'])

    def get_lr_group_dicts(self, ):
        return  {f'lr': self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def sample_point_similarities(self, backbone_features, code_features, num_points):
        # b c h w, num_points
        H_P, W_P = code_features.shape[-2:]
        H, W = H_P * self.patch_size, W_P * self.patch_size
        sampled_points = torch.rand(num_points, 2)
        sampled_points[:, 0] = sampled_points[:, 0] * H_P
        sampled_points[:, 1] = sampled_points[:, 1] * W_P
        sampled_points = sampled_points.long()
        sampled_points[:, 0].clamp_(0, H_P-1)
        sampled_points[:, 1].clamp_(0, W_P-1)
        similarities = []
        for point in sampled_points:
            query = code_features[:, :, point[0], point[1]] # 1 c
            sim = torch.einsum('c,chw->hw',
                                F.normalize(query[0], dim=0, eps=1e-10),
                                F.normalize(code_features[0], dim=0, eps=1e-10),).cpu() # -1, 1
            sim = F.interpolate(sim[None, None, ...], size=(H, W), align_corners=False, mode='bilinear').clamp(-1, 1)[0, 0]
            similarities.append(sim)

        backbone_similarities = []
        for point in sampled_points:
            query = backbone_features[:, :, point[0], point[1]] # 1 c
            sim = torch.einsum('c,chw->hw',
                                F.normalize(query[0], dim=0, eps=1e-10),
                                F.normalize(backbone_features[0], dim=0, eps=1e-10),).cpu() # -1, 1
            sim = F.interpolate(sim[None, None, ...], size=(H, W), align_corners=False, mode='bilinear').clamp(-1, 1)[0, 0]
            backbone_similarities.append(sim)

        sampled_points = sampled_points * self.patch_size
        return sampled_points, similarities, backbone_similarities


    @torch.no_grad()
    def self_cluster(self, features, gt_masks):
        from models.UN_IMG_SEM.kmeans.kmeans import kmeans
        # b c h w -> b
        _, _, H_P, W_P = features.shape
        assert features.shape[0] == 1
        if self.kmeans_strategy == 'adaptive':
            num_image_classes = len(set(gt_masks.unique().tolist()) - set([-1]))
        else:
            raise ValueError()
        features = features.permute(0, 2,3,1).flatten(0,2) # bhw c
        _, cluster_centers = kmeans(X=features, num_clusters=num_image_classes, device=self.device) # num c
        cluster_logits = torch.einsum('sc,nc->sn', 
                                    F.normalize(features, dim=-1, eps=1e-10),
                                    F.normalize(cluster_centers.to(self.device), dim=-1, eps=1e-10))
        # 把和cluster_center最近的点标注出来
        cluster_logits = rearrange(cluster_logits, '(b h w) n -> b n h w', b=1,h=H_P, w=W_P)
        cluster_logits = F.interpolate(cluster_logits, size=(H_P*self.patch_size, W_P*self.patch_size), mode='bilinear', align_corners=False)
        cluster_ids = cluster_logits.max(dim=1)[1].cpu()
        return cluster_ids, cluster_logits, num_image_classes


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ncrop c
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((  # 从warmup到teacher, 然后一直是teachder
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    @property
    def device(self):
        return self.center.device
    
    @torch.no_grad()
    def box_to_grid(self, coord, H, W, student_coord):
        x1, x2, y1, y2 = coord
        grid_W = int(abs(x1 - x2) / 2 * W)
        grid_H = int(abs(y1 - y2) / 2 * H)

        coord = torch.meshgrid([torch.linspace(y1, y2, grid_H), torch.linspace(x1, x2, grid_W)]) 
        teacher_coord = torch.stack(coord, dim=-1) # (y, x) H W


        x1, x2, y1, y2 = student_coord
        student_coord = torch.meshgrid([torch.linspace(y1, y2, grid_H), torch.linspace(x1, x2, grid_W)]) 
        student_coord = torch.stack(student_coord, dim=-1) # (y, x) H W        
        return  teacher_coord, student_coord


    def forward(self, student_output, teacher_output, epoch, common_samples, teacher_nqs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # list[b k h w], 2+10
        # list[b k h w], 2
        batch_size = len(common_samples)
        student_out = [ foo / self.student_temp for foo in student_output]

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        centered_teacher_out = [F.softmax((foo - self.center[:, :, None, None]) / temp, dim=1) for foo in teacher_output]

        all_loss = []
        for batch_idx in range(batch_size):
            grid_samples = common_samples[batch_idx] # dict
            for iq in range(2): # teacher global
                for v in range(self.ncrops): # student local
                    if v == iq:
                        # we skip cases where student and teacher operate on the same view
                        continue    
                    # g0, g1; g0 lx
                    # g1, g0; g1 lx
                    stu_prefix = f'g{v}' if v < 2 else f'l{v-2}'
                    pair_name = 'g0g1' if v < 2 else f'l{v-2}g{iq}'
                    teacher_feat = centered_teacher_out[iq][batch_idx] # k h w
                    student_feat = student_out[v][batch_idx] # k h' w'

                    teacher_rel_box =grid_samples[f'g{iq}_to_{pair_name}']
                    student_rel_box =grid_samples[f'{stu_prefix}_to_{pair_name}']
                    if teacher_rel_box is None:
                        assert student_rel_box is None
                        continue
                    if teacher_rel_box is not None:
                        assert student_rel_box is not None
                    # 1 h w 2
                    teacher_to_Common, student_to_Common = self.box_to_grid(teacher_rel_box, 
                                                                            teacher_feat.shape[0], teacher_feat.shape[1],
                                                                            student_coord=student_rel_box)
                    teacher_to_Common = teacher_to_Common.unsqueeze(0).to(self.device) # (x1, x2, y1, y2)
                    student_to_Common = student_to_Common.unsqueeze(0).to(self.device) # (x1, x2, y1, y2)

                    # 1 c h w, 1 h_co w_co 2 -> 1 c h_co w_co -> c h_co w_co
                    teacher_common = F.grid_sample(teacher_feat[None, ...], grid=teacher_to_Common, padding_mode="border", align_corners=False)[0].detach()
                    student_common = F.grid_sample(student_feat[None, ...], grid=student_to_Common, padding_mode="border", align_corners=False)[0]

                    # h_common, w_common
                    loss = (- teacher_common * F.log_softmax(student_common, dim=0)).sum(dim=0)

                    all_loss.append(loss.flatten())
        total_loss = torch.cat(all_loss).mean()
        self.update_center(teacher_nqs, teacher_output)
        return total_loss


    @torch.no_grad()
    def update_center(self, teacher_nqs, teacher_output):

        # list[list[ni c] global], batch
        # teacher_nqs = list(zip(*teacher_nqs))

        # list[list[ni c], batch] global
        # batch_center = torch.cat([torch.cat(foo, dim=0).mean(0, keepdim=True) for foo in teacher_nqs], dim=0).mean(0, keepdim=True)

        # 1.
        # list[list[ni c], batch], global
        # batch_center = torch.cat([torch.cat(foo, dim=0) for foo in teacher_nqs], dim=0)
        # batch_center = torch.sum(batch_center, dim=0, keepdim=True)
        # batch_center = batch_center / len(batch_center)

        # # list[b k h w], global=2
        batch_center = torch.cat(teacher_output, dim=0) # 2b k h w

        
        # 2. (ori)
        batch_center = batch_center.mean([2, 3]).sum(dim=0, keepdim=True) # 1 k
        dist.all_reduce(batch_center)

        # 3.
        # batch_center = batch_center.mean(0).sum([1, 2])[None, :] # 2b k
        
        # batch_center = batch_center.sum([0, 2, 3])[None, :]


        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


import models.UN_IMG_SEM.AggSample.utils as utils
import sys
from .FastUniAP import aggo_merge, aggo_merge_graph, aggo_whole_batch
class AggSample(OptimizeModel):
    def __init__(
        self,
        configs,
        num_classes,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],
        batch_size=None): # drop_last=True
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.num_classes = num_classes
        model_configs = configs['model']
        # backbone
        dino_configs = model_configs["backbone"]
        self.feat_type = dino_configs["dino_feat_type"]
        self.backbone = BACKBONE_REGISTRY.get(dino_configs['name'])(dino_configs)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval().cuda()
        self.backbone_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        self.project_dim = self.backbone_dim
        self.student_head = DINOHead(
            in_dim=self.project_dim,
            out_dim=self.num_classes,
            use_bn=False,
            norm_last_layer=model_configs['student_head_norm_last_layer'],
            nlayers=3,
            hidden_dim=2048,
            bottleneck_dim=256
        )
        self.teacher_head = DINOHead(
            in_dim=self.project_dim,
            out_dim=self.num_classes,
            use_bn=False,
        )
        # load head checkpoint from pt model
        if dino_configs['type'] == 'dinov1_vits8':
            state_dict = torch.load(os.path.join(os.environ['PT_PATH'], 'dinov1/dino_deitsmall8_300ep_pretrain_full_checkpoint.pth'), map_location='cpu')
            teacher_state = state_dict['teacher']
            # ['head.projection_head.0.weight', 'head.projection_head.0.bias', 'head.projection_head.2.weight', 'head.projection_head.2.bias', 'head.projection_head.4.weight', 'head.projection_head.4.bias', 'head.prototypes.weight']
            # teacher_state ['head.mlp.0.weight', 'head.mlp.0.bias', 'head.mlp.2.weight', 'head.mlp.2.bias', 'head.mlp.4.weight', 'head.mlp.4.bias', 'head.last_layer.weight_g', 'head.last_layer.weight_v']
            teacher_head_state = {key[21:]: value for key, value in teacher_state.items() if 'head.projection_head' in key}
            self.teacher_head.mlp.load_state_dict(teacher_head_state, strict=True)
            student_state = state_dict['student']
            student_head_state = {key[21:]: value for key, value in student_state.items() if 'head.projection_head' in key}
            self.student_head.mlp.load_state_dict(student_head_state, strict=True)

        self.teacher_head.last_layer.load_state_dict(self.student_head.last_layer.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.local_crops_number = model_configs['local_crop_number']
        self.num_epochs = configs['optim']['epochs']
        self.dino_loss = DINOLoss(
            out_dim=self.num_classes,
            ncrops=self.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            warmup_teacher_temp=model_configs['warmup_teacher_temp'],
            teacher_temp=model_configs['teacher_temp'],
            warmup_teacher_temp_epochs=model_configs['warmup_teacher_temp_epochs'],
            nepochs=self.num_epochs,
            student_temp = 0.1,
            center_momentum = model_configs['center_momentum']
        ).cuda() 

        self.merge_thresholds = model_configs['merge_thresholds']
        self.merge_minsize = model_configs['merge_minsize']
        self.nq_temperature = model_configs['nq_temperature']

        self.init_graph_utils(batch_size=batch_size)
        self.evaluate_key = model_configs['evaluate_key']
        self.num_queries = self.num_classes

    
    def init_graph_utils(self, batch_size):
        H = W = (224 // self.patch_size) 
        global_edge_index = []
        num_global_nodes = H*W
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    global_edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    global_edge_index.append([cluster_idx-W, cluster_idx])
                cluster_idx += 1
        global_edge_index = torch.tensor(global_edge_index).permute(1, 0).contiguous().long() # 2 N

        H = W = (96 // self.patch_size)
        num_local_nodes = H*W 
        local_edge_index = []
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    local_edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    local_edge_index.append([cluster_idx-W, cluster_idx])
                cluster_idx += 1
        local_edge_index = torch.tensor(local_edge_index).permute(1, 0).contiguous().long()

        from torch_geometric.data import Batch, Data
        global_graph = Data(edge_index=global_edge_index,)
        global_graph.num_nodes = num_global_nodes
        local_graph = Data(edge_index=local_edge_index)
        local_graph.num_nodes = num_local_nodes

        # list[b hi wi c], crop=2_global + 10_local
        whole_graph = []
        for _ in range(2):
            whole_global = [global_graph.clone() for _ in range(batch_size)]
            whole_graph.extend(whole_global)
        for _ in range(self.local_crops_number):
            whole_local = [local_graph.clone() for _ in range(batch_size)]
            whole_graph.extend(whole_local) 

        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in whole_graph]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in whole_graph]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        whole_graph = Batch.from_data_list(whole_graph)

        # self.nodes_batch_ids: Tensor = None
        # self.edges_batch_ids: Tensor = None
        # self.whole_graph_edge_index: Tensor = None
        # self.node_num_patches: Tensor = None
        self.register_buffer('nodes_batch_ids', torch.tensor(nodes_batch_ids).int())
        self.register_buffer('edges_batch_ids', torch.tensor(edges_batch_ids).int())
        self.register_buffer('whole_graph_edge_index', whole_graph.edge_index.long())

        self.register_buffer('node_num_patches', torch.ones(len(nodes_batch_ids)).int())

    def optimize_setup(self, configs, niter_per_ep):
        optim_configs = configs['optim']
        model_configs = configs['model']

        params_groups = utils.get_params_groups(self.student_head)
        self.optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        self.fp16_scaler = None
        if optim_configs['use_fp16']:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
        # # warmup阶段从0到base value, 到epochs之前: base余弦降到min_lr
        self.lr_schedule = utils.cosine_scheduler(
            base_value= optim_configs['lr'] * optim_configs['batch_size'] / 256.,  # batch_size越大，lr也应该越大
            final_value=optim_configs['min_lr'],
            epochs= optim_configs['epochs'], 
            niter_per_ep=niter_per_ep,
            warmup_epochs=optim_configs['warmup_epochs'],
            start_warmup_value=0,
        )

        self.wd_schedule = utils.cosine_scheduler( # warmup阶段从0到base value, 到epochs之前: base余弦降到min_lr
            base_value=optim_configs['weight_decay'],
            final_value=optim_configs['weight_decay_end'],
            epochs= optim_configs['epochs'],
            niter_per_ep=niter_per_ep,
            warmup_epochs = 0,
            start_warmup_value = 0
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = utils.cosine_scheduler(base_value= model_configs['momentum_teacher'], 
                                                        final_value=1,
                                                        epochs=optim_configs['epochs'],
                                                         niter_per_ep=niter_per_ep,
                                                        warmup_epochs = 0,
                                                        start_warmup_value = 0)

        self.optim_configs = optim_configs

    @torch.no_grad()
    def forward_backbone_features(self, x):
        # list[b 3 h w], crop
        self.backbone.eval()
        x = [torchv_Normalize(foo, self.pixel_mean, self.pixel_std, False) for foo in x]

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)

        out_features = [] # list[b hi wi c]
        for end_idx in idx_crops:
            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                _out = self.backbone(torch.cat(x[start_idx: end_idx])) # 2B 3 h w
            B, _, H, W = x[start_idx].shape
            if self.feat_type == 'feat':
                features = _out['features'][0] # b cls_hw c
                features = features[:, 1:, :].reshape(B*(end_idx-start_idx), H//self.patch_size, W//self.patch_size, -1)

            elif self.feat_type == 'key':
                features = _out['qkvs'][0] # 3 b head cls_hw head_dim
                features = features[1, :, :, 1:, :] # b head hw head_di
                features: torch.Tensor = rearrange(features, 'b head (h w) d -> b h w (head d)',h=H//self.patch_size, w=W//self.patch_size)
            else:
                raise ValueError()
            out_features.extend(features.chunk(end_idx-start_idx, dim=0))
            start_idx = end_idx       

        graph_node_features = torch.cat([foo.flatten(0, 2) for foo in out_features], dim=0) # list[bhw c], crop -> N c
        # list[list[ni c], threshold] crop_batch
        cluster_feats = aggo_whole_batch(nodes_feature=graph_node_features,
                                         edge_index=self.whole_graph_edge_index,
                                         node_batch_tensor=self.nodes_batch_ids,
                                         edge_batch_tensor=self.edges_batch_ids,
                                         node_num_patches=self.node_num_patches,) 
        # lenghts = [len(foo) for foo in cluster_feats[0]]
        cluster_feats = [torch.cat(foo, dim=0) for foo in cluster_feats] # list[ni c], crop_batch
        # list[b hi wi c]
        return out_features, cluster_feats  #, lenghts


    def forward_backward(self, batch_dict):
        current_epoch = batch_dict['num_epoch']
        device = self.device
        assert self.training
        trainingiter = batch_dict['num_iterations']
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[trainingiter]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[trainingiter]

        num_crops = self.local_crops_number + 2
        images = batch_dict['images']  # list[b 3 hi wi], crop
        images = [foo.to(device, non_blocking=True) for foo in images]
        batch_size = images[0].shape[0] 
        common_samples = batch_dict['common_samples'] 
        # list[b hi wi c], crop
        # list[ni c], crop_batch
        image_features, cluster_feats = self.forward_backbone_features(images)

        teacher_crops = []  # list[b k hi wi], global
        student_crops = [] # list[b k hi wi], global_local
        teacher_nqs = [] # list[list[ni c], batch], global
        with torch.cuda.amp.autocast(self.fp16_scaler is not None):
            for crop_idx in range(num_crops):
                if crop_idx < 2:
                    hw_Ks_teacher = []
                    teacher_nq_feats = []
                hw_Ks_student = []
                for batch_idx in range(batch_size):
                    nq_feats = cluster_feats[crop_idx * batch_size + batch_idx]
                    hw_feats = image_features[crop_idx][batch_idx] # h w c
                    with torch.no_grad():
                        cluster_attn = torch.einsum('hwc,nc->hwn', 
                                                    F.normalize(hw_feats, dim=-1, eps=1e-10), F.normalize(nq_feats, dim=-1, eps=1e-10)) # [-1, 1]
                        cluster_attn = F.softmax(cluster_attn / self.nq_temperature, dim=-1)
                    # region
                    # visualize_cutler_onlyAttn(image=images[crop_idx][batch_idx],
                    #                           cluster_attn=cluster_attn,
                    #                           patch_size=self.patch_size,
                    #                           lengths=lenghts,
                    #                           image_path='./',
                    #                           image_name='test.png')
                    # pass
                    # endregion
                    if crop_idx < 2:  
                        teacher_nq = self.teacher_head(nq_feats)
                        teacher_nq_feats.append(teacher_nq) 
                        hw_Ks_teacher.append(torch.einsum('hwn,nk->khw', cluster_attn, teacher_nq))
                    student_nq = self.student_head(nq_feats)
                    hw_Ks_student.append(torch.einsum('hwn,nk->khw', cluster_attn, student_nq))

                if crop_idx < 2:
                    teacher_crops.append(torch.stack(hw_Ks_teacher, dim=0)) # b k h w
                    teacher_nqs.append(teacher_nq_feats)
                student_crops.append(torch.stack(hw_Ks_student, dim=0))
    
            loss = self.dino_loss(student_output=student_crops, 
                                  teacher_output=teacher_crops, 
                                  epoch=current_epoch,
                                  common_samples=common_samples,
                                  teacher_nqs=teacher_nqs)
   
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        self.optimizer.zero_grad()
        param_norms = None
        if self.fp16_scaler is None:
            loss.backward()
            if self.optim_configs['clip_grad']:
                param_norms = utils.clip_gradients(self.student_head, self.optim_configs['clip_grad'])
            utils.cancel_gradients_last_layer(current_epoch, self.student_head, self.optim_configs['freeze_last_layer'])
            self.optimizer.step()
        else:
            self.fp16_scaler.scale(loss).backward()
            if self.optim_configs['clip_grad']:
                self.fp16_scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(self.student_head, self.optim_configs['clip_grad'])
            utils.cancel_gradients_last_layer(current_epoch, self.student_head, self.optim_configs['freeze_last_layer'])
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[trainingiter]  # momentum parameter
            for param_q, param_k in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)      
                
        torch.cuda.synchronize()
      
        return {'loss': loss.item()}

    @torch.no_grad()
    def forward_backbone_features_sample(self, x):
        # b 3 h w
        self.backbone.eval()
        x = torchv_Normalize(x, self.pixel_mean, self.pixel_std, False)
        B, _, H_ori, W_ori = x.shape
        H, W = H_ori//self.patch_size, W_ori//self.patch_size
        _out = self.backbone(x) #
        if self.feat_type == 'feat':
            features = _out['features'][0] # b cls_hw c
            features = features[:, 1:, :].reshape(B, H//self.patch_size, W//self.patch_size, -1)

        elif self.feat_type == 'key':
            features = _out['qkvs'][0] # 3 b head cls_hw head_dim
            features = features[1, :, :, 1:, :] # b head hw head_di
            features: torch.Tensor = rearrange(features, 'b head (h w) d -> b h w (head d)',h=H, w=W)
        else:
            raise ValueError()
        # b h w c, bhw c
        graph_node_features = features.flatten(0, 2) 
        edge_index = []
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    edge_index.append([cluster_idx-W, cluster_idx])
                cluster_idx += 1
        edge_index = torch.tensor(edge_index).permute(1, 0).contiguous().long().to(self.device) # 2 E

        from torch_geometric.data import Batch, Data
        graph = Data(edge_index=edge_index,)
        graph.num_nodes = H*W
        whole_graph = [graph.clone() for _ in range(B)]
        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in whole_graph]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in whole_graph]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        whole_graph = Batch.from_data_list(whole_graph)
        whole_graph_edge_index = whole_graph.edge_index

        nodes_batch_ids = torch.tensor(nodes_batch_ids).int().to(self.device)
        edges_batch_ids = torch.tensor(edges_batch_ids).int().to(self.device)
        node_num_patches = torch.ones(len(nodes_batch_ids)).int().to(self.device)

        cluster_feats = aggo_whole_batch(nodes_feature=graph_node_features,
                                         edge_index=whole_graph_edge_index,
                                         node_batch_tensor=nodes_batch_ids,
                                         edge_batch_tensor=edges_batch_ids,
                                         node_num_patches=node_num_patches,) 
        cluster_feats = [torch.cat(foo, dim=0) for foo in cluster_feats] # list[ni c], batch
        # list[b hi wi c]
        return features, cluster_feats 

    @torch.no_grad()
    def sample(self, batch_dict, visualize_all=False):
        assert not self.training
        self.backbone.eval()

        images = batch_dict['images']  # b 3 h w
        _, _, H, W = images.shape
        images = images.to(self.device)
        batch_size = images.shape[0]
        # b h w c
        # list[ni c], batch
        image_features, cluster_feats = self.forward_backbone_features_sample(images)

        outputs = [] # b k h w
        if self.evaluate_key == 'teacher':
            eval_head = self.teacher_head
        else:
            eval_head = self.student_head
        
        for batch_idx in range(batch_size):
            nq_feats = cluster_feats[batch_idx]
            hw_feats = image_features[batch_idx] # h w c
            cluster_attn = torch.einsum('hwc,nc->hwn', 
                                        F.normalize(hw_feats, dim=-1, eps=1e-10), F.normalize(nq_feats, dim=-1, eps=1e-10)) # [-1, 1]
            cluster_attn = F.softmax(cluster_attn / self.nq_temperature, dim=-1)
            nq_k = eval_head(nq_feats)
            outputs.append(torch.einsum('hwn,nk->khw', cluster_attn, nq_k))
           
        outputs = torch.stack(outputs, dim=0) # b k h w, logits
        outputs = F.interpolate(outputs, (H,W), mode='bilinear', align_corners=False)
        # sampled_points, similarities, backbone_similarities = None, None, None
        # kmeans_preds, num_kmeans_classes, kmeans_preds_backbone = None, None, None,
        # if visualize_all:
        #     sampled_points, similarities, backbone_similarities = self.sample_point_similarities(code_features=head_code,
        #                                                                                          backbone_features=image_features, 
        #                                                                                          num_points=10)
        #     kmeans_preds, _ , num_kmeans_classes = self.self_cluster(head_code, label)
        #     kmeans_preds_backbone, _ , _ = self.self_cluster(backbone_feats, label)


        return  {
            'cluster_preds': outputs,
            # 'sampled_points': sampled_points,
            # 'similarities': similarities,
            # 'backbone_similarities': backbone_similarities,

            # 'kmeans_preds': kmeans_preds,
            # 'kmeans_preds_bb': kmeans_preds_backbone,
            # 'num_kmeans_classes': num_kmeans_classes,
        }

@register_model
def agg_sample(configs, device):
    from data_schedule.unsupervised_image_semantic_seg.mapper import AggSampleAUXMapper
    aux_mapper = AggSampleAUXMapper()
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    model = AggSample(configs, num_classes=num_classes, batch_size=configs['optim']['batch_size'])
    model.to(device)
    model.optimize_setup(configs, niter_per_ep=len(train_loader))
    
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function


def visualize_cutler(image, cluster_masks,  cluster_attn, patch_size):
    # nq h w, nq h w
    from detectron2.data import  MetadataCatalog
    MetadataCatalog.get('single').set(stuff_classes = ['0'],
                    stuff_colors = [(156, 31, 23),])
    cluster_masks = cluster_masks.cpu()
    cluster_attn = cluster_attn.cpu().permute(2, 0, 1)
    cluster_masks = F.interpolate(cluster_masks[None, ...].float(), scale_factor=patch_size, mode='nearest')[0] > 0.5
    cluster_attn = F.interpolate(cluster_attn[None, ...], scale_factor=patch_size, mode='bilinear')[0]
    image = image.cpu()
    from data_schedule.unsupervised_image_semantic_seg.evaluator_alignseg import generate_semseg_canvas_uou
    import cv2
    H, W = image.shape[1:]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')
    
    gt_plots = []
    pred_plots = []
    
    gt = cluster_masks.int() - 1 # nq h w, -1/0
    gt[gt==-1] = 20000

    for gt_m in gt:
        gt_plots.append(torch.from_numpy(generate_semseg_canvas_uou(image=image, H=H, W=W, mask=gt_m, 
                                                                    num_classes=1, dataset_name='single',)) )
    for sim in cluster_attn:
        heatmap = ((sim + 1) / 2).clamp(0, 1).numpy()
        heatmap = np.uint8(255 * heatmap) 
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.addWeighted(heatmap, 0.7, image, 0.3, 0)
        pred_plots.append(torch.from_numpy(np.asarray(superimposed_img)))

    gt_plots = torch.cat(gt_plots, dim=1)
    pred_plots = torch.cat(pred_plots, dim=1)
    whole_image = torch.cat([gt_plots, pred_plots], dim=0)
    from PIL import Image
    Image.fromarray(whole_image.numpy()).save('./test.png')

def visualize_cutler_onlyAttn(image, cluster_attn, patch_size, lengths, image_path, image_name):
    # nq h w, nq h w
    from detectron2.data import  MetadataCatalog
    MetadataCatalog.get('single').set(stuff_classes = ['0'],
                    stuff_colors = [(156, 31, 23),])
    cluster_attn = cluster_attn.cpu().permute(2, 0, 1)
    cluster_attn = F.interpolate(cluster_attn[None, ...], scale_factor=patch_size, mode='bilinear')[0]
    image = image.cpu()
    from data_schedule.unsupervised_image_semantic_seg.evaluator_alignseg import generate_semseg_canvas_uou
    import cv2
    H, W = image.shape[1:]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')

    pred_plots = []
    for sim in cluster_attn:
        heatmap = ((sim + 1) / 2).clamp(0, 1).numpy()
        heatmap = np.uint8(255 * heatmap) 
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.addWeighted(heatmap, 0.7, image, 0.3, 0)
        pred_plots.append(torch.from_numpy(np.asarray(superimposed_img)))
    max_len = max(lengths) * W
    pred_plots = torch.stack(pred_plots, dim=0) # n h w 3
    pred_plots = pred_plots.split(lengths, dim=0) # list[ni h w 3], thre
    pred_plots = [foo.permute(1, 0, 2, 3).flatten(1, 2) for foo in pred_plots] # list[h niw 3]
    pred_plots = [F.pad(foo, [0, 0, 0, max_len-foo.shape[1]]) for foo in pred_plots]
    pred_plots = torch.cat(pred_plots, dim=0) # thre_h niw 3

    from PIL import Image
    if not os.path.exists(image_path):
        os.makedirs(image_path, exist_ok=True)
    Image.fromarray(pred_plots.numpy()).save(os.path.join(image_path, image_name))



 
