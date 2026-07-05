import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import backbone, bird_view, range_view
from networks.backbone import get_module
import deep_point

from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax
from utils.Lovasz_Softmax import Lovasz_softmax
from utils.boundary_loss import BoundaryLoss
import yaml
import copy


class PointFeatureProjector(nn.Module):
    def __init__(self, output_size, scale_rate=(1.0, 1.0)):
        super(PointFeatureProjector, self).__init__()
        self.output_size = output_size
        self.scale_rate = scale_rate

    def forward(self, point_feat, point_index):
        return VoxelMaxPool(
            pcds_feat=point_feat,
            pcds_ind=point_index,
            output_size=self.output_size,
            scale_rate=self.scale_rate,
        )


class FeatureEnhancement(nn.Module):
    def __init__(self, grid2point):
        super(FeatureEnhancement, self).__init__()
        self.grid2point = grid2point

    def forward(self, grid_feat, point_coord):
        return self.grid2point(grid_feat, point_coord)


class FusionHead(nn.Module):
    def __init__(self, fusion_layer, pred_layer):
        super(FusionHead, self).__init__()
        self.fusion_layer = fusion_layer
        self.pred_layer = pred_layer

    def forward(self, point_feat, point_bev_feat, point_rv_feat):
        fused_feat = self.fusion_layer(point_feat, point_bev_feat, point_rv_feat)
        pred_cls = self.pred_layer(fused_feat).float()
        return fused_feat, pred_cls


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size,
                                         scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat


class MFINet(nn.Module):
    def __init__(self, pModel):
        super(MFINet, self).__init__()
        self.pModel = pModel

        self.bev_shape = list(pModel.Voxel.bev_shape)
        self.rv_shape = list(pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.dx = (pModel.Voxel.range_x[1] - pModel.Voxel.range_x[0]) / (pModel.Voxel.bev_shape[0])
        self.dy = (pModel.Voxel.range_y[1] - pModel.Voxel.range_y[0]) / (pModel.Voxel.bev_shape[1])
        self.dz = (pModel.Voxel.range_z[1] - pModel.Voxel.range_z[0]) / (pModel.Voxel.bev_shape[2])

        self.point_feat_out_channels = pModel.point_feat_out_channels

        self.build_network()
        self.build_loss()

    def build_loss(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion_seg_cate = None
        print("Loss mode: {}".format(self.pModel.loss_mode))
        if self.pModel.loss_mode == 'ce':
            self.criterion_seg_cate = nn.CrossEntropyLoss(ignore_index=0)
        elif self.pModel.loss_mode == 'ohem':
            self.criterion_seg_cate = CE_OHEM(top_ratio=0.2, top_weight=4.0, ignore_index=0)

            content = torch.zeros(self.pModel.class_num, dtype=torch.float32)
            with open('datasets/semantic-kitti.yaml', 'r') as f:
                # task_cfg = yaml.load(f)
                task_cfg = yaml.safe_load(f)
                for cl, freq in task_cfg["content"].items():
                    x_cl = task_cfg['learning_map'][cl]
                    content[x_cl] += freq

            loss_w = 1 / (content + 0.001)
            loss_w[0] = 0

            print("Loss weights from content: ", loss_w)
            # self.criterion = nn.NLLLoss(loss_w).to(self.device)
            self.criterion = nn.NLLLoss(weight=loss_w).to(self.device)
            self.ls = Lovasz_softmax(ignore=0).to(self.device)
            self.bd = BoundaryLoss().to(self.device)

        elif self.pModel.loss_mode == 'wce':
            content = torch.zeros(self.pModel.class_num, dtype=torch.float32)
            with open('datasets/semantic-kitti.yaml', 'r') as f:
                # task_cfg = yaml.load(f)
                task_cfg = yaml.safe_load(f)
                for cl, freq in task_cfg["content"].items():
                    x_cl = task_cfg['learning_map'][cl]
                    content[x_cl] += freq

            loss_w = 1 / (content + 0.001)
            loss_w[0] = 0

            print("Loss weights from content: ", loss_w)
            self.criterion_seg_cate = nn.CrossEntropyLoss(weight=loss_w)
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')

    def build_network(self):
        # build network
        bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
        bev_layers = copy.deepcopy(self.pModel.BEVParam.layers)
        bev_base_block = self.pModel.BEVParam.base_block
        bev_grid2point = self.pModel.BEVParam.bev_grid2point

        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        rv_layers = copy.deepcopy(self.pModel.RVParam.layers)
        rv_base_block = self.pModel.RVParam.base_block
        rv_grid2point = self.pModel.RVParam.rv_grid2point

        fusion_mode = self.pModel.fusion_mode
        fusion_way = self.pModel.fusion_way

        bev_context_layer[0] = rv_context_layer[0]

        # network
        self.point_encoder = backbone.PointNetStacker(7, rv_context_layer[0], pre_bn=True, stack_num=2)
        self.bev_projector = PointFeatureProjector(self.bev_wl_shape, scale_rate=(1.0, 1.0))
        self.rv_projector = PointFeatureProjector(self.rv_shape, scale_rate=(1.0, 1.0))
        self.bev_net = bird_view.BEVNet(bev_base_block, bev_context_layer, bev_layers, use_att=True)
        self.rv_net = range_view.RVNet(rv_base_block, rv_context_layer, rv_layers, use_att=True)
        self.bev_feature_enhancement = FeatureEnhancement(get_module(bev_grid2point, in_dim=self.bev_net.out_channels))
        self.rv_feature_enhancement = FeatureEnhancement(get_module(rv_grid2point, in_dim=self.rv_net.out_channels))

        point_fusion_channels = (rv_context_layer[0], self.bev_net.out_channels, self.rv_net.out_channels)
        fusion_layer = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels,
                                                               out_channel=self.point_feat_out_channels,
                                                               way=fusion_way)
        pred_layer = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)
        self.fusion_head = FusionHead(fusion_layer, pred_layer)

        self.semantic_output = nn.Conv2d(64, 20, 1)

        self.semantic_output1 = nn.Conv2d(96, 20, 1)
        self.semantic_output0 = nn.Conv2d(64, 20, 1)






    def stage_forward(self, point_feat, pcds_coord, pcds_sphere_coord, pcds_target):
        BS, C, N, _ = point_feat.shape
        pcds_cood_cur = pcds_coord[:, :, :2].contiguous()
        pcds_sphere_coord_cur = pcds_sphere_coord.contiguous()

        # pcds_target = pcds_target.cpu().numpy()
        pcds_target = pcds_target.unsqueeze(1)
        bev_input_target = VoxelMaxPool(pcds_feat=pcds_target, pcds_ind=pcds_coord[:, :, :2].contiguous(),
                                        output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0))  # (BS, C, H, W)
        bev_input_target = bev_input_target.squeeze(1)
        # bev_input_target_cpu = bev_input_target.cpu().numpy()

        rv_input_traget = VoxelMaxPool(pcds_feat=pcds_target, pcds_ind=pcds_sphere_coord_cur, output_size=self.rv_shape,
                                       scale_rate=(1.0, 1.0))
        rv_input_traget = rv_input_traget.squeeze(1)
        # rv_input_target_cpu = rv_input_traget.cpu().numpy()

        point_feat = self.point_encoder(point_feat)
        bev_input = self.bev_projector(point_feat, pcds_coord[:, :, :2].contiguous())
        bev_feat, bev_merge1_up, bev_merge0_up = self.bev_net(bev_input)
        point_bev_feat = self.bev_feature_enhancement(bev_feat, pcds_cood_cur)

        rv_input = self.rv_projector(point_feat, pcds_sphere_coord_cur)
        rv_feat, rv_merge1_up, rv_merge0_up = self.rv_net(rv_input)
        point_rv_feat = self.rv_feature_enhancement(rv_feat, pcds_sphere_coord_cur)

        point_feat_out, pred_cls = self.fusion_head(point_feat, point_bev_feat, point_rv_feat)
        bev_pred_cls = self.semantic_output(bev_feat)
        bev_pred_cls = F.softmax(bev_pred_cls, dim=1)
        rv_pred_cls = self.semantic_output(rv_feat)
        rv_pred_cls = F.softmax(rv_pred_cls, dim=1)

        #pred_aud
        bev_merge1_up_cls = self.semantic_output1(bev_merge1_up)#
        bev_merge1_up_cls = F.softmax(bev_merge1_up_cls, dim=1)
        bev_merge0_up_cls = self.semantic_output0(bev_merge0_up)#
        bev_merge0_up_cls = F.softmax(bev_merge0_up_cls, dim=1)

        rv_merge1_up_cls = self.semantic_output1(rv_merge1_up)#
        rv_merge1_up_cls = F.softmax(rv_merge1_up_cls, dim=1)
        rv_merge0_up_cls = self.semantic_output0(rv_merge0_up)#
        rv_merge0_up_cls = F.softmax(rv_merge0_up_cls, dim=1)

        return pred_cls, bev_pred_cls, rv_pred_cls, bev_input_target, rv_input_traget, bev_merge1_up_cls, bev_merge0_up_cls, rv_merge1_up_cls, rv_merge0_up_cls

    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target):
        pred_cls, bev_pred_cls, rv_pred_cls, bev_input_target, rv_input_traget, bev_merge1_up_cls, bev_merge0_up_cls, rv_merge1_up_cls, rv_merge0_up_cls = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target)
        loss_3D = self.criterion_seg_cate(pred_cls, pcds_target) + 2 * lovasz_softmax(pred_cls, pcds_target, ignore=0)
        lossbev_2D = self.criterion(torch.log(bev_pred_cls.clamp(min=1e-8)), bev_input_target) + 1.5 * self.ls(bev_pred_cls, bev_input_target.long())
        lossbev_2D_1 = self.criterion(torch.log(bev_merge1_up_cls.clamp(min=1e-8)), bev_input_target) + 1.5 * self.ls(bev_merge1_up_cls, bev_input_target.long())
        lossbev_2D_0 = self.criterion(torch.log(bev_merge0_up_cls.clamp(min=1e-8)), bev_input_target) + 1.5 * self.ls(bev_merge0_up_cls, bev_input_target.long())
        lossrv_2D = self.criterion(torch.log(rv_pred_cls.clamp(min=1e-8)), rv_input_traget) + 1.5 * self.ls(rv_pred_cls, rv_input_traget.long())
        lossrv_2D_1 = self.criterion(torch.log(rv_merge1_up_cls.clamp(min=1e-8)), rv_input_traget) + 1.5 * self.ls(rv_merge1_up_cls, rv_input_traget.long())
        lossrv_2D_0 = self.criterion(torch.log(rv_merge0_up_cls.clamp(min=1e-8)), rv_input_traget) + 1.5 * self.ls(rv_merge0_up_cls, rv_input_traget.long())
        bdlosssbev = self.bd(bev_pred_cls, bev_input_target.long()) + self.bd(bev_merge1_up_cls, bev_input_target.long()) + self.bd(bev_merge0_up_cls, bev_input_target.long())
        bdlosssrv = self.bd(rv_pred_cls, rv_input_traget.long()) + self.bd(rv_merge1_up_cls, rv_input_traget.long()) + self.bd(rv_merge0_up_cls, rv_input_traget.long())

        # loss_2D = self.criterion(torch.log(bev_pred_cls.clamp(min=1e-8)), bev_input_target) + 1.5 * self.ls(bev_pred_cls, bev_input_target.long())
        loss = loss_3D + lossbev_2D + lossrv_2D + bdlosssbev + bdlosssrv + lossbev_2D_1 + lossbev_2D_0 + lossrv_2D_1 + lossrv_2D_0

        return loss

    def infer(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target):
        pred_cls = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target)
        return pred_cls
