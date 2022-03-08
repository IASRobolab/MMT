from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a


__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a', 'wacv_model_ibn']


class wacv_model_ibn(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }
    def __init__(self):
        super(wacv_model_ibn, self).__init__()
        # num_classes = 767  # cuhk03
        # num_classes = 702  # duke
        #num_classes = 751  # market
        num_classes = 1041  # msmt
        feats = 256
        #########################resnet50################################

        resnet = wacv_model_ibn.__factory[50](pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.AdaptiveMaxPool2d((1, 1))  # nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_h2 = nn.AdaptiveMaxPool2d((2, 1))
        self.maxpool_h3 = nn.AdaptiveMaxPool2d((3, 1))

        self.reduction_p1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # p1
        self.reduction_p2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # p2
        self.reduction_p2_s1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s2
        self.reduction_p2_s2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s2
        self.reduction_p2_c1 = nn.Sequential(nn.Conv2d(1024, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c2
        self.reduction_p2_c2 = nn.Sequential(nn.Conv2d(1024, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c2
        self.reduction_p3 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # p3
        self.reduction_p3_c1 = nn.Sequential(nn.Conv2d(683, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c3
        self.reduction_p3_c2 = nn.Sequential(nn.Conv2d(683, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c3
        self.reduction_p3_c3 = nn.Sequential(nn.Conv2d(682, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c3
        self.reduction_p3_s1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s3
        self.reduction_p3_s2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s3
        self.reduction_p3_s3 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s3

        self._init_reduction(self.reduction_p1)  # p1
        self._init_reduction(self.reduction_p2)  # p2
        self._init_reduction(self.reduction_p2_s1)
        self._init_reduction(self.reduction_p2_s2)
        self._init_reduction(self.reduction_p2_c1)
        self._init_reduction(self.reduction_p2_c2)
        self._init_reduction(self.reduction_p3)  # p3
        self._init_reduction(self.reduction_p3_s1)
        self._init_reduction(self.reduction_p3_s2)
        self._init_reduction(self.reduction_p3_s3)
        self._init_reduction(self.reduction_p3_c1)
        self._init_reduction(self.reduction_p3_c2)
        self._init_reduction(self.reduction_p3_c3)

        self.fc_id_p1 = nn.Linear(feats, num_classes)

        self.fc_id_p2 = nn.Linear(feats, num_classes)
        self.fc_id_p2_s1 = nn.Linear(feats, num_classes)
        self.fc_id_p2_s2 = nn.Linear(feats, num_classes)
        self.fc_id_p2_c1 = nn.Linear(feats, num_classes)
        self.fc_id_p2_c2 = nn.Linear(feats, num_classes)

        self.fc_id_p3 = nn.Linear(feats, num_classes)
        self.fc_id_p3_s1 = nn.Linear(feats, num_classes)
        self.fc_id_p3_s2 = nn.Linear(feats, num_classes)
        self.fc_id_p3_s3 = nn.Linear(feats, num_classes)
        self.fc_id_p3_c1 = nn.Linear(feats, num_classes)
        self.fc_id_p3_c2 = nn.Linear(feats, num_classes)
        self.fc_id_p3_c3 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_p1)

        self._init_fc(self.fc_id_p2)
        self._init_fc(self.fc_id_p2_s1)
        self._init_fc(self.fc_id_p2_s2)
        self._init_fc(self.fc_id_p2_c1)
        self._init_fc(self.fc_id_p2_c2)

        self._init_fc(self.fc_id_p3)
        self._init_fc(self.fc_id_p3_s1)
        self._init_fc(self.fc_id_p3_s2)
        self._init_fc(self.fc_id_p3_s3)
        self._init_fc(self.fc_id_p3_c1)
        self._init_fc(self.fc_id_p3_c2)
        self._init_fc(self.fc_id_p3_c3)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)

        zg_p2 = self.maxpool_zg_p1(p2)
        zp2 = self.maxpool_h2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]
        z_c0_p2 = zg_p2[:, :1024, :, :]
        z_c1_p2 = zg_p2[:, 1024:2048, :, :]

        zg_p3 = self.maxpool_zg_p1(p3)
        zp3 = self.maxpool_h3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        z_c0_p3 = zg_p3[:, :683, :, :]
        z_c1_p3 = zg_p3[:, 683:683 * 2, :, :]
        z_c2_p3 = zg_p3[:, 683 * 2:2048, :, :]

        f_p1 = self.reduction_p1(zg_p1).squeeze(dim=3).squeeze(dim=2)

        f_p2 = self.reduction_p2(zg_p2).squeeze(dim=3).squeeze(dim=2)
        f_p2_c1 = self.reduction_p2_c1(z_c0_p2).squeeze(dim=3).squeeze(dim=2)
        f_p2_c2 = self.reduction_p2_c2(z_c1_p2).squeeze(dim=3).squeeze(dim=2)
        f_p2_s1 = self.reduction_p2_s1(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f_p2_s2 = self.reduction_p2_s2(z1_p2).squeeze(dim=3).squeeze(dim=2)

        f_p3 = self.reduction_p3(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_c1 = self.reduction_p3_c1(z_c0_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_c2 = self.reduction_p3_c2(z_c1_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_c3 = self.reduction_p3_c3(z_c2_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_s1 = self.reduction_p3_s1(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_s2 = self.reduction_p3_s2(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_s3 = self.reduction_p3_s3(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_p1(f_p1)

        l_p2 = self.fc_id_p2(f_p2)
        l_p2_c1 = self.fc_id_p2_c1(f_p2_c1)
        l_p2_c2 = self.fc_id_p2_c2(f_p2_c2)
        l_p2_s1 = self.fc_id_p2_s1(f_p2_s1)
        l_p2_s2 = self.fc_id_p2_s2(f_p2_s2)

        l_p3 = self.fc_id_p3(f_p3)
        l_p3_c1 = self.fc_id_p3_c1(f_p3_c1)
        l_p3_c2 = self.fc_id_p3_c2(f_p3_c2)
        l_p3_c3 = self.fc_id_p3_c3(f_p3_c3)
        l_p3_s1 = self.fc_id_p3_s1(f_p3_s1)
        l_p3_s2 = self.fc_id_p3_s2(f_p3_s2)
        l_p3_s3 = self.fc_id_p3_s3(f_p3_s3)


        predict = torch.cat([
            f_p1, f_p2, f_p3,
            f_p2_c1, f_p2_c2, f_p2_s1, f_p2_s2,
            f_p3_c1, f_p3_c2, f_p3_c3, f_p3_s1, f_p3_s2, f_p3_s3,
        ], dim=1)

        return predict, \
               f_p1, f_p2, f_p3, \
               l_p1, l_p2, l_p3, l_p2_c1, l_p2_c2, l_p2_s1, l_p2_s2, l_p3_c1, l_p3_c2, l_p3_c3, l_p3_s1, l_p3_s2, l_p3_s3



class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:
            return bn_x, prob
        return x, prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNetIBN.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.relu.state_dict())
        self.base[3].load_state_dict(resnet.maxpool.state_dict())
        self.base[4].load_state_dict(resnet.layer1.state_dict())
        self.base[5].load_state_dict(resnet.layer2.state_dict())
        self.base[6].load_state_dict(resnet.layer3.state_dict())
        self.base[7].load_state_dict(resnet.layer4.state_dict())


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)
