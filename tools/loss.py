
import torch.nn as nn

from .metric import *

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1])).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / size[1]
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class SingleMultiFeatureLoss(nn.Module):

    def __init__(self, chunks):
        super(SingleMultiFeatureLoss, self).__init__()
        self.chunks = chunks

    def forward(self, single_features, multi_features):
        single_features = torch.chunk(single_features, chunks=4, dim=0)
        multi_features = torch.chunk(multi_features, chunks=4, dim=0)
        loss = 0.0
        for chunk in range(self.chunks):
            for index in range(4):
                each_loss = torch.norm((multi_features[chunk][index] - single_features[chunk][0]), p=2) + \
                       torch.norm((multi_features[chunk][index] - single_features[chunk][1]), p=2) + \
                       torch.norm((multi_features[chunk][index] - single_features[chunk][2]), p=2) + \
                       torch.norm((multi_features[chunk][index] - single_features[chunk][3]), p=2)
                loss += each_loss
        loss = loss/16.0
        return loss

class SourceIdentityDomainLoss(nn.Module):
    
    def __init__(self, theta, use_gpu=True):
        super(SourceIdentityDomainLoss, self).__init__()
        self.theta = theta
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1] - 1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1) * self.theta
        targets_domain = torch.ones((size[0], 1)) * (1 - self.theta)
        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
            targets_domain = targets_domain.to(torch.device('cuda'))
        targets = torch.cat((targets, targets_domain), dim=1)
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class TargetIdentityDomainLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(TargetIdentityDomainLoss, self).__init__()
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        log_probs = self.logsoftmax(inputs)
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1] - 1))
        targets_domain = torch.ones((size[0], 1))
        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
            targets_domain = targets_domain.to(torch.device('cuda'))
        targets = torch.cat((targets, targets_domain), dim=1)
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class DiscrepencyLoss(nn.Module):
    def __init__(self):
        super(DiscrepencyLoss, self).__init__()

    def __call__(self, pid_cls, pid_domain_cls):
        pid_domain_cls = pid_domain_cls[:, 0: pid_domain_cls.size(1) - 1]
        loss = torch.mean(torch.abs(F.softmax(pid_cls, dim=1) - F.softmax(pid_domain_cls, dim=1)))
        return loss

class CameraClassifierLoss(nn.Module):
    def __init__(self, source_cam, target_cam):
        super(CameraClassifierLoss, self).__init__()
        self.source_cam = source_cam
        self.target_cam = target_cam
        self.entropy = nn.CrossEntropyLoss()

    def forward(self, source_cam_cls, target_cam_cls, source_cid, target_cid):
        target_cid = target_cid + self.source_cam
        cls = torch.cat([source_cam_cls, target_cam_cls], dim=0)
        real_label = torch.cat([source_cid, target_cid], dim=0).long()
        real_loss = self.entropy(cls, real_label)
        loss = real_loss
        return loss

class CameraFeatureExtractorLoss(nn.Module):
    def __init__(self, source_cam, target_cam):
        super(CameraFeatureExtractorLoss, self).__init__()
        self.source_cam = source_cam
        self.target_cam = target_cam
        self.entropy = nn.CrossEntropyLoss()

    def forward(self, source_cam_cls, target_cam_cls):
        cls = torch.cat([source_cam_cls, target_cam_cls], dim=0)
        label = torch.tensor([self.source_cam + self.target_cam] * cls.size(0)).long().cuda()
        loss = self.entropy(cls, label)
        return loss








