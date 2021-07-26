import os
import torch
import torch.nn as nn
import torch.optim as optim

from bisect import bisect_right

from network import Res50BNNeck, Res50IBNaBNNeck, IdentityClassifier, IdentityDomainClassifier, CameraClassifier
from tools import CrossEntropyLabelSmooth, SingleMultiFeatureLoss, SourceIdentityDomainLoss,\
    TargetIdentityDomainLoss, DiscrepencyLoss, CameraClassifierLoss, CameraFeatureExtractorLoss, os_walk

class Base:

    def __init__(self, config):
        self.config = config
        self.cnnbackbone = config.cnnbackbone
        self.source_pid_num = config.source_pid_num
        self.source_cid_num = config.source_cid_num
        self.target_cid_num = config.target_cid_num
        self.chunks = config.chunks
        self.beta = config.beta

        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.e_learning_rate = config.e_learning_rate
        self.ic_learning_rate = config.ic_learning_rate
        self.idc_learning_rate = config.idc_learning_rate
        self.cc_learning_rate = config.cc_learning_rate
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones

        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):
        if self.cnnbackbone == 'res50':
            self.feature_extractor = Res50BNNeck()
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)
            self.identity_classifier = IdentityClassifier(2048, source_pid_num=self.source_pid_num)
            self.identity_classifier = nn.DataParallel(self.identity_classifier).to(self.device)
            self.identitydomain_classifier = IdentityDomainClassifier(2048, source_pid_num=self.source_pid_num)
            self.identitydomain_classifier = nn.DataParallel(self.identitydomain_classifier).to(self.device)
            self.camera_classifier = CameraClassifier(2048, source_cam=self.source_cid_num,
                                                      target_cam=self.target_cid_num)
            self.camera_classifier = nn.DataParallel(self.camera_classifier).to(self.device)
        elif self.cnnbackbone == 'res50ibna':
            self.feature_extractor = Res50IBNaBNNeck()
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)
            self.identity_classifier = IdentityClassifier(2048, source_pid_num=self.source_pid_num)
            self.identity_classifier = nn.DataParallel(self.identity_classifier).to(self.device)
            self.identitydomain_classifier = IdentityDomainClassifier(2048, source_pid_num=self.source_pid_num)
            self.identitydomain_classifier = nn.DataParallel(self.identitydomain_classifier).to(self.device)
            self.camera_classifier = CameraClassifier(2048, source_cam=self.source_cid_num,
                                                      target_cam=self.target_cid_num)
            self.camera_classifier = nn.DataParallel(self.camera_classifier).to(self.device)

    def _init_creiteron(self):
        self.source_pid_creiteron = CrossEntropyLabelSmooth()
        self.source_identity_domain_creiteron = SourceIdentityDomainLoss(self.beta)
        self.target_identity_domain_creiteron = TargetIdentityDomainLoss()
        self.single_multi_creiteron = SingleMultiFeatureLoss(self.chunks)
        self.dispency_creiteron = DiscrepencyLoss()
        self.camera_classifier_creiteron = CameraClassifierLoss(self.source_cid_num, self.target_cid_num)
        self.camera_feature_extractor_creiteron = CameraFeatureExtractorLoss(self.source_cid_num, self.target_cid_num)


    def _init_optimizer(self):
        self.feature_extractor_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=self.e_learning_rate,
                                                      weight_decay=self.weight_decay)
        self.feature_extractor_lr_scheduler = WarmupMultiStepLR(self.feature_extractor_optimizer, self.milestones,
                                                                gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        self.identity_classifier_optimizer = optim.Adam(self.identity_classifier.parameters(),
                                                            lr=self.ic_learning_rate, weight_decay=self.weight_decay)
        self.identity_classifier_lr_scheduler = WarmupMultiStepLR(self.identity_classifier_optimizer,
                                                        self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)
        self.identitydomain_classifier_optimizer = optim.Adam(self.identitydomain_classifier.parameters(),
                                                    lr=self.idc_learning_rate, weight_decay=self.weight_decay)
        self.identitydomain_classifier_lr_scheduler = WarmupMultiStepLR(self.identitydomain_classifier_optimizer,
                                                        self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        self.camera_classifier_optimizer = optim.Adam(self.camera_classifier.parameters(),
                                                        lr=self.cc_learning_rate, weight_decay=self.weight_decay)

    def save_model(self, save_epoch):
        feature_extractor_file_path = os.path.join(self.save_model_path, 'feature_extractor_{}.pkl'.format(save_epoch))
        torch.save(self.feature_extractor.state_dict(), feature_extractor_file_path)
        identity_classifier_file_path = os.path.join(self.save_model_path, 'identity_classifier_{}.pkl'
                                                     .format(save_epoch))
        torch.save(self.identity_classifier.state_dict(), identity_classifier_file_path)
        identitydomain_classifier_file_path = os.path.join(self.save_model_path, 'identitydomain_classifier_{}.pkl'
                                                            .format(save_epoch))
        torch.save(self.identitydomain_classifier.state_dict(), identitydomain_classifier_file_path)
        camera_classifier_file_path = os.path.join(self.save_model_path, 'camera_classifier_{}.pkl'
                                                     .format(save_epoch))
        torch.save(self.camera_classifier.state_dict(), camera_classifier_file_path)
        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            for file in files:
                if '.pkl' not in file:
                    files.remove(file)
            if len(files) > 4 * self.max_save_model_num:
                file_iters = sorted([int(file.replace('.pkl', '').split('_')[2]) for file in files], reverse=False)
                feature_extractor_file_path = os.path.join(root, 'feature_extractor_{}.pkl'.format(file_iters[0]))
                os.remove(feature_extractor_file_path)
                identity_classifier_file_path = os.path.join(root, 'identity_classifier_{}.pkl'.
                                                                 format(file_iters[0]))
                os.remove(identity_classifier_file_path)
                identitydomain_classifier_file_path = os.path.join(root, 'identitydomain_classifier_{}.pkl'.
                                                                 format(file_iters[0]))
                os.remove(identitydomain_classifier_file_path)
                camera_classifier_file_path = os.path.join(root, 'camera_classifier_{}.pkl'.format(file_iters[0]))
                os.remove(camera_classifier_file_path)

    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if '.pkl' not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch):
        feature_extractor_path = os.path.join(self.save_model_path, 'feature_extractor_{}.pkl'.format(resume_epoch))
        self.feature_extractor.load_state_dict(torch.load(feature_extractor_path), strict=False)
        print('Successfully resume feature_extractor from {}'.format(feature_extractor_path))
        identity_classifier_path = os.path.join(self.save_model_path, 'identity_classifier_{}.pkl'.
                                                    format(resume_epoch))
        self.identity_classifier.load_state_dict(torch.load(identity_classifier_path), strict=False)
        print('Successfully resume identity_classifier from {}'.format(identity_classifier_path))
        identitydomain_classifier_path = os.path.join(self.save_model_path, 'identitydomain_classifier_{}.pkl'.
                                                    format(resume_epoch))
        self.identitydomain_classifier.load_state_dict(torch.load(identitydomain_classifier_path), strict=False)
        print('Successfully resume identitydomain_classifier from {}'.format(identitydomain_classifier_path))
        camera_classifier_path = os.path.join(self.save_model_path, 'camera_classifier_{}.pkl'.format(resume_epoch))
        self.camera_classifier.load_state_dict(torch.load(camera_classifier_path), strict=False)
        print('Successfully resume camera_classifier from {}'.format(camera_classifier_path))

    def set_train(self):
        self.feature_extractor = self.feature_extractor.train()
        self.identity_classifier = self.identity_classifier.train()
        self.identitydomain_classifier = self.identitydomain_classifier.train()
        self.camera_classifier = self.camera_classifier.train()
        self.training = True

    def set_eval(self):
        self.feature_extractor = self.feature_extractor.eval()
        self.identity_classifier = self.identity_classifier.eval()
        self.identitydomain_classifier = self.identitydomain_classifier.eval()
        self.camera_classifier = self.camera_classifier.eval()
        self.training = False


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
        ]