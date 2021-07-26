
import torch
from collections import OrderedDict
import numpy as np
from network import MultiViewReasoning
from tools import MultiItemAverageMeter, accuracy

def train_stage1(config, base, loaders):

    base.set_train()
    source_loader = loaders.source_loader
    meter = MultiItemAverageMeter()
    for i in range(1032):
        source_imgs, source_pids, source_cids = source_loader.next_one()
        source_imgs, source_pids, source_cids = source_imgs.to(base.device), source_pids.to(base.device), \
                                                source_cids.to(base.device)

        source_features = base.feature_extractor(source_imgs)
        pid_cls_score = base.identity_classifier(source_features)
        pid_domain_cls_score = base.identitydomain_classifier(source_features)
        pid_loss = base.source_pid_creiteron(pid_cls_score, source_pids)
        pid_domain_loss = base.source_identity_domain_creiteron(pid_domain_cls_score, source_pids)

        loss = pid_loss + pid_domain_loss

        base.feature_extractor_optimizer.zero_grad()
        base.identity_classifier_optimizer.zero_grad()
        base.identitydomain_classifier_optimizer.zero_grad()

        loss.backward()

        base.feature_extractor_optimizer.step()
        base.identity_classifier_optimizer.step()
        base.identitydomain_classifier_optimizer.step()

        meter.update({'pid_loss': pid_loss.data, 'pid_domain_loss': pid_domain_loss.data})


    return meter.get_val(), meter.get_str()

def train_stage2(config, base, loaders):

    base.set_train()
    source_loader = loaders.source_loader
    target_loader = loaders.target_loader
    meter = MultiItemAverageMeter()
    for i in range(808):
        source_imgs, source_pids, source_cids = source_loader.next_one()
        source_imgs, source_pids, source_cids = source_imgs.to(base.device), source_pids.to(base.device), \
                                                source_cids.to(base.device)
        target_imgs, target_pids, target_cids = target_loader.next_one()
        target_imgs, target_pids, target_cids = target_imgs.to(base.device), target_pids.to(base.device), \
                                                target_cids.to(base.device)

        source_features = base.feature_extractor(source_imgs)
        target_features = base.feature_extractor(target_imgs)
        source_pid_cls_score = base.identity_classifier(source_features)
        source_cid_cls_score = base.camera_classifier(source_features)
        multi_view_features = MultiViewReasoning().__call__(source_features, source_features, source_pids, source_pids,
                                                            source_pid_cls_score)
        multi_view_cls_score = base.identity_classifier(multi_view_features)
        source_pid_domain_cls_score = base.identitydomain_classifier(source_features)
        target_cid_cls_score = base.camera_classifier(target_features)
        target_pid_domain_cls_score = base.identitydomain_classifier(target_features)
        source_pid_loss = base.source_pid_creiteron(source_pid_cls_score, source_pids)
        multi_view_pid_loss = base.source_pid_creiteron(multi_view_cls_score, source_pids)
        source_pid_domain_loss = base.source_identity_domain_creiteron(source_pid_domain_cls_score, source_pids)
        target_pid_domain_loss = base.target_identity_domain_creiteron(target_pid_domain_cls_score)
        source_dispency_loss = base.dispency_creiteron(source_pid_cls_score, source_pid_domain_cls_score)
        cid_classifier_loss = base.camera_classifier_creiteron(source_cid_cls_score, target_cid_cls_score, source_cids,
                                                               target_cids)

        loss = source_pid_loss + multi_view_pid_loss + source_pid_domain_loss + config.lambda1 * target_pid_domain_loss\
               - config.lambda2 * source_dispency_loss + config.lambda3 * cid_classifier_loss

        base.feature_extractor_optimizer.zero_grad()
        base.identity_classifier_optimizer.zero_grad()
        base.identitydomain_classifier_optimizer.zero_grad()
        base.camera_classifier_optimizer.zero_grad()

        loss.backward()

        base.identity_classifier_optimizer.step()
        base.identitydomain_classifier_optimizer.step()
        base.camera_classifier_optimizer.step()

        source_features = base.feature_extractor(source_imgs)
        target_features = base.feature_extractor(target_imgs)
        source_pid_cls_score = base.identity_classifier(source_features)
        source_cid_cls_score = base.camera_classifier(source_features)
        multi_view_features = MultiViewReasoning().__call__(source_features, source_features, source_pids, source_pids,
                                                            source_pid_cls_score)
        multi_view_cls_score = base.identity_classifier(multi_view_features)
        source_pid_domain_cls_score = base.identitydomain_classifier(source_features)
        target_cid_cls_score = base.camera_classifier(target_features)
        target_pid_domain_cls_score = base.identitydomain_classifier(target_features)
        source_pid_loss = base.source_pid_creiteron(source_pid_cls_score, source_pids)
        multi_view_pid_loss = base.source_pid_creiteron(multi_view_cls_score, source_pids)
        single_multi_view_feature_loss = base.single_multi_creiteron(source_features, multi_view_features)
        source_pid_domain_loss = base.source_identity_domain_creiteron(source_pid_domain_cls_score, source_pids)
        target_pid_domain_loss = base.target_identity_domain_creiteron(target_pid_domain_cls_score)
        source_dispency_loss = base.dispency_creiteron(source_pid_cls_score, source_pid_domain_cls_score)
        cid_feature_extractor_loss = base.camera_feature_extractor_creiteron(source_cid_cls_score, target_cid_cls_score)


        loss = source_pid_loss + multi_view_pid_loss + source_pid_domain_loss + \
               config.beta * single_multi_view_feature_loss + config.lambda1 * target_pid_domain_loss \
               + config.lambda2 * source_dispency_loss + config.lambda3 * cid_feature_extractor_loss
        base.feature_extractor_optimizer.zero_grad()
        base.identity_classifier_optimizer.zero_grad()
        base.identitydomain_classifier_optimizer.zero_grad()
        base.camera_classifier_optimizer.zero_grad()

        loss.backward()

        base.feature_extractor_optimizer.step()

        meter.update({'source_pid_loss': source_pid_loss.data, 'multi_view_pid_loss': multi_view_pid_loss.data,
                      'single_multi_view_feature_loss': single_multi_view_feature_loss.data, 'source_pid_domain_loss':
                      source_pid_domain_loss.data, 'target_pid_domain_loss': target_pid_domain_loss.data,
                      'source_dispency_loss': source_dispency_loss.data, 'cid_classifier_loss': cid_classifier_loss.data,
                      'cid_feature_extractor_loss': cid_feature_extractor_loss})


    return meter.get_val(), meter.get_str()