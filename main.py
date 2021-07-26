
import os
import ast
import argparse

from data_loader.loader import Loader
from core import Base, train_stage1, train_stage2, test
from tools import make_dirs, Logger, os_walk, time_now

def main(config):

    loaders = Loader(config)
    model = Base(config)

    make_dirs(model.output_path)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)

    if config.mode == 'train':
        if config.resume_train_epoch >= 0:
            model.resume_model(config.resume_train_epoch)
            start_train_epoch = config.resume_train_epoch
        else:

            start_train_epoch = 0

        if config.auto_resume_training_from_lastest_step:
            root, _, files = os_walk(model.save_model_path)
            if len(files) > 0:
                indexes = []
                for file in files:
                    indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
                indexes = sorted(list(set(indexes)), reverse=False)
                model.resume_model(indexes[-1])
                start_train_epoch = indexes[-1]
                logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(),
                                    indexes[-1]))

        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            model.save_model(current_epoch)
            model.feature_extractor_lr_scheduler.step(current_epoch)
            model.identity_classifier_lr_scheduler.step(current_epoch)
            model.identitydomain_classifier_lr_scheduler.step(current_epoch)

            if current_epoch < 120:
                _, result = train_stage1(config, model, loaders)
                logger('Time: {}; Epoch: {}; {}'.format(time_now(), current_epoch, result))
            else:
                _, result = train_stage2(config, model, loaders)
                logger('Time: {}; Epoch: {}; {}'.format(time_now(), current_epoch, result))

            if current_epoch + 1 >= 120 and (current_epoch + 1) % 20 == 0:
                source_mAP, source_CMC, target_mAP, target_CMC = test(config, model, loaders)
                logger('Time: {}; Test on Source Dataset: {}, \nmAP: {} \n Rank: {}'.format(time_now(),
                                                                                            config.source_dataset,
                                                                                            source_mAP, source_CMC))
                logger('Time: {}; Test on Target Dataset: {}, \nmAP: {} \n Rank: {}'.format(time_now(),
                                                                                            config.target_dataset,
                                                                                            target_mAP, target_CMC))

    elif config.mode == 'test':
        model.resume_model(config.resume_test_model)
        source_mAP, source_CMC, target_mAP, target_CMC = test(config, model, loaders)
        logger('Time: {}; Test on Source Dataset: {}, \nmAP: {} \n Rank: {}'.format(time_now(), config.source_dataset,
                                                                                    source_mAP, source_CMC))
        logger('Time: {}; Test on Target Dataset: {}, \nmAP: {} \n Rank: {}'.format(time_now(), config.target_dataset,
                                                                                    target_mAP, target_CMC))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='test', help='train, test')
    parser.add_argument('--market_path', type=str, default='G:\datasets\Market-1501-v15.09.15')
    parser.add_argument('--duke_path', type=str, default='G:\datasets\DukeMTMC\DukeMTMC-reID')
    parser.add_argument('--msmt17_path', type=str, default='G:\datasets\MSMT17V1')
    parser.add_argument('--prid_path', type=str, default='G:\datasets\PRID2011\split')
    parser.add_argument('--grid_path', type=str, default='G:\datasets\GRID')
    parser.add_argument('--source_dataset', type=str, default='market',
                        help='market_train, duket_train, msmt17_train')
    parser.add_argument('--target_dataset', type=str, default='duke',
                        help='market_train, duket_train, msmt17_train')
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--use_rea', default=False, help='use random erasing augmentation')
    parser.add_argument('--use_colorjitor', default=True, help='use random erasing augmentation')
    parser.add_argument('--p', type=int, default=4)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--cnnbackbone', type=str, default='res50', help='res50, res50ibna')
    parser.add_argument('--in_dim', type=int, default=2048)
    parser.add_argument('--source_pid_num', type=int, default=751)
    parser.add_argument('--source_cid_num', type=int, default=6)
    parser.add_argument('--target_cid_num', type=int, default=8)
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.015)
    parser.add_argument('--lambda1', type=float, default=0.01)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.2)
    parser.add_argument('--chunks', type=int, default=4)
    parser.add_argument('--e_learning_rate', type=float, default=0.0003)
    parser.add_argument('--ic_learning_rate', type=float, default=0.0003)
    parser.add_argument('--idc_learning_rate', type=float, default=0.0003)
    parser.add_argument('--cc_learning_rate', type=float, default=0.00003)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    parser.add_argument('--output_path', type=str, default='market_duke_id_domain_0.5_baseline/base/',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=True)
    parser.add_argument('--total_train_epoch', type=int, default=140)
    parser.add_argument('--resume_test_model', type=int, default=139, help='-1 for no resuming')
    parser.add_argument('--test_mode', type=str, default='inter-camera', help='inter-camera, intra-camera, all')

    config = parser.parse_args()
    main(config)