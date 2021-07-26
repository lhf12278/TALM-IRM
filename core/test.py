
import torch

from tools import CatMeter, time_now, ReIDEvaluator

def test(config, base, loader):

    base.set_eval()

    source_query_features_meter, source_query_pids_meter, source_query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    source_gallery_features_meter, source_gallery_pids_meter, source_gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    target_query_features_meter, target_query_pids_meter, target_query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    target_gallery_features_meter, target_gallery_pids_meter, target_gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    source_loaders = [loader.source_query_loader, loader.source_gallery_loader]
    target_loaders = [loader.target_query_loader, loader.target_gallery_loader]

    print(time_now(), 'source_feature start')

    with torch.no_grad():
        for source_loader_id, source_loader in enumerate(source_loaders):
            for source_data in source_loader:
                source_images, source_pids, source_cids = source_data
                source_images = source_images.to(base.device)
                source_features = base.feature_extractor(source_images).squeeze()

                if source_loader_id == 0:
                    source_query_features_meter.update(source_features.data)
                    source_query_pids_meter.update(source_pids)
                    source_query_cids_meter.update(source_cids)
                elif source_loader_id == 1:
                    source_gallery_features_meter.update(source_features.data)
                    source_gallery_pids_meter.update(source_pids)
                    source_gallery_cids_meter.update(source_cids)

    print(time_now(), 'source_features done')

    print(time_now(), 'target_feature start')

    with torch.no_grad():
        for target_loader_id, target_loader in enumerate(target_loaders):
            for target_data in target_loader:
                target_images, target_pids, target_cids = target_data
                target_images = target_images.to(base.device)
                target_features = base.feature_extractor(target_images).squeeze()

                if target_loader_id == 0:
                    target_query_features_meter.update(target_features.data)
                    target_query_pids_meter.update(target_pids)
                    target_query_cids_meter.update(target_cids)
                elif target_loader_id == 1:
                    target_gallery_features_meter.update(target_features.data)
                    target_gallery_pids_meter.update(target_pids)
                    target_gallery_cids_meter.update(target_cids)

    print(time_now(), 'target_features done')

    source_query_features = source_query_features_meter.get_val_numpy()
    source_gallery_features = source_gallery_features_meter.get_val_numpy()

    target_query_features = target_query_features_meter.get_val_numpy()
    target_gallery_features = target_gallery_features_meter.get_val_numpy()

    source_mAP, source_CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
        source_query_features, source_query_pids_meter.get_val_numpy(), source_query_cids_meter.get_val_numpy(),
        source_gallery_features, source_gallery_pids_meter.get_val_numpy(), source_gallery_cids_meter.get_val_numpy())

    target_mAP, target_CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
        target_query_features, target_query_pids_meter.get_val_numpy(), target_query_cids_meter.get_val_numpy(),
        target_gallery_features, target_gallery_pids_meter.get_val_numpy(), target_gallery_cids_meter.get_val_numpy())

    return source_mAP, source_CMC[0: 20], target_mAP, target_CMC[0: 20]



