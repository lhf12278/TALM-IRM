
import os
import copy
import numpy as np
from PIL import Image

from tools import os_walk

class SourcePersonReIDSamples:

    def __init__(self, dataset_path):

        self.dataset_path = os.path.join(dataset_path, 'bounding_box_train/')
        samples = self._load_samples(self.dataset_path)
        samples = self._reorder_labels(samples, 1)
        samples = self._reorder_labels(samples, 2)
        self.samples = samples

    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        ids = list(set(ids))
        ids.sort()
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name:
                identity_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[1])
        return identity_id, camera_id

class TargetPersonReIDSamples:

    def __init__(self, dataset_path):

        self.dataset_path = os.path.join(dataset_path, 'bounding_box_train/')
        samples = self._load_samples(self.dataset_path)
        samples = self._reorder_labels(samples, 2)
        self.samples = samples

    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        ids = list(set(ids))
        ids.sort()
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        file_names = []
        for file_name in files_name:
            if 'jpg' in file_name:
                file_names.append(file_name)
        file_names = file_names[::-1]
        for i, sample in enumerate(file_names):
            identity_id = i
            camera_id = self._analysis_file_name(sample)
            samples.append([root_path + sample, identity_id, camera_id])

        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        camera_id = int(split_list[1])
        return camera_id

class TestPersonReIDSamples:

    def __init__(self, dataset_path):

        self.query_path = os.path.join(dataset_path, 'query/')
        self.gallery_path = os.path.join(dataset_path, 'bounding_box_test/')
        query_samples = self._load_samples(self.query_path)
        gallery_samples = self._load_samples(self.gallery_path)
        self.query_samples = query_samples
        self.gallery_samples = gallery_samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name:
                identity_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[1])
        return identity_id, camera_id

class SourceSamples4Market(SourcePersonReIDSamples):

    pass

class SourceSamples4Duke(SourcePersonReIDSamples):

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class SourceSamples4MSMT17(SourcePersonReIDSamples):

    def _analysis_file_name(self, file_name):

        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[2])
        return identi_id, camera_id

class TargetSamples4Market(TargetPersonReIDSamples):

    pass

class TargetSamples4Duke(TargetPersonReIDSamples):

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        camera_id = int(split_list[1])
        return camera_id

class TargetSamples4MSMT17(TargetPersonReIDSamples):

    def _analysis_file_name(self, file_name):

        split_list = file_name.replace('.jpg', '').split('_')
        camera_id = int(split_list[2])
        return camera_id

class TargetSamples4Prid:

    def __init__(self, dataset_path):

        self.dataset_path = os.path.join(dataset_path, 'train_9/')
        samples = self._load_samples(self.dataset_path)
        samples = self._reorder_labels(samples, 2)
        self.samples = samples

    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        ids = list(set(ids))
        ids.sort()
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        file_names = []
        for file_name in files_name:
            if 'png' in file_name:
                file_names.append(file_name)
        file_names = file_names[::-1]
        for i, sample in enumerate(file_names):
            identity_id = i
            camera_id = self._analysis_file_name(sample)
            samples.append([root_path + sample, identity_id, camera_id])

        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.png', '').split('_')
        camera_id = int(split_list[2])
        return camera_id

class TargetSamples4Grid:

    def __init__(self, dataset_path):

        self.dataset_path = os.path.join(dataset_path, 'train_0/')
        samples = self._load_samples(self.dataset_path)
        samples = self._reorder_labels(samples, 2)
        self.samples = samples

    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        ids = list(set(ids))
        ids.sort()
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        file_names = []
        for file_name in files_name:
            if 'jpeg' in file_name:
                file_names.append(file_name)
        file_names = file_names[::-1]
        for i, sample in enumerate(file_names):
            identity_id = i
            camera_id = self._analysis_file_name(sample)
            samples.append([root_path + sample, identity_id, camera_id])

        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpeg', '').split('_')
        camera_id = int(split_list[7])
        return camera_id

class TestSamples4Market(TestPersonReIDSamples):

    pass

class TestSamples4Duke(TestPersonReIDSamples):

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class TestSamples4MSMT17(TestPersonReIDSamples):

    def _analysis_file_name(self, file_name):

        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[2])
        return identi_id, camera_id

class TestSamples4Prid:

    def __init__(self, dataset_path):

        self.query_path = os.path.join(dataset_path, 'query_9/')
        self.gallery_path = os.path.join(dataset_path, 'gallery_9/')
        query_samples = self._load_samples(self.query_path)
        gallery_samples = self._load_samples(self.gallery_path)
        self.query_samples = query_samples
        self.gallery_samples = gallery_samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'png' in file_name:
                identity_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.png', '').split('_')
        identity_id, camera_id = int(split_list[1]), int(split_list[2])
        return identity_id, camera_id

class TestSamples4Grid:

    def __init__(self, dataset_path):

        self.query_path = os.path.join(dataset_path, 'query_0/')
        self.gallery_path = os.path.join(dataset_path, 'gallery_0/')
        query_samples = self._load_samples(self.query_path)
        gallery_samples = self._load_samples(self.gallery_path)
        self.query_samples = query_samples
        self.gallery_samples = gallery_samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpeg' in file_name:
                identity_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpeg', '').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[7])
        return identity_id, camera_id

class PersonReIDDataset:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        this_sample = copy.deepcopy(self.samples[index])
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])
        this_sample[2] = np.array(this_sample[2])
        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')
