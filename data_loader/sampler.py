
import random
import torch.utils.data as data

class ClassUniformlySampler(data.sampler.Sampler):

    def __init__(self, data_source, class_position, k):
        self.data_source = data_source
        self.class_position = class_position
        self.k = k

        self.samples = self.data_source.samples
        self.class_dict = self._tuple2dict(self.samples)
        self.cid_list = self._cid_list(self.samples)
        self.sample_list = self._generate_list(self.class_dict, self.cid_list)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict, self.cid_list)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _cid_list(self, inputs):
        cid_list = []
        for input in inputs:
            cid_list.append(input[2])
        return cid_list

    def _tuple2dict(self, inputs):
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict

    def _generate_list(self, dict, cid_list):
        sample_list = []
        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            same_views = []
            different_views = []
            values = dict_copy[key]
            anchor = random.choice(values)
            for value in values:
                if cid_list[value] == cid_list[anchor]:
                    same_views.append(value)
                else:
                    different_views.append(value)
            if len(different_views) >= self.k:
                random.shuffle(different_views)
                sample_list.extend(different_views[0: self.k])
            else:
                different_views = different_views * self.k
                random.shuffle(different_views)
                sample_list.extend(different_views[0: self.k])

        return sample_list