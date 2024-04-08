import torch
import pandas as pd
import torchvision.transforms as transforms 
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupShuffleSplit

default_transform = transforms.Compose([
            transforms.ToTensor()
])

aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
])

img_types = {"DOPU": "dopu", "Optic Axis":'optic', "Retardation":"retard", "Total Intensity":"oct"}
img_channels = {"DOPU": "L", "Optic Axis":'RGB', "Retardation":"RGB", "Total Intensity":"L"}

class PSImg_Dataset(Dataset):
    def __init__(self, transforms = default_transform, meta_path = './data/meta_ps.csv', split='all', split_mode='imgOut', augment=False, aug_transform = aug_transform):
        super(PSImg_Dataset, self).__init__()
        self.transforms = transforms
        self.meta_path = meta_path
        self.split = split
        self.split_mode = split_mode
        self.augment = augment
        if augment:
            self.aug_transform = aug_transform

        crop_set = pd.read_csv(meta_path, dtype={'location':"int64", 'label':'int64'})
        img_set = crop_set[['origin_name', 'label']].drop_duplicates().reset_index(drop=True)
        label = img_set.pop('label').to_numpy()

        if split == 'simple':
            label = crop_set.pop('label').to_numpy()
            self.data_table, self.label = crop_set.iloc[:8], label[:8]
        elif split != 'all':
            if split_mode == 'stratified':
                spliter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
                train_index, val_test_index = next(spliter.split(list(range(len(crop_set))), 
                                                                 crop_set['label'], 
                                                                 crop_set['patient']))
                spliter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=0)
                val_index, test_index = next(spliter.split(val_test_index, 
                                                           crop_set['label'].iloc[val_test_index],
                                                           crop_set['patient'].iloc[val_test_index]))
                val_index, test_index = val_test_index[val_index], val_test_index[test_index]
                X_select = train_index if split == 'train' else (val_index if split == 'val' else test_index)
                self.data_table, self.label = crop_set.iloc[X_select], crop_set['label'].iloc[X_select].to_numpy()

            elif split_mode == 'groupRatio':
                spliter = GroupShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
                train_index, val_test_index = next(spliter.split(list(range(len(crop_set))), 
                                                                 crop_set['label'], 
                                                                 crop_set['patient']))
                spliter = GroupShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
                val_index, test_index = next(spliter.split(val_test_index, 
                                                           crop_set['label'].iloc[val_test_index],
                                                           crop_set['patient'].iloc[val_test_index]))
                val_index, test_index = val_test_index[val_index], val_test_index[test_index]
                X_select = train_index if split == 'train' else (val_index if split == 'val' else test_index)
                self.data_table, self.label = crop_set.iloc[X_select], crop_set['label'].iloc[X_select].to_numpy()

            else:
                X_train, X_test, Y_train, Y_test = train_test_split(list(range(len(img_set))),
                                                                    label, test_size=0.3, random_state=0)
                X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

                if split == 'train':
                    X_select = X_train
                elif split == 'val':
                    X_select = X_val
                else:
                    X_select = X_test

                data_table = img_set.iloc[X_select][['origin_name']]
                self.data_table = pd.merge(data_table, crop_set, on=['origin_name'])
                self.label = self.data_table.pop('label').to_numpy()
            pass
        else:
            self.data_table = pd.merge(img_set, crop_set, on=['origin_name'])
            self.label = self.data_table.pop('label').to_numpy()

    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        item = self.data_table.iloc[index]
        img_name = item.img_name
        label = self.label[index]

        img_group = {}
        for k, v in img_types.items():
            img = self.transforms(Image.open('./data/imgs_ps/{}_{}.png'.format(img_name, v)).convert(img_channels[k]))
            if img.shape[0] == 1:
                img = torch.cat([img, img, img])
            img_group[v] = img

        if self.augment and (self.split == 'train'):
            img_ = torch.cat([img_group[k] for k in sorted(img_group.keys())])
            img_ = self.aug_transform(img_)
            img_ = torch.split(img_, [3] * len(img_group))
            for i, k in enumerate(sorted(img_group.keys())):
                img_group[k] = img_[i]

        return img_group, label
    
    def get_collate(self):
        def collate_fn(data):
            imgs, labels = map(list, zip(*data))
            img_batch = {}
            for k in img_types.values():
                img_list = list(map(lambda x: x[k], imgs))
                img_batch[k] = torch.stack(img_list)
            return img_batch, torch.LongTensor(labels)
        return collate_fn
