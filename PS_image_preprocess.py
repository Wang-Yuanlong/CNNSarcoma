import os
import pandas as pd
from PIL import Image

base = './data'
out_folder = './data/imgs_ps'
path = './data/raw_ps'
subfolder = ["negative", "positive"]
img_types = {"DOPU": "dopu", "Optic Axis":'optic', "Retardation":"retard", "Total Intensity":"oct"}
img_channels = {"DOPU": "L", "Optic Axis":'RGB', "Retardation":"RGB", "Total Intensity":"L"}
meta_data = pd.read_csv(os.path.join(base, 'meta_ps_origin.csv'), dtype={'label':'int32'})
meta_data['file_name'] = meta_data.apply(lambda x: x['img_name'].replace(' ', '') + '.png', axis=1)
# meta_data['oct_path'] = meta_data.apply(lambda x: os.path.join(path, subfolder[x['label']], 'OCT', x['file_name']), axis=1)
# meta_data['psoct_path'] = meta_data.apply(lambda x: os.path.join(path, subfolder[x['label']], 'PSOCT', x['file_name']), axis=1)
for img_type, img_type_name in img_types.items():
    meta_data['{}_path'.format(img_type_name)] = meta_data.apply(lambda x: os.path.join(path, subfolder[x['label']], img_type, x['file_name']), axis=1)
meta_data['patient'] = meta_data.apply(lambda x: (x['patient'].split(',')[0]), axis=1)

meta_data_new = pd.DataFrame({'img_name':[], 'label':[], 'origin_name':[], 'location':[], 'patient':[]})
for i in range(len(meta_data)):
    item = meta_data.iloc[i]
    img_name = item.img_name.replace(' ', '')
    img_group = {}
    for img_type, img_type_name in img_types.items():
        img = Image.open(item['{}_path'.format(img_type_name)]).convert(img_channels[img_type])
        img_group[img_type] = img
    size = img.size
    for k, v in img_group.items():
        assert v.size == size         
    # width, height = oct_img.size
    width, height = img.size
    width, height = (width*436) // height, 436
    for k, v in img_group.items():
        img_group[k] = v.resize((width, height), resample=Image.LANCZOS)

    if width < height:
            img_group_patch = {k:Image.new(img_channels[k], (height, height)) for k, v in img_group}

            for k, v in img_group_patch.items():
                v.paste(img_group[k], (height-width // 2, 0))

            for k, v in img_group_patch.items():
                v.save(os.path.join(out_folder, img_name + '_{}.png'.format(img_types[k])))

            meta_data_new = pd.concat([meta_data_new, pd.DataFrame({'img_name':[img_name],
                              'label':[item.label],
                              'origin_name':[img_name],
                              'location':[0],
                              'patient':[item.patient]})], ignore_index=True)
    else:
        start_list = [i * 50 for i in range((width - height) // 50 + 1)]
        if start_list[-1] < (width - height):
            start_list.append(width - height)
        for idx, start in enumerate(start_list):
            img_group_patch = {}
            for k, v in img_group.items():
                img_group_patch[k] = v.crop((start, 0, start + height, height))

            new_name = img_name + '_{:0>2d}'.format(idx)
            for k, v in img_group_patch.items():
                v.save(os.path.join(out_folder, new_name + '_{}.png'.format(img_types[k])))
            meta_data_new = pd.concat([meta_data_new, pd.DataFrame({'img_name':[new_name],
                                                            'label':[item.label],
                                                            'origin_name':[img_name],
                                                            'location':[start],
                                                            'patient':[item.patient]})], ignore_index=True)

meta_data_new['label'] = meta_data_new['label'].astype('int64')
meta_data_new['location'] = meta_data_new['location'].astype('int64')
meta_data_new.to_csv(os.path.join(base, 'meta_ps.csv'), index=False)
print('Done')