import os
import pandas as pd
from PIL import Image

base = './data'
out_folder = './data/imgs'
path = './data/raw_1'
subfolder = ["positive", "negative"]

meta_data = pd.DataFrame({'img_name':[], 'label':[], 'origin_name':[], 'location':[]})

for sub in subfolder:
    img_list = os.listdir(os.path.join(path, sub))
    for img_name in img_list:
        img = Image.open(os.path.join(path, sub, img_name))
        width, height = img.size
        if width < height:
            new_img = Image.new('L', (height, height))
            new_img.paste(img, (height-width // 2, 0))
            new_img.save(os.path.join(out_folder, img_name))
            meta_data = pd.concat([meta_data, pd.DataFrame({'img_name':[img_name],
                              'label':[1 if sub == "positive" else 0],
                              'origin_name':[img_name],
                              'location':[0]})], ignore_index=True)
        else:
            start_list = [i * 50 for i in range((width - height) // 50 + 1)]
            if start_list[-1] < (width - height):
                start_list.append(width - height)
            for idx, start in enumerate(start_list):
                new_img = img.crop((start, 0, start + height, height))
                new_name, ext = os.path.splitext(img_name)
                new_name = new_name + '_{:0>2d}{}'.format(idx, ext)
                new_img.save(os.path.join(out_folder, new_name))
                meta_data = pd.concat([meta_data, pd.DataFrame({'img_name':[new_name],
                                                                'label':[1 if sub == "positive" else 0],
                                                                'origin_name':[img_name],
                                                                'location':[start]})], ignore_index=True)

meta_data['label'] = meta_data['label'].astype('int64')
meta_data['location'] = meta_data['location'].astype('int64')

map_dict = {'Miska P': 'Miska P', 'Boo H': 'Boo H', 'Stormy B': 'Stormy B',
            'Kaiser B': 'Kaiser B', 'Briley S': 'Briley S', 'Sadie R': 'Sadie R',
            'Kaiser Sarcoma': 'Kaiser B', 'Gus P': 'Gus P', 'Molly B': 'Molly B',
            'Bella P': 'Bella P', 'Bailey L': 'Bailey L', 'Bear B': 'Bear B', 'Kodi H': 'Kodi H',
            'Dakota B': 'Dakota B', 'Lotus S': 'Lotus S', 'Argus S': 'Argus S', 'Tipper R': 'Tipper R', 
            'Deuce J': 'Deuce J', 'Lexi B': 'Lexi B', 'Blitzen Muscle': 'Blitzen', 'Libby O': 'Libby O',
            'Murphy G': 'Murphy G', 'Mia Li': 'Mia Li', 'Brandy N': 'Brandy N'}

meta_data['patient'] = meta_data.apply(lambda x: map_dict[' '.join(list(x['origin_name'].split(' '))[:2])], axis=1)

meta_data.to_csv(os.path.join(base, 'meta.csv'), index=False)
