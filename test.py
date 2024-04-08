import os
import torch
from model.ResNet import PSOCT_module
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import argparse


default_transform = transforms.Compose([
            transforms.ToTensor()
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help="image fusion mode", type=str, default='dup_backbone')
parser.add_argument('-a', '--augmentation', help='use data augmentation method', action='store_true')
parser.add_argument('-s', '--saved_model', help='model saving file', default='./saved_model/best_model_psoct_{}.pth')
parser.add_argument('--keys', help='psoct measurements to use', default='dor')
args = parser.parse_args()

mode = args.mode
if args.augmentation:
    saved_model = args.saved_model.format(mode + '_' + args.keys + '_aug')
else:
    saved_model = args.saved_model.format(mode + '_' + args.keys)

model = PSOCT_module(mode=mode).to(device)
model.load_state_dict(torch.load(saved_model, map_location=device))

@torch.no_grad()
def test_curve(model, device, test_img, step=5):
    model.eval()
    img_group = {}
    for k, v in test_img.items():
        img_group[k] = v.to(device)
    height, width = v.shape[1], v.shape[2]
    start_list = [i * step for i in range((width - height) // step + 1)]
    if start_list[-1] < (width - height):
        start_list.append(width - height)
    pred_list = []

    for start in start_list:
        img_patch = {}
        for k, v in img_group.items():
            img_patch[k] = v[:,:,start:(start + height)].unsqueeze(0)
        if (model.mode == 'dup_backbone') or (model.mode == 'oct_only'):
            pred = model(img_patch).squeeze()
        else:
            x = torch.cat(list(img_patch.values()), dim=1).to(device)
            pred = model(x)
        pred = torch.softmax(pred, dim=0)
        pred_list.append(pred[1])
    pred_list = torch.tensor(pred_list)

    curve_mat = torch.zeros((width, len(start_list)))
    for idx, start in enumerate(start_list):
        curve_mat[start:(start + height), idx] = 1
    curve_mat = curve_mat / curve_mat.sum(dim=1, keepdim=True)

    curve = curve_mat @ pred_list

    return curve

img_types = {"DOPU": "dopu", "Optic Axis":'optic', "Retardation":"retard", "Total Intensity":"oct"}
img_channels = {"DOPU": "L", "Optic Axis":'RGB', "Retardation":"RGB", "Total Intensity":"L"}

if __name__ == '__main__':
    subfolder = ['positive', 'negative']
    for folder in subfolder:
        path = 'data/test_ps/{}/'.format(folder)
        img_list = os.listdir(path + 'DOPU')
        for img_name in img_list:
            img_group = {}
            for k, v in img_types.items():
                img = Image.open(os.path.join(path, k, img_name)).convert(img_channels[k])
                img_group[v] = Image.open(os.path.join(path, k, img_name)).convert(img_channels[k])
            for k, v in img_group.items():
                assert img.size == v.size
            width, height = img.size
            width, height = (width*436) // height, 436
            for k, v in img_group.items():
                img = default_transform(v.resize((width, height), resample=Image.LANCZOS))
                if img.shape[0] == 1:
                    img = torch.cat([img, img, img])
                img_group[k] = img
            y = test_curve(model, device, img_group, step=5)
            plt.figure(figsize=((img.shape[2]+50)/200, (img.shape[1]+10)/200), dpi=200)
            plt.tight_layout()
            plt.plot(y)
            plt.xlim([0, img.shape[2]])
            plt.ylim([-0.03, 1.03])
            plt.savefig(os.path.join(path, '{}.png'.format(img_name)))
            plt.close()

print('Done')
    