import torch
import numpy as np
import argparse
from pprint import pformat
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import PSImg_Dataset
from model import PSOCT_module
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve
from sklearn.utils import resample
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', help="number of epoch", type=int, default=20)
parser.add_argument('-m', '--mode', help="image fusion mode", type=str, default='dup_backbone')
parser.add_argument('-b', '--best_test_only', help='run best test only', action='store_true')
parser.add_argument('-a', '--augmentation', help='use data augmentation method', action='store_true')
parser.add_argument('-o', '--out_file', help='model saving file', default='./saved_model/best_model_psoct_{}.pth')
parser.add_argument('--keys', help='psoct measurements to use', default='dor')
args = parser.parse_args()

epoch = args.epoch
best_test_only = args.best_test_only
mode = args.mode
aug = args.augmentation
if aug:
    out_file = args.out_file.format(mode + '_' + args.keys + '_aug')
else:
    out_file = args.out_file.format(mode + '_' + args.keys)
use_amp = False
ps_keys = []
if 'd' in args.keys:
    ps_keys.append('dopu')
if 'o' in args.keys:
    ps_keys.append('optic')
if 'r' in args.keys:
    ps_keys.append('retard')
ps_keys += ['oct']

print('args - {}'.format(args))

train_dataset = PSImg_Dataset(split='train', split_mode='stratified', augment=aug)
val_dataset = PSImg_Dataset(split='val', split_mode='stratified', augment=aug)
test_dataset = PSImg_Dataset(split='test', split_mode='stratified', augment=aug)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.get_collate())
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=val_dataset.get_collate())
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=test_dataset.get_collate())

model = PSOCT_module(mode=mode, ps_keys=ps_keys).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criteria = torch.nn.CrossEntropyLoss().to(device)
scaler = GradScaler()

def train_epoch(model, device, train_loader, optimizer):
    model.train()
    total_loss = []
    for idx, (img_group, label) in enumerate(train_loader):
        for k, v in img_group.items():
            img_group[k] = v.to(device)
        label = label.to(device)
        with autocast(enabled=use_amp):
            if (mode == 'dup_backbone') or (mode == 'oct_only'):
                preds = model(img_group)
                assert not preds.isnan().any()
            else:
                input_group = {k: img_group[k] for k in ps_keys}
                x = torch.cat(list(input_group.values()), dim=1).to(device)
                preds = model(x)
                assert not preds.isnan().any()
            loss = criteria(preds, label)
        assert not loss.isnan().any()

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        assert not loss.isnan().any()
        loss_num = loss.data.item()
        total_loss.append(loss.data * len(label))
        if idx % 1 == 0:
            print('batch [{}/{}] loss: {:.3f}'.format(idx + 1, len(train_loader), loss_num))
    avg_loss = torch.sum(torch.stack(total_loss)) / len(train_dataset)
    return avg_loss

@torch.no_grad()
def val_epoch(model, device, val_loader):
    model.eval()
    all_targets = []
    all_preds = []
    for idx, (img_group, label) in enumerate(val_loader):
        for k, v in img_group.items():
            img_group[k] = v.to(device)
        label = label.to(device)
        with autocast(enabled=use_amp):
            if (mode == 'dup_backbone') or (mode == 'oct_only'):
                preds = model(img_group)
                assert not preds.isnan().any()
            else:
                input_group = {k: img_group[k] for k in ps_keys}
                x = torch.cat(list(input_group.values()), dim=1).to(device)
                preds = model(x)
        all_targets.append(label)
        all_preds.append(preds.to('cpu'))
    all_targets = torch.cat(all_targets).to('cpu').float().numpy()
    all_preds = torch.cat(all_preds).float()
    all_preds = torch.softmax(all_preds, dim=1)[:, 1].to('cpu').numpy()
    auroc = roc_auc_score(all_targets, all_preds)
    return auroc

@torch.no_grad()
def bootstrap_test(results, boot_num = 1000, threshold=None):
    all_targets = results['targets']
    all_probs = results['probs']
    all_preds = results['preds']
    test_size = len(all_targets)

    aurocs, aps, precisions, recalls, f1s, accs = [], [], [], [], [], []
    for _ in range(boot_num):
        boot = resample(np.arange(test_size), replace=True, n_samples=test_size)
        targets = all_targets[boot]
        probs = all_probs[boot]
        preds = all_preds[boot]

        auroc = roc_auc_score(targets, probs)
        ap = average_precision_score(targets, probs)
        report = classification_report(targets, preds, target_names=['negative', 'positive'], output_dict=True)
        precision, recall, f1 = report['positive']['precision'], report['positive']['recall'], report['positive']['f1-score']
        acc = report['accuracy']

        aurocs.append(auroc)
        aps.append(ap)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accs.append(acc)
    aurocs, aps, precisions, recalls, f1s, accs = np.array(aurocs), np.array(aps), np.array(precisions), np.array(recalls), np.array(f1s), np.array(accs)
    auroc, ap, precision, recall, f1, acc = np.mean(aurocs), np.mean(aps), np.mean(precisions), np.mean(recalls), np.mean(f1s), np.mean(accs)
    auroc_std, ap_std, precision_std, recall_std, f1_std, acc_std = np.std(aurocs), np.std(aps), np.std(precisions), np.std(recalls), np.std(f1s), np.std(accs)
    return {'auroc': (auroc, auroc_std),
            'ap': (ap, ap_std), 
            'precision': (precision, precision_std), 
            'recall': (recall, recall_std), 
            'f1': (f1, f1_std), 
            'acc': (acc, acc_std)}

@torch.no_grad()
def cal_threshold(all_targets, all_probs):
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
    eval_metric = tpr - fpr
    candidates = np.argmax(eval_metric)

    mask = eval_metric == eval_metric[candidates]
    tpr_candidates = tpr[mask]
    best_idx = np.argmax(tpr_candidates)
    best_thresh = thresholds[mask][best_idx]
    
    if best_thresh == 0 or best_thresh == np.inf:
        best_thresh = 0.5
    return best_thresh

@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    all_targets = []
    all_preds = []
    for idx, (img_group, label) in enumerate(test_loader):
        for k, v in img_group.items():
            img_group[k] = v.to(device)
        label = label.to(device)
        with autocast(enabled=use_amp):
            if (mode == 'dup_backbone') or (mode == 'oct_only'):
                preds = model(img_group)
                assert not preds.isnan().any()
            else:
                input_group = {k: img_group[k] for k in ps_keys}
                x = torch.cat(list(input_group.values()), dim=1).to(device)
                preds = model(x)
        all_targets.append(label)
        all_preds.append(preds.to('cpu'))
    all_targets = torch.cat(all_targets).to('cpu').float().numpy()
    all_preds = torch.cat(all_preds).float()
    all_probs = torch.softmax(all_preds, dim=1).to('cpu').numpy()
    all_probs = all_probs[:, 1]
    auroc = roc_auc_score(all_targets, all_probs)
    ap = average_precision_score(all_targets, all_probs)
    threshold = cal_threshold(all_targets, all_probs)
    all_preds = np.zeros_like(all_probs)
    all_preds[all_probs > threshold] = 1
    report = classification_report(all_targets, all_preds, target_names=['negative', 'positive'])
    boot_results = bootstrap_test({'targets': all_targets, 'probs': all_probs, 'preds': all_preds})
    positive_num = all_preds.sum()
    return auroc, ap, report, positive_num, boot_results

@torch.no_grad()
def best_test(model, device, test_loader):
    model.load_state_dict(torch.load(out_file))
    auroc, ap, report, positive_num, boot_results = test(model, device, test_loader)
    print('test metric -- auroc:{:.3f}'.format(auroc))
    print('test metric -- ap:{:.3f}'.format(ap))
    print('test metric -- predicted positive:{}'.format(positive_num))
    print('test metric -- report:\n{}'.format(report))
    print('test metric -- boot results:\n{}'.format(pformat(boot_results)))
    return auroc, ap, report, positive_num


def train(model, device, train_loader, val_loader, test_loader, optimizer, epoch):
    best_roc = 0
    for epoch_idx in tqdm(range(epoch)):
        print('Epoch [{}/{}] '.format(epoch_idx + 1, epoch))
        epoch_loss = train_epoch(model, device, train_loader, optimizer)
        print('Epoch [{}/{}] loss:{:.3f}'.format(epoch_idx + 1, epoch, epoch_loss))

        auroc = val_epoch(model, device, val_loader)

        if auroc > best_roc:
            print('new best auroc: {} -> {}'.format(best_roc, auroc))
            best_roc = auroc
            print('model saved.')
            torch.save(model.state_dict(), out_file)

    best_test(model, device, test_loader)

if __name__ == "__main__":
    if best_test_only:
        best_test(model, device, test_loader)
    else:
        train(model, device, train_loader, val_loader, test_loader, optimizer, epoch)
    print('Done')