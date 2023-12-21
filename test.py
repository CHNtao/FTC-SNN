# '/home/mem/ct/paper4-2023-8-14/MSNN_CIFAR10DVS_2023/2023_9_11_17.pth'
import argparse
import shutil
import os
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data

from pathlib import Path
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import datetime
import xlwt
from layers import *
from torch.cuda import amp
import odata_loaders as data_loaders
from VGG_models import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=5e-2)
parser.add_argument('--dts', type=str, default='CIFAR10DVS')
parser.add_argument('--model', type=str, default='MSNN')
parser.add_argument('--result_path', type = Path, default = './result' )
parser.add_argument('--T', type=int, default=2)
parser.add_argument('--TET', type=bool, default=True)
parser.add_argument('--means', type=float, default=1,help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--lamb', type=float, default=1e-3)
parser.add_argument('--amp', type=bool,default=True)



opt = parser.parse_args()
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


save_path = './' + opt.model + '_' + opt.dts + '_' + str(opt.seed)
if not os.path.exists(save_path):
    os.mkdir(save_path)
    

    
    
if opt.dts == 'CIFAR10':
    train_dataset = dsets.CIFAR10(root='/home/mem/ct_pre/data/CIFAR10', train=True, transform=transform_train, download=True)
    test_dataset = dsets.CIFAR10(root='/home/mem/ct_pre/data/CIFAR10', train=False, transform=transform_test)
elif opt.dts == 'CIFAR100':
    train_dataset = dsets.CIFAR100(root='/home/mem/ct_pre/data/CIFAR100', train=True, transform=transform_train, download=True)
    test_dataset = dsets.CIFAR100(root='/home/mem/ct_pre/data/CIFAR100', train=False, transform=transform_test)
elif opt.dts == 'MNIST':
    train_dataset = dsets.MNIST(root='/media/mem/2879EFC362FD4C9E/data/mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='/media/mem/2879EFC362FD4C9E/data/mnist', train=False,
                                    transform=transforms.ToTensor())
elif opt.dts == 'Fashion-MNIST':
    train_dataset = dsets.FashionMNIST(root = '/media/mem/2879EFC362FD4C9E/data/fashion', train = True, transform = transforms.ToTensor(), download = True)
    test_dataset = dsets.FashionMNIST(root = '/media/mem/2879EFC362FD4C9E/data/fashion', train = False, transform = transforms.ToTensor())
elif opt.dts == 'CIFAR10DVS':
    #  TET datasets
    train_dataset, test_dataset = data_loaders.build_dvscifar('/home/mem/ct_pre/data/CIFAR10DVS')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory = True,drop_last = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory = True)

    



train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,num_workers = 4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False,num_workers = 4)

pretrained_model = 'MSNN_CIFAR10DVS_2023/2023_9_11_17.pth'
model = VGGSNN2()

model.to(device)
model.load_state_dict(torch.load(pretrained_model,map_location=torch.device('cpu')))
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        out = torch.mean(out, dim=1)
        pred = out.max(1)[1]
        total += targets.size(0)
        correct += (pred ==targets).sum()
    acc = 100.0 * correct.item() / total   
    print('Test correct: %d Accuracy: %.2f%%' % (correct, acc))
    
        
    











    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
