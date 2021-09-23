from models.mobilenet_v1 import MobileNetV1
from models.mobilenet_v2 import MobileNetV2
from models.lenet5 import LeNet5
from models.vgg import VGG_small
from utils.datasets import load_dataset
from torch import nn, optim
import torch.nn.utils.prune as prune
import torch
import numpy as np
import argparse
import os
import time


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda Available")
else:
    device = torch.device("cpu")
    print("No Cuda Available")


def main(args):
    # check whether checkpoints directory exist
    path = os.path.join(os.getcwd(), './checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)

    # set flag & parallel processing
    torch.backends.cudnn.benchmark = True
    if args.dataset_name == 'MNIST':
        num_threads = 1
    elif args.dataset_name == 'CIFAR10':
        num_threads = 8
    elif args.dataset_name == 'ImageNet':
        num_threads = 16
    torch.set_num_threads(num_threads)

    # load dataset
    in_channels, num_classes, dataloader_train, dataloader_test = load_dataset(args)

    # build neural network
    if args.model_name == 'LeNet5':
        model = LeNet5(input_channel=in_channels, n_classes=num_classes)
    elif args.model_name == 'VGG':
        model = VGG_small(input_channel=in_channels, n_classes=num_classes).to(device)
    elif args.model_name == 'MobileNetV1':
        model = MobileNetV1(input_channel=in_channels, n_classes=num_classes).to(device)
    elif args.model_name == 'MobileNetV2':
        model = MobileNetV2(input_channel=in_channels, n_classes=num_classes).to(device)
    else:
        print('Architecture not supported! Please choose from: LeNet5, MobileNetV1, MobileNetV2, VGG and ResNet.')

    # parallel training
    if args.dataset_name == 'ImageNet':
        model = torch.nn.DataParallel(model).to(device)

    # train
    if args.init_param_path == None:
        init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '.pth'
    else:
        init_param_path = args.init_param_path
    # save initial parameters
    torch.save(model.state_dict(), init_param_path)
    train(model, dataloader_train, dataloader_test, args)


def save_checkpoint(file_path, model, optimizer, test_acc, train_loss, epoch, args):
    if args.dataset_name == 'ImageNet':
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'train_loss': train_loss,
            'epoch': epoch,
            }, file_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'train_loss': train_loss,
            'epoch': epoch,
        }, file_path)


def compute_zero_percentage(model):
    n_zeros = 0
    n_params = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            n_params += m.weight.numel()
            n_zeros += torch.sum(m.weight == 0).item()
    return n_zeros * 100 / n_params


def validate(model, dataloader_test):
    # validate
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader_test):
            images = images.to(device)
            x = model(images)
            _, pred = torch.max(x, 1)
            pred = pred.data.cpu()
            total += x.size(0)
            correct += torch.sum(pred == labels)
    return correct*100.0/total


def train(model, dataloader_train, dataloader_test, args):
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    if args.resume:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                              nesterov=args.nesterov)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epoch)
        best_test_acc = checkpoint['test_acc']
        best_epoch = checkpoint['epoch']
        cur_epoch = checkpoint['epoch']
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                              nesterov=args.nesterov)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epoch)
        best_test_acc = 0
        best_epoch = 0
        cur_epoch = 0
    for epoch in range(cur_epoch, cur_epoch + args.max_epoch):
        t0 = time.time()  # start time
        model.train()
        for i, (images, labels) in enumerate(dataloader_train):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

        # validate
        dur.append(time.time() - t0)
        test_accuracy = float(validate(model, dataloader_test))
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch
        print("Epoch {:03d} | Training Loss {:.4f} | Test Acc {:.4f}% | Time(s) {:.4f}".format(epoch + 1, loss, test_accuracy, np.mean(dur)))

        # adjust lr
        # scheduler.step()
        optimizer.param_groups[0]['lr'] *= 0.99

        # prune
        if (epoch // 30 != 0) and (epoch % 30 == 0):
            amount = args.amount * 0.2 * (epoch // 30)
            parameters_to_prune = []
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    parameters_to_prune.append((m, 'weight'))
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
            for parameter_to_prune in parameters_to_prune:
                prune.remove(parameter_to_prune[0], parameter_to_prune[1])
            zero_percentage = compute_zero_percentage(model)
            print("Weights contain {:.4f}% 0s.".format(zero_percentage))
    print("Training finished! Best test accuracy = {:.4f}%, found at Epoch {:03d}.".format(best_test_acc, best_epoch + 1))


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--dataset_name', default='ImageNet', help='choose dataset from: MNIST, CIFAR10, ImageNet, ImageNet_mini, COCO')
    parser.add_argument('--model_name', default='MobileNetV1', help='choose architecture from: LeNet5, MobileNetV1, MobileNetV2, VGG, ResNet')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for training')
    parser.add_argument('--max_epoch', type=int, default=180, help='max training epoch')
    parser.add_argument('--lr', type=float, default=0.045, help='learning rate of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay of optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of optimizer')
    parser.add_argument('--nesterov', action='store_true', help='nesterov of optimizer')
    parser.add_argument('--perm_size', type=int, default=16, help='permutation size')
    parser.add_argument('--amount', type=float, default=0.5, help='how many parameters to be pruned')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stop')
    parser.add_argument('--resume', action='store_true', help='if true, resume training')
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--init_param_path', default=None)
    parser.add_argument('--final_param_path', default=None)
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")