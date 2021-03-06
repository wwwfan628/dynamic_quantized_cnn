from models.mobilenet_v1 import MobileNetV1
from models.mobilenet_v2 import MobileNetV2
from models.lenet5 import LeNet5
from models.vgg import VGG_small
from utils.datasets import load_dataset
from torch import nn, optim
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
    # traditional training
    train(model, dataloader_train, dataloader_test, args)


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
    optimizer = optim.SGD(model.parameters(), lr=0.045, weight_decay=0.00004, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epoch)
    best_test_acc = 0
    best_epoch = 0
    cur_step = 0
    for epoch in range(150):
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
        print("Epoch {:03d} | Training Loss {:.4f} | Test Acc {:.4f}% | Time(s) {:.4f}".format(epoch + 1, loss, test_accuracy, np.mean(dur)))

        # adjust lr
        scheduler.step()
        # optimizer.param_groups[0]['lr'] *= 0.98

        # early stop
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch
            cur_step = 0
            # save checkpoint
            if args.final_param_path == None:
                final_param_path = './checkpoints/checkpoint_dense_training_' + args.model_name + '_' + args.dataset_name + '.tar'
            else:
                final_param_path = args.final_param_path
            if args.dataset_name == 'ImageNet':
                torch.save(model.module.state_dict(), final_param_path)
            else:
                torch.save(model.state_dict(), final_param_path)
        else:
            cur_step += 1
            if cur_step == 20:
                break
    print("Training finished! Best test accuracy = {:.4f}%, found at Epoch {:03d}.".format(best_test_acc, best_epoch + 1))

if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--dataset_name', default='ImageNet', help='choose dataset from: MNIST, CIFAR10, ImageNet, ImageNet_mini, COCO')
    parser.add_argument('--model_name', default='MobileNetV1', help='choose architecture from: LeNet5, MobileNetV1, MobileNetV2, VGG, ResNet')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for training')
    parser.add_argument('--max_epoch', type=int, default=150, help='max training epoch')
    parser.add_argument('--lr', type=float, default=0.045, help='learning rate of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.00004, help='weight decay of optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of optimizer')
    parser.add_argument('--nesterov', action='store_true', help='nesterov of optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    parser.add_argument('--resume', action='store_true', help='if true, resume training')
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--init_param_path', default=None)
    parser.add_argument('--final_param_path', default=None)
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")