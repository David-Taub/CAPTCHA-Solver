import string
import os
import argparse

import mlflow.pytorch
from mlflow.tracking import MlflowClient

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

import captcha_datasets
import generate_captcha


def get_output_shape(layers, input_size, verbose=False):
    head_layer = torch.rand(*input_size)
    if verbose:
        print(f'Input size: {input_size[1:]} [{np.prod(input_size):,}]')
    for layer in layers:
        head_layer = layer(head_layer)
        if verbose:
            print(
                f'Layer {layer.__class__.__name__} size: {tuple(head_layer.shape[1:])} [{np.prod(head_layer.shape):,}]')
    return np.prod(tuple(head_layer.shape))


"""
VGG64
- Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
- ReLU(inplace)
- MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
- Linear(in_features=25088, out_features=4096, bias=True)
- ReLU(inplace)
- Dropout(p=0.5)
- Linear(in_features=4096, out_features=4096, bias=True)
- ReLU(inplace)
- Dropout(p=0.5)
- Linear(in_features=4096, out_features=1000, bias=True)
"""


class Net(nn.Module):
    def __init__(self, input_size, text_length, alphabet_size):
        super(Net, self).__init__()
        expected_input_size = [1, 3] + list(input_size)
        self.text_length = text_length
        self.alphabet_size = alphabet_size
        self.layers = nn.ModuleList()
        # self.layers.append(nn.AvgPool2d(2))
        # self.layers.append(nn.Dropout(0.25))

        self.layers.append(nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.layers.append(nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.layers.append(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        # self.layers.append(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.layers.append(nn.Flatten())
        # self.layers.append(nn.Dropout(0.5))
        self.layers.append(nn.Linear(get_output_shape(self.layers, expected_input_size), 1024, bias=True))
        # self.layers.append(nn.Linear(get_output_shape(self.layers, expected_input_size), 1024))
        # self.layers.append(nn.MaxPool1d(2))
        self.layers.append(nn.Linear(get_output_shape(self.layers, expected_input_size),
                                     self.text_length * self.alphabet_size, bias=True))
        self.layers.append(nn.Unflatten(1, (self.alphabet_size, self.text_length)))
        self.layers.append(nn.LogSoftmax(1))
        get_output_shape(self.layers, expected_input_size, True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch, title=''):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'{title} Train Epoch: {epoch:3d} [{batch_idx * len(data):5d}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):3.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break


def test(model, device, test_loader, title=''):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).to(dtype=torch.float).mean(2).sum().item()
    # show(data, output, test_loader)
    test_loss /= len(test_loader.dataset)
    print(f'{title} Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct:.1f}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.1f}%)')
    return test_loss


def show(data, output, test_loader):
    import matplotlib.pyplot as plt
    inv_dict = {v: k for k, v in test_loader.dataset.alphabet_dict.items()}
    txt = ''.join([inv_dict[i] for i in np.argmax(np.array(output[0].cpu()), 0)])
    plt.imshow(np.array(data.cpu())[0].T)
    plt.title(txt)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    return parser.parse_args()


def pil_transform(img):
    img = np.array(img)
    img = img / 255.0
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()
    img = np.transpose(img, (2, 1, 0))
    return img


def main():
    #######################
    # Training settings
    #######################
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        print('using CUDA')
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #######################
    # Train
    #######################
    NET_INPUT_SIZE = list(np.array([100, 600]) // 4)
    transform = transforms.Compose(
        [
            transforms.Resize(NET_INPUT_SIZE),
            transforms.Lambda(pil_transform),
            # transforms.ToTensor(),
            # transforms.Normalize((0, 0, 0), (1, 1, 1)),
        ])
    PADDED_TEXT_INPUT_SIZE = 10
    TRAIN_EPOCH = 5000
    TEST_EPOCH = 500
    output_alphabet_size = len(string.ascii_lowercase + string.ascii_uppercase + string.digits) + 1
    model = Net(input_size=NET_INPUT_SIZE, text_length=PADDED_TEXT_INPUT_SIZE,
                alphabet_size=output_alphabet_size).to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    print(f'Trainable parameters: {sum([len(p) for p in model.parameters()])}')
    # for lvl_dir in os.listdir('data')[-2:]:
    for i, generator in enumerate(generate_captcha.generators):
        if i != 3:
            continue
        # train_dataset = captcha_datasets.CaptchaDataset(rf'data\\{lvl_dir}\train', transform=transform,
        #                                                 padded_text_length=PADDED_TEXT_INPUT_SIZE)
        # test_dataset = captcha_datasets.CaptchaDataset(rf'data\\{lvl_dir}\test', transform=transform,
        #                                                padded_text_length=PADDED_TEXT_INPUT_SIZE)
        train_dataset = captcha_datasets.DynamicCaptchaDataset(generator, transform=transform,
                                                               padded_text_length=PADDED_TEXT_INPUT_SIZE,
                                                               fake_length=TRAIN_EPOCH)
        test_dataset = captcha_datasets.DynamicCaptchaDataset(generator, transform=transform,
                                                              padded_text_length=PADDED_TEXT_INPUT_SIZE,
                                                              fake_length=TEST_EPOCH)
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        test_loss = np.inf
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, title=f'level_{i}')
            new_test_loss = test(model, device, test_loader, title=f'level_{i}')
            # train(args, model, device, train_loader, optimizer, epoch, title=lvl_dir)
            # new_test_loss = test(model, device, test_loader, title=lvl_dir)
            if new_test_loss > test_loss and epoch > 15:
                break
            test_loss = new_test_loss
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "{lvl_dir}mnist_cnn.pt")


if __name__ == '__main__':
    main()
