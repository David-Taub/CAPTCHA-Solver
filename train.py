import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import CaptchaDataset


def get_output_shape(layers, input_size, verbose=False):
    head_layer = torch.rand(*input_size)
    if verbose:
        print(f'Input size: {input_size} [{np.prod(input_size):,}]')
    for layer in layers:
        head_layer = layer(head_layer)
        if verbose:
            print(
                f'Layer {layer.__class__.__name__} size: {tuple(head_layer.shape)} [{np.prod(head_layer.shape):,}]')
    return np.prod(tuple(head_layer.shape))


class Net(nn.Module):
    def __init__(self, image_width, image_height, text_length, alphabet_size):
        super(Net, self).__init__()
        expected_input_size = (1, 3, image_height, image_width)
        image_width * image_height * 128 // 2
        self.text_length = text_length
        self.alphabet_size = alphabet_size
        self.layers = nn.ModuleList()
        # self.layers.append(nn.AvgPool2d(2))
        # self.layers.append(nn.Dropout(0.5))

        self.layers.append(nn.Conv2d(3, 32, 3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(2))

        self.layers.append(nn.Conv2d(32, 64, 3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Dropout(0.5))

        self.layers.append(nn.Conv2d(64, 128, 3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(2))

        # self.layers.append(nn.Conv2d(64, 128, 3))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.MaxPool2d(2))

        # self.layers.append(nn.Conv2d(64, 128, 3))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.MaxPool2d(2))

        # self.layers.append(nn.Conv2d(128, 256, 3))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.MaxPool2d(2))

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Dropout(0.5))
        self.layers.append(nn.Linear(get_output_shape(self.layers, expected_input_size), 512))
        # self.layers.append(nn.Linear(get_output_shape(self.layers, expected_input_size), 1024))
        # self.layers.append(nn.MaxPool1d(2))
        self.layers.append(nn.Linear(get_output_shape(self.layers, expected_input_size),
                                     self.text_length * self.alphabet_size))
        self.layers.append(nn.Unflatten(1, (self.alphabet_size, self.text_length)))
        self.layers.append(nn.LogSoftmax(1))
        get_output_shape(self.layers, expected_input_size, True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch:3d} [{batch_idx * len(data):5d}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):3.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break


def test(model, device, test_loader):
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
    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct:.1f}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.1f}%)')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = CaptchaDataset.CaptchaDataset(r'data\train')
    dataset2 = CaptchaDataset.CaptchaDataset(r'data\test')
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    dataset1
    model = Net(image_width=dataset1.image_width, image_height=dataset1.image_height,
                text_length=dataset1.text_length, alphabet_size=len(dataset1.alphabet_dict)).to(device)
    print(sum([len(p) for p in model.parameters()]))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
