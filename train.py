import string
import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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

NET_INPUT_SIZE = list(np.array([100, 600]) // 3)
PADDED_TEXT_INPUT_SIZE = 10
TRAIN_EPOCH = 4096
TEST_EPOCH = 512


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


class Net(pl.LightningModule):
    def __init__(self, input_size, text_length, alphabet_size):
        super(Net, self).__init__()
        expected_input_size = [1, 3] + list(input_size)
        self.input_size = input_size
        self.text_length = text_length
        self.alphabet_size = alphabet_size
        self.t_acc = pl.metrics.Accuracy()
        self.v_acc = pl.metrics.Accuracy()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.layers.append(nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.layers.append(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.layers.append(nn.ReLU())
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

    def training_step(self, batch, batch_idx):
        data, target = batch
        data = data.float()
        target = target.long()
        output = self(data)
        loss = F.nll_loss(output, target)
        self.t_acc(output, target)
        self.log('t_acc', self.t_acc, on_step=True, on_epoch=False, prog_bar=True)
        # self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        data = data.float()
        target = target.long()
        output = self(data)
        val_loss = F.nll_loss(output, target).item()
        self.v_acc(output, target)
        self.log('v_acc', self.v_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('v_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)


def show(captcha_input, text_output, dataset):
    import matplotlib.pyplot as plt
    inv_dict = {v: k for k, v in dataset.alphabet_dict.items()}
    text_output_str = ''.join([inv_dict[i] for i in np.argmax(np.array(text_output[0].cpu()), 0)])
    plt.imshow(np.array(captcha_input.cpu())[0].T)
    plt.title(text_output_str)
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
    img -= 0.5
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
    val_kwargs = {'batch_size': args.test_batch_size,
                  'shuffle': False}
    if use_cuda:
        print('using CUDA')
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    #######################
    # Train
    #######################
    transform = transforms.Compose(
        [
            transforms.Resize(NET_INPUT_SIZE),
            transforms.Lambda(pil_transform),
        ])
    output_alphabet_size = len(string.ascii_lowercase + string.ascii_uppercase + string.digits) + 1
    model = Net(input_size=NET_INPUT_SIZE, text_length=PADDED_TEXT_INPUT_SIZE,
                alphabet_size=output_alphabet_size).to(device)
    print(f'Trainable parameters: {sum([len(p) for p in model.parameters()])}')
    for i, generator in enumerate(generate_captcha.generators):
        early_stop_callback = EarlyStopping(
            monitor='v_loss',
            min_delta=0.00,
            patience=4,
            mode='min'
        )
        trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback], min_epochs=6)
        print(f'\nLevel {i}:')
        if i < 2:
            continue
        train_dataset = captcha_datasets.DynamicCaptchaDataset(generator, transform=transform,
                                                               padded_text_length=PADDED_TEXT_INPUT_SIZE,
                                                               fake_length=TRAIN_EPOCH)
        val_dataset = captcha_datasets.DynamicCaptchaDataset(generator, transform=transform,
                                                             padded_text_length=PADDED_TEXT_INPUT_SIZE,
                                                             fake_length=TEST_EPOCH, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
        trainer.fit(model, train_loader, val_loader)
        if args.save_model:
            torch.save(model.state_dict(), "level_{i}_mnist_cnn.pt")


if __name__ == '__main__':
    main()
