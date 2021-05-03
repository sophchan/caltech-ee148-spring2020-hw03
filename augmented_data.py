import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.ndimage.interpolation import shift

import os

from main import ConvNet, train, test

def shift_image(image_info):
    '''
    Data augmentation: outputs original and two shifted versions of the image.
    The shifting of the image is by 4 pixels either +/- along either the x or y axis
    ''' 
    [image, truth] = image_info
    pixel_shift = 4
    img_shifts = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    output = [image_info]
    image.size()
    image = image.reshape((28, 28))
    image.size()
    for xy in img_shifts:
        x_dir, y_dir = xy
        dx = pixel_shift * x_dir
        dy = pixel_shift * y_dir
        
        shifted_image = shift(image, [dy, dx],cval=0, mode='constant')
        output.append([shifted_image, truth])

    return output
        

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = fcNet().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('data', train=True, #download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),          # Add data augmentation here
                    transforms.RandomAffine(degrees=5, translate=(4/28, 4/28)),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))


    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.

    images = {}
    count = [0]*10
    for i in range(10):
        images[i] = []

    for index, img in enumerate(train_dataset):
        images[int(img[1])].append(index)
        count[int(img[1])] += 1
    
    np.random.seed(2021)
    val_count = np.array(count)*.15
    val_count = val_count.astype(int)
    subset_indices_valid = np.array([])
    for j, k in enumerate(val_count):
        subset_indices_valid = np.concatenate((subset_indices_valid, \
            np.random.choice(images[j], size=k, replace=False)))
    subset_indices_valid = subset_indices_valid.astype(int)
    subset_indices_valid = torch.from_numpy(subset_indices_valid)
    subset_indices_train = [l for l in range(len(train_dataset)) \
         if (l not in subset_indices_valid)]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = ConvNet().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()


