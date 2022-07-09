'''
Code for question 8
-------------------
Feed True as an argument for one of the four flags --error, --kernels, --matrix, or 
--feature to output the figure(s) for the problems. 

(b) 9 examples where the classifier made a mistake
    --errors True

(c) Display learned kernels from the first convolutional layer
    --kernels True

(d) Generate a confusion matrix
    --matrix True

(e) (i) Visualize the predictions of each of the test images
    (ii) Choose 4 feature images and determine the 8 closest images by Euclidean distance
    --feature True
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.manifold import TSNE

import os

from main import Net 

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def test2(model, device, test_loader, errors=False, kernels=False, matrix=False, feature=False):
    '''
    errors: Visualize classification errors (HW #8b)
    kernels: Visualize kernels in the first layer (HW #8c)
    matrix: Calculate confusion matrix (HW #8d)
    feature: Visualize images w/ tSNE
    '''
    model.eval()    # Set the model to inference mode
    mat = np.zeros((10, 10))
    aggregate_data = torch.Tensor([])
    features = torch.Tensor([])
    labels = torch.Tensor([])
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if feature: 
                model.fc2.register_forward_hook(get_activation('fc2'))
                output = model(data)
                aggregate_data = torch.cat((aggregate_data, data.to(device)))
                features = torch.cat((features, activation['fc2']))
                labels = torch.cat((labels, target))
            else:
                output = model(data)
                if errors: 
                    total = 0
                    for i, j in enumerate(target): 
                        if j != np.argmax(output[i]):
                            plt.imshow(data[i].view(28, 28))
                            plt.axis('off')
                            plt.show()
                            total += 1
                            if total > 9:
                                return
                elif kernels:
                    kernel_lst = model.conv1.weight.cpu().detach().clone()
                    fig = plt.figure()
                    fig.tight_layout()
                    plt.title('Conv1 Kernels')
                    plt.axis('off')
                    # 16 kernels 
                    for ker in range(16):
                        ax = fig.add_subplot(4, 4, ker+1)
                        ax.imshow(kernel_lst[ker][0])
                        plt.axis('off')                
                    plt.show()
                    return 
                elif matrix: 
                    for i, j in enumerate(target): 
                        mat[j, np.argmax(output[i])] += 1
        print(mat.astype(int)) if matrix else None
        if feature: 
            tsne = TSNE(n_components=2).fit_transform(features)
            x = tsne[:, 0]
            y = tsne[:, 1]
            fig, ax = plt.subplots()

            for c in range(10):
                indices = []
                for s, t in enumerate(labels): 
                    indices.append(s) if c == t else None
                x_c = x[indices]
                y_c = y[indices]
                ax.scatter(x_c, y_c)
            plt.axis('off')
            plt.title('2D Visualization of MNIST Set')
            plt.legend(list(range(10)))
            plt.show()

            fig = plt.figure()
            fig.tight_layout()
            plt.axis('off')
            plt.title('Most Similar Images by Euclidean Distance')

            num = 4
            indices = [[], [], [], []]
            for f, vec1 in enumerate(features[0:num]):
                remaining = torch.cat((features[0:f], features[f+1:]))
                indices[f].append(f-1)
                norms = torch.Tensor([])

                for vec2 in remaining: 
                    norms = torch.cat((norms, torch.cdist(vec1.view(1, 64), \
                        vec2.view(1, 64))))

                for i in range(8):
                    new_norms = norms.clone()
                    if len(indices[f]) != 0: 
                        indices[f].sort(reverse=True)
                        for taken in indices[f]: 
                            new_norms = torch.cat((new_norms[0:taken], new_norms[taken+1:]))
                
                    min_norm = torch.min(new_norms)
                    min_norm_idx = (norms == min_norm).nonzero(as_tuple=True)[0]
                    indices[f].append(min_norm_idx)
            
            for s, ind in enumerate(indices): 
                indices[s].sort()

            for l, lst in enumerate(indices):
                for k, img_num in enumerate(indices[l]):
                    ax = fig.add_subplot(num, 9, l*9+k+1)
                    ax.imshow(aggregate_data[img_num+1].view(28, 28))
                    plt.axis('off')
            plt.show()
        return 


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

    parser.add_argument('--errors', action='store_true', default=False,
                        help='display 9 classifier errors')
    parser.add_argument('--kernels', action='store_true', default=False,
                        help='display learned kernels from first convolutional layer')
    parser.add_argument('--matrix', action='store_true', default=False,
                        help='generate a confusion matrix')
    parser.add_argument('--feature', action='store_true', default=False,
                        help='(i) visualize test image predictions (ii) choose \
                            4 feature images and corresponding 8 closest images by \
                                Euclidean distance')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = 'cpu'

    kwargs = {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        test2(model, device, test_loader, errors=args.errors, kernels=args.kernels, \
            matrix=args.matrix, feature=args.feature)

        return

if __name__ == '__main__':
    main()
