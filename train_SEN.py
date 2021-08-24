import sys
import warnings
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from DataLoad import DataGenerator
import argparse
import numpy as np
import losses
from model import Net, SpatialTransform, CompositionTransform
from torch.autograd import Variable
from tools import generate_grid
import glob

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='param')
parser.add_argument('--iterations', default=150001, type=int)
parser.add_argument('--loss_name', default='NCC', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--model_dir', default="./Model/", type=str)
parser.add_argument('--range_flow', default=100.0, type=float)
parser.add_argument('--reg_smooth', default=3, type=float)  # 1 for loss_smooth NCC
parser.add_argument('--vol_shape', nargs='+', type=int, default=[96, 112, 96])
parser.add_argument('-train_path', default='/home/mamingrui/data/data_AAAI/', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--checkpoints', default=10000, type=int)
parser.add_argument('--check_point', default='./Model/SEN/', type=str)

args = parser.parse_args()


# torch.backends.cudnn.benchmark = True


def train():
    grid = generate_grid(vol_orig_shape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()

    model = Net(len(vol_orig_shape)).cuda()
    com = CompositionTransform().cuda()
    transform = SpatialTransform().cuda()

    train_set = DataGenerator(trainset, train_path, batch_size)
    trainset_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                 pin_memory=True, drop_last=False)
    loss_fun = losses.NCC()

    Grad_loss = losses.smoothloss

    opt = Adam(model.parameters(), lr=lr)

    data_size = len(train_set)
    print("Data size is {}. ".format(data_size))

    step = 0
    lossall = np.zeros((5, iterations))

    while step <= iterations:
        for X, Y in trainset_loader:
            X_cuda = X.unsqueeze(1).cuda()
            Y_cuda = Y.unsqueeze(1).cuda()
            X = torch.cat([X_cuda, Y_cuda], 1)

            Fxy, Fyx = model(X)

            X_Y_flow = com(Fxy, -Fyx, grid, range_flow)
            Y_X_flow = com(Fyx, -Fxy, grid, range_flow)

            X_Y = transform(X_cuda, X_Y_flow.permute(0, 2, 3, 4, 1) * range_flow, grid)
            Y_X = transform(Y_cuda, Y_X_flow.permute(0, 2, 3, 4, 1) * range_flow, grid)

            loss_sim_1 = loss_fun(X_Y, Y_cuda)
            loss_sim_2 = loss_fun(Y_X, X_cuda)
            loss_grad_1 = Grad_loss(X_Y_flow * range_flow)
            loss_grad_2 = Grad_loss(Y_X_flow * range_flow)
            loss = loss_sim_1 + loss_sim_2 + reg_smooth * (loss_grad_1 + loss_grad_2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            lossall[:, step] = np.array(
                [loss.item(), loss_sim_1.item(), loss_sim_2.item(), loss_grad_1.item(), loss_grad_2.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_direction "{2:.4f}" - sim_inverse "{3:4f}" - grad_direct "{4:.4f}" - grad_inverse "{5:.10f}"'.format(
                    step, loss.item(), loss_sim_1.item(), loss_sim_2.item(), loss_grad_1.item(), loss_grad_2.item()))
            sys.stdout.flush()

            if (step % checkpoints) == 0:
                modelname = model_dir + 'SEN_' + str(step) + '.pth.tar'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + 'SEN_' + str(step) + '.npy', lossall)
            step += 1

            if step > iterations:
                break

    np.save(model_dir + '/loss_SEN.npy', lossall)


if __name__ == '__main__':
    iterations = args.iterations
    train_path = args.train_path
    lr = args.lr
    range_flow = args.range_flow
    reg_smooth = args.reg_smooth
    batch_size = args.batch_size
    checkpoints = args.checkpoints
    model_dir = args.model_dir
    vol_orig_shape = args.vol_shape
    dataset = args.train_path

    trainset = glob.glob(dataset+'*.npy')
    print(trainset)
    train()
