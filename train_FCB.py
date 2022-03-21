import sys
import warnings
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse
import numpy as np
import losses
from model import Net, FoldingCorrection
from torch.autograd import Variable
from tools import generate_grid
import glob
from DataLoad import BrainDataGenerator as DataGenerator

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='param')
parser.add_argument('--iterations', default=160001, type=int)
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--model_dir', default="./Model/", type=str)
parser.add_argument('--range_flow', default=100.0, type=float)
parser.add_argument('--reg_smooth', default=3, type=float)
parser.add_argument('--vol_shape', nargs='+', type=int, default=[96, 112, 96])
parser.add_argument('-train_path', default='./Data/data/', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--checkpoints', default=2000, type=int)
args = parser.parse_args()


def set_param(vol_shape):
    """
    vol_shape: input image shape [DHW] or [HW].
    :return: alpha for Jacobian regularization and beta for balance.
    """
    if len(vol_shape) == 2:
        lambda1 = 40
        lambda2 = -1
    elif len(vol_shape) == 3:
        lambda1 = 50000
        lambda2 = -0.01
    else:
        raise ValueError('Parameters setting need the correct input image shape, [DHW] or [HW]')
    return lambda1, lambda2


def train():
    lambda1, lambda2 = set_param(vol_orig_shape)

    grid = generate_grid(vol_orig_shape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()

    model = Net(len(vol_orig_shape)).cuda()
    com = CompositionTransform().cuda()

    FCB = FoldingCorrection(len(vol_orig_shape)).cuda()

    train_set = DataGenerator(trainset)
    trainset_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                 pin_memory=True, drop_last=False)

    loss_sim2 = losses.MSE().loss
    loss_jac = losses.jac_correct3D
    loss_grad = losses.smoothloss3D
    opt = Adam(model.parameters(), lr=lr)

    data_size = len(train_set)
    print("Data size is {}. ".format(data_size))

    n = 1
    for i in vol_orig_shape:
        n *= i

    step = 0
    lossall = np.zeros((5, iterations))

    while step <= iterations:
        for X, Y in trainset_loader:
            X_cuda = X.unsqueeze(1).cuda()
            Y_cuda = Y.unsqueeze(1).cuda()
            X = torch.cat([X_cuda, Y_cuda], 1)

            with torch.no_grad():
                Fxy, Fyx = model(X)
                F_X_Y = com(Fxy, -Fyx, grid, range_flow)

            diff = FCB(F_X_Y * range_flow)

            adjusted_flow = (F_X_Y * range_flow - diff)

            loss1 = loss_sim2(F_X_Y * range_flow, adjusted_flow)

            loss2 = lambda1 * loss_jac(adjusted_flow.permute(0, 2, 3, 4, 1), grid, n)

            loss3 = lambda2 * torch.log(loss_grad(diff))

            loss = loss1 + loss2 + loss3

            opt.zero_grad()
            loss.backward()
            opt.step()
            lossall[:, step] = np.array(
                [step, loss.item(), loss1.item(), loss2.item(), loss3.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - displacement_sim "{2:.10f}" - Jac_reg "{3:.4f}" - grad_balance "{4:.10f}"'.format(
                    step, loss.item(), loss1.item(), loss2.item(), loss3.item()))
            sys.stdout.flush()

            if (step % checkpoints) == 0:
                modelname = model_dir + 'FCB_' + str(step) + '.pth.tar'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + 'FCB_' + str(step) + '.npy', lossall)
            step += 1

            if step > iterations:
                break

    np.save(model_dir + '/loss_FCB.npy', lossall)


if __name__ == '__main__':
    iterations = args.iterations
    lr = args.lr
    range_flow = args.range_flow
    reg_smooth = args.reg_smooth
    batch_size = args.batch_size
    checkpoints = args.checkpoints
    model_dir = args.model_dir
    vol_orig_shape = args.vol_shape
    dataset = args.train_path

    if len(vol_orig_shape) == 2:
        from model import CompositionTransform2D as CompositionTransform

        loss_jac = losses.jac_correct2D
        loss_grad = losses.smoothloss2D
    elif len(vol_orig_shape) == 3:
        from model import CompositionTransform3D as CompositionTransform

        loss_fun = losses.jac_correct3D
        loss_grad = losses.smoothloss3D

    else:
        raise ValueError('Input image shape must be [DHW] for 3D or [HW] for 2D.')

    trainset = glob.glob(dataset + '*.npy')
    print(trainset)
    train()
