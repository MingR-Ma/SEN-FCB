import torch
import argparse
import numpy as np
from model import Net, SpatialTransform, CompositionTransform, Auot_Folding_Adjustment
from torch.autograd import Variable
from tools import generate_grid

parser = argparse.ArgumentParser()
parser.add_argument("--SEN_model_path", type=str, default='./Model/weights/SEN.pth.tar')
parser.add_argument("--AFT_model_path", type=str, default='./Model/weights/AFT.pth')
parser.add_argument("--fixed", type=str, default='./Data/A.npy')
parser.add_argument("--moving", type=str, default='./Data/B.npy')
parser.add_argument('--Result_dir', default='./Data/Result/', type=str)
parser.add_argument('--range_flow', default=100, type=int)
parser.add_argument('--vol_shape', nargs='+', type=int, default=[96, 112, 96])

parser.add_argument('--mode', default='SEN', type=str,
                    help='Switch  SEN mode or SEN+AFT mode')

args = parser.parse_args()


def test_SEN(A, B, range_flow, model_path, vol_orig_shape, result_dir):
    grid = generate_grid(vol_orig_shape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()

    model = Net(len(vol_orig_shape)).cuda()
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    com = CompositionTransform().cuda()
    transform = SpatialTransform().cuda()

    A = torch.Tensor(np.load(A)).unsqueeze(0).unsqueeze(0).cuda()
    B = torch.Tensor(np.load(B)).unsqueeze(0).unsqueeze(0).cuda()

    X = torch.cat([A, B], 1)

    FAB, FBA = model(X)

    BA_flow = com(FBA, -FAB, grid, range_flow)
    AB_flow = com(FAB, -FBA, grid, range_flow)

    warped_AB = transform(A, AB_flow.permute(0, 2, 3, 4, 1) * range_flow, grid)
    warped_BA = transform(B, BA_flow.permute(0, 2, 3, 4, 1) * range_flow, grid)

    warped_AB = warped_AB.squeeze(0).squeeze(0).detach().cpu().numpy()
    warped_BA = warped_BA.squeeze(0).squeeze(0).detach().cpu().numpy()

    np.save(result_dir + 'warped_A', warped_AB)
    np.save(result_dir + 'warped_B', warped_BA)


def test_SEN_AFT(A, B, range_flow, model_path1, model_path2, vol_orig_shape, result_dir):
    grid = generate_grid(vol_orig_shape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()

    model1 = Net(len(vol_orig_shape)).cuda()
    model1.load_state_dict(torch.load(model_path1))
    com = CompositionTransform().cuda()
    transform = SpatialTransform().cuda()

    model2 = Auot_Folding_Adjustment().cuda()
    model2.load_state_dict(torch.load(model_path2))

    A = torch.Tensor(np.load(A)).unsqueeze(0).unsqueeze(0).cuda()
    B = torch.Tensor(np.load(B)).unsqueeze(0).unsqueeze(0).cuda()

    X = torch.cat([A, B], 1)

    FAB, FBA = model1(X)

    BA_flow = com(FBA, -FAB, grid, range_flow)
    AB_flow = com(FAB, -FBA, grid, range_flow)

    delta_AB = model2(AB_flow * range_flow)
    AB_flow = AB_flow * range_flow - delta_AB

    delta_BA = model2(BA_flow * range_flow)
    BA_flow = BA_flow * range_flow - delta_BA

    warped_AB = transform(A, AB_flow.permute(0, 2, 3, 4, 1), grid)
    warped_BA = transform(B, BA_flow.permute(0, 2, 3, 4, 1), grid)

    warped_AB = warped_AB.squeeze(0).squeeze(0).detach().cpu().numpy()
    warped_BA = warped_BA.squeeze(0).squeeze(0).detach().cpu().numpy()

    np.save(result_dir + 'warped_A_AFT', warped_AB)
    np.save(result_dir + 'warped_B_AFT', warped_BA)


if __name__ == '__main__':
    range_flow = args.range_flow
    SEN_model_path = args.SEN_model_path
    AFT_model_path = args.AFT_model_path
    vol_orig_shape = args.vol_shape
    result_dir = args.Result_dir
    A = args.fixed
    B = args.moving

    mode = args.mode

    if mode == 'SEN':
        test_SEN(A, B, range_flow, SEN_model_path, vol_orig_shape, result_dir)
    elif mode == 'SEN+AFT':
        test_SEN_AFT(A, B, range_flow, SEN_model_path, AFT_model_path, vol_orig_shape, result_dir)
    else:
        raise Exception("Please check your 'mode' command.")
