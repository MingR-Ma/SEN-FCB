import torch
import argparse
import numpy as np
from model import Net, SpatialTransform3D as SpatialTransform, CompositionTransform3D as CompositionTransform, FoldingCorrection
from torch.autograd import Variable
from tools import generate_grid
from evaluation import jacobian_determinant, dice

parser = argparse.ArgumentParser()
parser.add_argument("--SEN_model_path", type=str, default='./Model/weights/SEN.pth')
parser.add_argument("--FCB_model_path", type=str, default='./Model/weights/FCB.pth')
parser.add_argument("--fixed", type=str, default='./Data/validation_set/atlases/OASIS_OAS1_0022_MR1.npy')
parser.add_argument("--moving", type=str, default='./Data/validation_set/valsets/OASIS_OAS1_0017_MR1.npy')
parser.add_argument("--fixed_label", type=str, default='./Data/validation_set/atlases_label/OASIS_OAS1_0022_MR1_label.npy')
parser.add_argument("--moving_label", type=str, default='./Data/validation_set/valsets_label/OASIS_OAS1_0017_MR1_label.npy')
parser.add_argument('--Result_dir', default='./Data/Result/', type=str)
parser.add_argument('--range_flow', default=100, type=int)
parser.add_argument('--vol_shape', nargs='+', type=int, default=[96, 112, 96])
parser.add_argument('--evaluation', default=True)
parser.add_argument('--mode', default='SEN', type=str,
                    help="Switch 'SEN' mode or 'SEN+FCB' mode")

args = parser.parse_args()


def test_SEN(A, B, range_flow, model_path, vol_orig_shape, result_dir):
    grid = generate_grid(vol_orig_shape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).float()

    model = Net(len(vol_orig_shape))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    com = CompositionTransform()
    transform = SpatialTransform()

    A = torch.Tensor(np.load(A)).unsqueeze(0).unsqueeze(0) #moving
    B = torch.Tensor(np.load(B)).unsqueeze(0).unsqueeze(0) #fixed

    X = torch.cat([A, B], 1)

    FAB, FBA = model(X)

    BA_flow = com(FBA, -FAB, grid, range_flow)
    AB_flow = com(FAB, -FBA, grid, range_flow)
    warped_AB = transform(A, AB_flow.permute(0, 2, 3, 4, 1) * range_flow, grid)
    warped_BA = transform(B, BA_flow.permute(0, 2, 3, 4, 1) * range_flow, grid)

    warped_AB = warped_AB.squeeze(0).squeeze(0).detach().numpy()
    warped_BA = warped_BA.squeeze(0).squeeze(0).detach().numpy()

    np.save(result_dir + 'warped_A', warped_AB)
    np.save(result_dir + 'warped_B', warped_BA)

    if evaluation:
        return BA_flow, grid


def test_SEN_FCB(A, B, range_flow, model_path1, model_path2, vol_orig_shape, result_dir):
    grid = generate_grid(vol_orig_shape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).float()

    model1 = Net(len(vol_orig_shape))
    model1.load_state_dict(torch.load(model_path1, map_location='cpu'))
    com = CompositionTransform()
    transform = SpatialTransform()

    model2 = FoldingCorrection(len(vol_orig_shape))
    model2.load_state_dict(torch.load(model_path2, map_location='cpu'))

    A = torch.Tensor(np.load(A)).unsqueeze(0).unsqueeze(0)
    B = torch.Tensor(np.load(B)).unsqueeze(0).unsqueeze(0)

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

    np.save(result_dir + 'warped_A_FCB', warped_AB)
    np.save(result_dir + 'warped_B_FCB', warped_BA)

    if evaluation:
        return BA_flow, grid


if __name__ == '__main__':
    range_flow = args.range_flow
    SEN_model_path = args.SEN_model_path
    FCB_model_path = args.FCB_model_path
    vol_orig_shape = args.vol_shape
    result_dir = args.Result_dir
    A = args.fixed
    B = args.moving
    evaluation = args.evaluation

    mode = args.mode

    if mode == 'SEN':
        output = test_SEN(A, B, range_flow, SEN_model_path, vol_orig_shape, result_dir)
        if evaluation:
            A_label = args.fixed_label
            A_label = np.load(A_label)
            B_label = args.moving_label
            B_label_tensor = torch.Tensor(np.load(B_label)).unsqueeze(0).unsqueeze(0)

            labels = list(np.unique(A_label))[1:]
            flow_BA, grid = output
            transform = SpatialTransform()
            warped_B_label = transform(B_label_tensor, (range_flow * flow_BA).permute(0, 2, 3, 4, 1), grid, 'nearest')
            warped_B_label = warped_B_label.squeeze(0).squeeze(0).detach().numpy()
            dice_score = np.sum(dice(warped_B_label, A_label, labels)) / len(labels)
            np.save(result_dir + 'warped_B_label', warped_B_label)
            n_jac_det = np.sum(
                jacobian_determinant(range_flow * flow_BA.permute(0, 2, 3, 4, 1).squeeze(0).detach()) <= 0)

            flow_BA=range_flow*flow_BA.permute(0,2,3,4,1).squeeze(0).detach().cpu().numpy()
            np.save(result_dir + 'warped_BA_flow', flow_BA)

            print(f"The warp DSC score is {dice_score} with {n_jac_det} Jacobian determinants.")

    elif mode == 'SEN+FCB':
        output = test_SEN_FCB(A, B, range_flow, SEN_model_path, FCB_model_path, vol_orig_shape, result_dir)
        if evaluation:
            A_label = args.fixed_label
            A_label = np.load(A_label)
            B_label = args.moving_label
            B_label_tensor = torch.Tensor(np.load(B_label)).unsqueeze(0).unsqueeze(0)

            labels = list(np.unique(A_label))[1:]
            flow_BA, grid = output
            transform = SpatialTransform()
            warped_B_label = transform(B_label_tensor, flow_BA.permute(0, 2, 3, 4, 1), grid, 'nearest')
            warped_B_label = warped_B_label.squeeze(0).squeeze(0).detach().numpy()
            dice_score = np.sum(dice(warped_B_label, A_label, labels)) / len(labels)
            np.save(result_dir + 'warped_B_label_FCB', warped_B_label)
            n_jac_det = np.sum(jacobian_determinant(flow_BA.permute(0, 2, 3, 4, 1).squeeze(0).detach()) <= 0)
            flow_BA=flow_BA.permute(0,2,3,4,1).squeeze(0).detach().cpu().numpy()
            np.save(result_dir + 'warped_BA_flow_FCB', flow_BA)
            print(f"The warp DSC score is {dice_score} with {n_jac_det} Jacobian determinants.")

    else:
        raise Exception("Please check your 'mode' command.")
