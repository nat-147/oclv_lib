import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

import wandb


def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # if using convs, use fixed conv backend algorithm
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    fix_randomness(seed=42)

    return


def convert_range(x, old_range, new_range):
    return ((x - old_range[0])/(old_range[1] - old_range[0])) * (new_range[1] - new_range[0]) + new_range[0]


def get_dataset_dataloader(config):
    dataset_args = config['dataset_args']
    dataloader_args = config['dataloader_args']

    dataset_args.update(train=True)
    train_dataset = config['dataset_class'](**dataset_args)

    dataset_args.update(train=False)
    test_dataset = config['dataset_class'](**dataset_args)

    dataloader_args.update(
        shuffle=True,
    )
    train_loader = DataLoader(dataset=train_dataset, worker_init_fn=worker_init_fn, **dataloader_args)

    dataloader_args.update(
        shuffle=False
    )
    test_loader = DataLoader(dataset=test_dataset, worker_init_fn=worker_init_fn, **dataloader_args)


    return train_dataset, test_dataset, train_loader, test_loader


def cosine_dissim(x, y):
    return (1 - nn.CosineSimilarity(dim=-1)(x, y)).clamp_(0.0, 1.0)


def get_i1_i2(loop_path):
    if loop_path.ndim == 4:
        loop_path = loop_path.unsqueeze(0)

    return loop_path[:, :-1, :, :, :], loop_path[:, 1:, :, :, :]


def get_i1_i2_i3(loop_path):
    if loop_path.ndim == 4:
        loop_path = loop_path.unsqueeze(0)

    return loop_path[:, :-1:2], loop_path[:, 1::2], loop_path[:, 2::2]


def mse_loss_fn(input_data, target_data, mask=None, weight=1):
        mse = nn.MSELoss(reduction='none')

        if (input_data.numel() == 0) or (target_data.numel() == 0):
            return 0

        loss = weight * mse(input_data, target_data.view_as(input_data))

        if mask is not None:
            loss = loss[mask]

        return loss.mean()


def plotly_scatter(data, colors, title, x_range=None, y_range=None):
    if colors is None:
        hex_colors = ['#000000' for _ in range(data.shape[0])]
    else:
        colors = np.array(colors)

        hex_colors = ['#%02x%02x%02x' % tuple((rgba * 255).astype(int)[:3]) for rgba in colors]

    if data.shape[1] == 3:
        scatter = go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=hex_colors
            )
        )

        args = dict(
            scene = dict(
                xaxis_title='dim 1',
                yaxis_title='dim 2',
                zaxis_title='dim 3',
            )
        )
    elif data.shape[1] == 2:
        scatter = go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=hex_colors
            )
        )

        args = dict(
            xaxis_title='dim 1',
            yaxis_title='dim 2',

        )
    elif data.shape[1] == 1 or data.ndim == 1:
        data = data.reshape(-1, 1)
        scatter = go.Scatter(
            x=data[:, 0],
            y=np.zeros(data.shape[0]),
            mode='markers',
            marker=dict(
                size=5,
                color=hex_colors
            )
        )

        args = dict(
            xaxis_title='dim 1',
            yaxis_title='dim 2',
        )
    else:
        raise Exception('Unsupported data dimensionality for a plotly scatter plot.')


    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title={'text' : title},
        **args,
    )

    if data.shape[1] == 2 or data.shape[1] == 1:
        if x_range:
            fig.update_xaxes(range=x_range)
        if y_range:
            fig.update_yaxes(range=y_range)

    return fig


def estimate_A_pred_to_true(pred, true):
    """
    true: (T, 2)
    pred: (T, 2)
    returns A: (2, 2) such that pred ≈ true @ A.T
    """
    assert true.shape == pred.shape
    assert true.shape[1] == 2

    # Least squares: true @ A.T = pred
    A_T, _, _, _ = np.linalg.lstsq(pred, true, rcond=None)
    return A_T

def pred_to_true(pred, A_T):
    """
    pred: (T, 2)
    A: (2, 2)
    """
    true_hat = pred @ A_T
    return true_hat
