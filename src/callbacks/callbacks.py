import matplotlib.pyplot as plt
from torch import Tensor
import torch
from matplotlib.figure import Figure
import numpy as np

def plot_images(samples : Tensor, height : int | None = None, width : int | None = None) -> Figure:
    # assume samples have shape (k, c, h, w)
    # and have values between -1 and 1
    if height is None and width is None:
        k = int(samples.size(0) ** 0.5)
        height = width = k

    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    cmap = 'gray' if samples.shape[1] == 1 else None
    samples = samples.permute(0, 2, 3, 1)
    fig, axs = plt.subplots(height, width, figsize=(width*5, height*5), dpi=300)
    axs : list[plt.Axes]
    for i in range(height):
        for j in range(width):
            ax = axs[i, j] if height > 1 else axs[j]
            ax.imshow(samples[i * height + j], cmap=cmap)
            ax.axis('off')
    return fig

def plot_points(points : list[Tensor], keys : list[str], colors : list[str]) -> Figure:
    fig, ax = plt.subplots()
    for point, color, key in zip(points, colors, keys):
        dim = point.dim()
        if dim == 2:
            # points have shape (n, 2)
            ax.scatter(point[:, 0], point[:, 1], c=color, label=key, s=10)
        elif dim == 3:
            # points have shape (num_steps, n, 2)
            # visualize the trajectory for each point
            # only visualize some of the trajectory
            n_points = 7
            indices = torch.linspace(0, point.size(0) - 1, n_points).round().to(torch.int64)
            point = point[indices]

            batch_size = point.size(1)
            for i in range(batch_size):
                label = key if i == 0 else None
                # only visualize some of the points
                ax.scatter(point[:, i, 0], point[:, i, 1], c=color, label=label, s=1, alpha=0.3)
                
    ax.legend()
    return fig

def visualize_encodings(encodings : Tensor) -> tuple[Figure, Figure]:
    while encodings.dim() < 3:
        encodings = encodings.unsqueeze(0)
    while encodings.dim() > 3:
        encodings = encodings.flatten(0, 1)
    
    n_channels = encodings.size(0)
    v_min, v_max = encodings.min(), encodings.max()
    fig, axs = plt.subplots(1, n_channels, figsize=(n_channels*5, 5))
    axs : list[plt.Axes]
    axs = np.atleast_1d(axs)
    norm = plt.Normalize(vmin=v_min, vmax=v_max)
    cmap = 'gray' if n_channels == 1 else 'viridis'
    for i in range(n_channels):
        im = axs[i].imshow(encodings[i], cmap=cmap, norm=norm)
        axs[i].axis('off')
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Intensity')
    
    return fig

def plot_histogram(data : Tensor) -> Figure:   
    fig, ax = plt.subplots()
    ax.hist(data.flatten(), bins=100, density=True)
    return fig

def plot_graph(y, x = None, title = None, xlabel = None, ylabel = None):
    fig, _ = plt.subplots()
    if x is None:
        x = range(len(y))
    plt.plot(x, y)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    return fig