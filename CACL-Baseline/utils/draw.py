import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import copy
import torch


def classification_map(map, H, W, dpi, savePath):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(W * 2.0 / dpi, H * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    # ax.imshow(map, aspect='normal')
    fig.savefig(savePath, dpi=dpi)
    plt.close(fig)
    return 0

def draw_pred_figure(pred_results, H, W, save_ROOT, datasetName, known_num):
    pred_results += 1
    y = np.zeros((pred_results.shape[0], 3))
    for item in range(0, known_num + 2):
        index = (pred_results == item).cpu().numpy()
        if item > known_num:
            y[index] = np.array([255, 255, 255]) / 255.  # White
        else:
            if item == 0:
                y[index] = np.array([0, 0, 0]) / 255.  # Black
            if item == 1:
                y[index] = np.array([140, 67, 46]) / 255.  # green
            if item == 2:
                y[index] = np.array([0, 0, 255]) / 255.  # Blue
            if item == 3:
                y[index] = np.array([255, 100, 0]) / 255.  # Yellow
            if item == 4:
                y[index] = np.array([0, 255, 123]) / 255.  # Cyan
            if item == 5:
                y[index] = np.array([164, 75, 155]) / 255.  # Magenta
            if item == 6:
                y[index] = np.array([101, 174, 255]) / 255.  # Magenta1
            if item == 7:
                y[index] = np.array([118, 254, 172]) / 255.  # Orange
            if item == 8:
                y[index] = np.array([60, 91, 112]) / 255.  # pink
            if item == 9:
                y[index] = np.array([213, 26, 33]) / 255.  # DarkViolet
            if item == 10:
                y[index] = np.array([255, 255, 125]) / 255.  # Salmon
            if item == 11:
                y[index] = np.array([255, 0, 255]) / 255.  # Magenta
            if item == 12:
                y[index] = np.array([100, 0, 255]) / 255.  # purple
            if item == 13:
                y[index] = np.array([0, 172, 254]) / 255.  # NavyBlue
            if item == 14:
                y[index] = np.array([0, 255, 0]) / 255.  # Orange
            if item == 15:
                y[index] = np.array([171, 175, 80]) / 255.  # red
            if item == 16:
                y[index] = np.array([101, 193, 60]) / 255.  # DarkRed
    y_re = np.reshape(y, (H, W, 3))
    dpi = 1600
    classification_map(y_re, H, W, dpi, save_ROOT + "/{}_pred.png".format(datasetName))
