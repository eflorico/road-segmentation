from config import PATCH_SIZE, IMG_SIZE, TEST_IMG_SIZE, BATCH_SIZE, THRESHOLD
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from input_pipeline import input_pipeline_test
import models
import itertools


def join_img(Y_patches, train):
    img_size = IMG_SIZE if train else TEST_IMG_SIZE
    
    Y = np.zeros((img_size, img_size))
    idx = 0
    for y in range(0, img_size - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, img_size - PATCH_SIZE + 1, PATCH_SIZE):
            Y[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = Y_patches[idx]
            idx += 1

    return Y


def plot_img(img):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.imshow(img);
    loc = plticker.MultipleLocator(base=16)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    plt.grid(which='major', axis='both', linestyle='-')

    
def predict_img(model, x, y_true=None):
    patches_per_img = ((x.shape[0] // PATCH_SIZE) ** 2)
    y_true = y_true[np.newaxis, ...] if y_true is not None else None
    
    params = models.get_pipeline_params(model)
    gen = input_pipeline_test(x[np.newaxis, ...], y_true, **params)
    
    patches = list(itertools.islice(gen, 0, patches_per_img // BATCH_SIZE + 2))
    
    if y_true is None:
        x_patches = np.concatenate(patches)
    else:
        x_patches = np.concatenate([ x[0] for x in list(patches) ])
        y_true_patches = np.concatenate([ x[1] for x in list(patches) ])
        if not params['reduce_to_patches']:
            y_true_patches = y_true_patches[..., 1]
        y_true_img = join_img(y_true_patches[:patches_per_img], x.shape[0] == IMG_SIZE)[..., np.newaxis]

    y_patches = model.predict(x_patches[:patches_per_img])
    if not params['reduce_to_patches']:
        y_patches = y_patches[..., 1]
    y_img = join_img(y_patches[:patches_per_img], x.shape[0] == IMG_SIZE)
    y_img = np.minimum(np.maximum(y_img[..., np.newaxis], 0), 1) > THRESHOLD

    if y_true is None:
        colored = np.minimum(x + y_img * [.3,0,0], 1)
    else:
        correct = y_img * y_true_img
        alpha = (1-y_img) * y_true_img
        beta = y_img * (1-y_true_img)
        colored = correct*np.array([0.,.3,0.]) + alpha*np.array([.3,0.,0.]) + beta*np.array([0.,0.,.3])
        colored = np.minimum(x + colored, 1)
    
    return colored
