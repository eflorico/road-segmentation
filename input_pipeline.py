from config import TRAIN_DATA_DIR, TEST_DATA_DIR, NUM_TRAIN_FILES, NUM_TEST_FILES, IMG_SIZE, TEST_IMG_SIZE, PATCH_SIZE, PATCH_INPUT_SIZE, BATCH_SIZE, BIG_STEP_SIZE, SMALL_STEP_SIZE, AUGMENT_BORDERS, PAD_MODE
from sklearn.utils.extmath import cartesian
from scipy.ndimage.interpolation import rotate
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm


def load_data():
    """
    Loads data from paths specified in config.py and
    Return X_train, Y_train, X_test
    """

    # Load data
    X = np.zeros((NUM_TRAIN_FILES, IMG_SIZE, IMG_SIZE, 3))
    for i in range(1, NUM_TRAIN_FILES+1):
        X[i-1] = mpimg.imread(TRAIN_DATA_DIR + "images/satImage_%.3d.png" % i)

    Y = np.zeros((NUM_TRAIN_FILES, IMG_SIZE, IMG_SIZE))
    for i in range(1, NUM_TRAIN_FILES+1):
        Y[i-1] = mpimg.imread(TRAIN_DATA_DIR + "groundtruth/satImage_%.3d.png" % i)
        
    # Fix distribution of Y [0, .96] -> [0, 1]   
    Y /= Y.max()
        
    X_test = np.zeros((NUM_TEST_FILES, TEST_IMG_SIZE, TEST_IMG_SIZE, 3))
    for i in range(1, NUM_TEST_FILES+1):
        X_test[i-1] = mpimg.imread(TEST_DATA_DIR + "test_%d/test_%d.png" % (i, i))
        
    return X, Y, X_test


def input_pipeline_train(X, Y, reduce_to_patches, two_classes):
    """
    Input pipeline for training. 
    Shuffles data and augments images by random rotation, flipping, brightness/contrast changes and channel shifts.
    X: training images of shape (#images, height, width, 3)
    Y: training ground truth of shape (#images, height, width)
    reduce_to_patches: if true, returns only one scalar for Y for each patch
    two_classes: if true, returns y and 1-y
    Returns generator yielding batches.
    """

    # Add padding to images for larger input patch size and rotation. Mirror image to fill padding
    x_padding = (int(np.ceil(np.sqrt(2))) * PATCH_INPUT_SIZE - PATCH_SIZE) // 2
    X_padded = np.pad(X, ((0,0), (x_padding,x_padding), (x_padding,x_padding), (0,0)), PAD_MODE)
    
    # Also add padding to ground truth for rotation. Fill padding with zeroes
    y_padding = int(np.ceil(np.sqrt(2) - 1)) * PATCH_SIZE // 2
    Y_padded = np.pad(Y, ((0,0), (y_padding,y_padding), (y_padding,y_padding)), 'constant')

    # Prepare array for batches
    X_batch = np.zeros((BATCH_SIZE, PATCH_INPUT_SIZE, PATCH_INPUT_SIZE, 3))
    
    Y_shape = (BATCH_SIZE,)
    if not reduce_to_patches:
        Y_shape += (PATCH_SIZE, PATCH_SIZE)
    if two_classes:
        Y_shape += (2,)

    Y_batch = np.zeros(Y_shape)

    batch_idx = 0
    
    # Prepare indices of form (i, y, x) for shuffling
    if not AUGMENT_BORDERS:
        # Uniform grid
        indices = cartesian([
            np.arange(X.shape[0]),
            np.arange(0, IMG_SIZE - PATCH_SIZE + 1, BIG_STEP_SIZE),
            np.arange(0, IMG_SIZE - PATCH_SIZE + 1, BIG_STEP_SIZE)
        ])
    else:
        # Augment borders: loop over data to generate indices
        indices = []

        for i in tqdm(range(X.shape[0])):
            for y in range(0, IMG_SIZE - PATCH_SIZE + 1, SMALL_STEP_SIZE):
                for x in range(0, IMG_SIZE - PATCH_SIZE + 1, SMALL_STEP_SIZE):
                    if x % BIG_STEP_SIZE == 0 and y % BIG_STEP_SIZE == 0:
                        indices.append([i, y, x])
                    elif x % SMALL_STEP_SIZE == 0 and y % SMALL_STEP_SIZE == 0:
                        # Patch is at border if there are both road and non-road pixels
                        is_border = Y[i, y:y+PATCH_SIZE, x:x+PATCH_SIZE].max() >= .9 and Y[i, y:y+PATCH_SIZE, x:x+PATCH_SIZE].min() <= .1
                        if is_border:
                            indices.append([i, y, x])

        indices = np.array(indices)
        print("Augmented borders, generated %d patches" % indices.shape[0])

    # Class weights for imbalanced data, obtained by sampling from 20 batches
    W_batch = np.zeros((BATCH_SIZE,))
    num_ones, num_total = 0., 20 * BATCH_SIZE
    for j in range(num_total):
        i, y, x = indices[j]
        if Y[i, y:y+PATCH_SIZE, x:x+PATCH_SIZE].max() > .5:
            num_ones += 1

    class_weights = np.array([
        num_ones,
        num_total - num_ones,
    ]) / num_total

    print("Class weights: non-road %.3f, road %.3f" % (class_weights[0], class_weights[1]))

    # Infinite epochs
    while True:
        np.random.shuffle(indices)
        
        for j in range(indices.shape[0]):
            i, y, x = indices[j]
        
            # Calculate maximum possible rotation at this place
            available_space = PATCH_SIZE + 2 * min(x, y, IMG_SIZE - x - PATCH_SIZE, IMG_SIZE - y - PATCH_SIZE)
            if available_space >= np.sqrt(2) * PATCH_SIZE:
                max_angle = 45
            else:
                max_angle = np.arcsin(available_space / np.sqrt(2) / IMG_SIZE) / np.pi * 360

            # Crop out patch with necessary padding p
            x_patch = X_padded[i, y:y+PATCH_SIZE+2*x_padding, x:x+PATCH_SIZE+2*x_padding]
            y_patch = Y_padded[i, y:y+PATCH_SIZE+2*y_padding, x:x+PATCH_SIZE+2*y_padding]

            # Rotate
            angle = np.random.rand() * max_angle
            x_patch = rotate(x_patch, angle, order=1, reshape=False)
            y_patch = rotate(y_patch, angle, order=1, reshape=False)

            # Crop back to desired patch size
            x_pad_delta = x_padding - (PATCH_INPUT_SIZE - PATCH_SIZE) // 2
            x_patch = x_patch[x_pad_delta:-x_pad_delta, x_pad_delta:-x_pad_delta]
            y_patch = y_patch[y_padding:-y_padding, y_padding:-y_padding]

            # Flip
            if np.random.rand() > .5:
                x_patch, y_patch = x_patch[::-1, :], y_patch[::-1, :]
            if np.random.rand() > .5:
                x_patch, y_patch = x_patch[:, ::-1], y_patch[:, ::-1]
                
            # Brightness & contrast
            contrast = np.random.rand() * .6 + .7
            brightness = np.random.rand() * .2 - .1 + np.random.rand() * .1 - .05
            x_patch = np.minimum(np.maximum(contrast * (x_patch - .5) + .5 + brightness, 0), 1)           

            # Write to batch
            X_batch[batch_idx] = x_patch

            if reduce_to_patches and not two_classes:
                Y_batch[batch_idx] = y_patch.mean()
            elif reduce_to_patches and two_classes:
                Y_batch[batch_idx, 1] = y_patch.mean()
                Y_batch[batch_idx, 0] = 1 - Y_batch[batch_idx, 1]
            elif not reduce_to_patches and not two_classes:
                Y_batch[batch_idx] = y_patch
            elif not reduce_to_patches and two_classes:
                Y_batch[batch_idx, :, :, 1] = y_patch
                Y_batch[batch_idx, :, :, 0] = 1 - y_patch

            W_batch[batch_idx] = class_weights[int(np.round(y_patch.mean()))]

            # Yield batch when full. Make sure to set up a new numpy array
            batch_idx = (batch_idx + 1) % BATCH_SIZE
            if batch_idx == 0: 
                yield X_batch, Y_batch, W_batch
                W_batch = np.zeros((BATCH_SIZE,))

                X_batch = np.zeros((BATCH_SIZE, PATCH_INPUT_SIZE, PATCH_INPUT_SIZE, 3))
                Y_batch = np.zeros(Y_shape)

                    
                    
def input_pipeline_test(X, Y, reduce_to_patches, two_classes):
    """
    Input pipeline for testing.
    Does not shuffle or augment data.
    X: images of shape (#images, height, width, 3)
    Y: ground truth of shape (#images, height, width), or None
    Returns generator yielding batches.
    """

    # Pad images for larger input patch size. Mirror image to fill padding
    x_padding = (PATCH_INPUT_SIZE - PATCH_SIZE) // 2
    X_padded = np.pad(X, ((0,0), (x_padding,x_padding), (x_padding,x_padding), (0,0)), 'reflect')

    X_batch = np.zeros((BATCH_SIZE, PATCH_INPUT_SIZE, PATCH_INPUT_SIZE, 3))

    Y_shape = (BATCH_SIZE,)
    if not reduce_to_patches:
        Y_shape += (PATCH_SIZE, PATCH_SIZE)
    if two_classes:
        Y_shape += (2,)

    Y_batch = np.zeros(Y_shape)

    batch_idx = 0
    
    # Infinite epochs
    while True:
        # Loop over images, y and x axis
        for i in range(X.shape[0]):
            for y in range(0, X.shape[1] - PATCH_SIZE + 1, PATCH_SIZE):
                for x in range(0, X.shape[1] - PATCH_SIZE + 1, PATCH_SIZE):
                    # Crop out image
                    X_batch[batch_idx] = X_padded[i, y:y+PATCH_INPUT_SIZE, x:x+PATCH_INPUT_SIZE]

                    if Y is not None:
                        if reduce_to_patches and not two_classes:
                            Y_batch[batch_idx] = Y[i, y:y+PATCH_SIZE, x:x+PATCH_SIZE].mean()
                        elif reduce_to_patches and two_classes:
                            Y_batch[batch_idx, 1] = Y[i, y:y+PATCH_SIZE, x:x+PATCH_SIZE].mean()
                            Y_batch[batch_idx, 0] = 1 - Y_batch[batch_idx, 1]
                        elif not reduce_to_patches and not two_classes:
                            Y_batch[batch_idx] = Y[i, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                        elif not reduce_to_patches and two_classes:
                            Y_batch[batch_idx, :, :, 1] = Y[i, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                            Y_batch[batch_idx, :, :, 0] = 1 - Y_batch[batch_idx, :, :, 1]

                    # Yield batch when full. Make sure to set up a new numpy array
                    batch_idx = (batch_idx + 1) % BATCH_SIZE
                    if batch_idx == 0: 
                        if Y is not None: yield X_batch.copy(), Y_batch.copy()
                        else: yield X_batch.copy()


def sample_class_weights(X, Y):
    """
    Samples class weights from training data by running the full input pipeline on them
    and sampling from 20 batches.
    Returns np array with weights for classes 0 (not road) and 1 (road)
    """
    gen = input_pipeline_train(X, Y)
    num_ones, num_total = 0., 20 * BATCH_SIZE
    for i in tqdm(range(20)):
        X_batch, Y_batch = next(gen)
        num_ones += Y_batch[:].sum()

    return np.array([
        num_ones,
        num_total - num_ones,
    ]) / num_total
