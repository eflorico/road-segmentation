import keras.backend as K
from config import PATCH_SIZE

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    y_true, y_pred = K.round(y_true), K.round(y_pred)
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r + K.epsilon())

def acc(y_true, y_pred):
    y_true, y_pred = K.round(y_true), K.round(y_pred)
    return K.sum(K.cast(K.equal(y_true, y_pred), 'float32')) / K.cast(K.shape(y_true)[0], 'float32')

def to_patches(y):
    """
    Flattens two-class pixels into patches by averaging 
    """
    return K.mean(K.reshape(y[:,:,:,1], (-1, PATCH_SIZE**2)), axis=1)

def f1_pat(y_true, y_pred):
    """
    Per-patch F1 score for two-class pixels
    """
    return f1(to_patches(y_true), to_patches(y_pred))

def acc_pat(y_true, y_pred):
    """
    Per-patch accuracy for two-class pixels
    """
    return acc(to_patches(y_true), to_patches(y_pred))

def f1_pix(y_true, y_pred):
    """
    Per-pixel F1 score for two-class pixels.
    """
    return f1(K.flatten(y_true[:,:,:,1]), K.flatten(y_pred[:,:,:,1]))

def acc_pix(y_true, y_pred):
    """
    Per-pixel accuracy for two-class pixels.
    """
    return acc(K.flatten(y_true[:,:,:,1]), K.flatten(y_pred[:,:,:,1]))
