from config import RANDOM_SEED, IMG_SIZE, PATCH_SIZE, BATCH_SIZE
from input_pipeline import input_pipeline_train, input_pipeline_test, load_data
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import models


# Choose model
print("Compiling model...")
model = models.convnetv2()

# Choose filename for saved models here
RUN_NAME = 'conv2'

# Choose checkpoint to start from
WEIGHTS_FILE = None
INITIAL_EPOCH = 0

# Load data
print("Loading data...")
X_train, Y_train, X_test = load_data()

# Train/validate split
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=RANDOM_SEED)

params = models.get_pipeline_params(model)
if params['reduce_to_patches']:
	filename = RUN_NAME + '.{epoch:02d}-{val_acc:.4f}-{val_f1:.4f}.hdf5'
else:
	filename = RUN_NAME + '.{epoch:02d}-{val_acc_pat:.4f}-{val_f1_pat:.4f}.hdf5'

checkpoint = ModelCheckpoint(filename, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

# Prepare pipeline
print("Preparing pipeline...")
train_gen = input_pipeline_train(X_train, Y_train, **params)
val_gen = input_pipeline_test(X_valid, Y_valid, **params)

train_epoch_size = X_train.shape[0] * (IMG_SIZE // PATCH_SIZE) ** 2 // BATCH_SIZE
val_epoch_size = X_valid.shape[0] * (IMG_SIZE // PATCH_SIZE) ** 2 // BATCH_SIZE

if WEIGHTS_FILE is not None:
	model.load_weights(WEIGHTS_FILE)

# Train model
model.fit_generator(train_gen,
                    validation_data=val_gen,
                    steps_per_epoch=train_epoch_size,
                    validation_steps=val_epoch_size,
                    epochs=30,
                    initial_epoch=INITIAL_EPOCH,
                    verbose=1,
                    max_q_size=50,
                    workers=6,
                    pickle_safe=True,
                    callbacks=[checkpoint])
