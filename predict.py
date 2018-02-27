from config import TEST_IMG_SIZE, PATCH_SIZE, BATCH_SIZE, THRESHOLD, NUM_TEST_FILES
from input_pipeline import load_data, input_pipeline_test
import models

# Choose model
model = models.convnetv2()

# Choose weights file
WEIGHTS_FILE = "conv.17-0.7321-0.9302.hdf5"

# Load data
_, _, X_test = load_data()
model.load_weights(WEIGHTS_FILE)

patches_per_axis = TEST_IMG_SIZE // PATCH_SIZE
patches_per_img = patches_per_axis ** 2
num_steps = X_test.shape[0] * patches_per_img // BATCH_SIZE + 1

params = models.get_pipeline_params(model)
Y_test = model.predict_generator(
    input_pipeline_test(X_test, None, **params), 
    num_steps,
    verbose=1)

Y_test = Y_test[:X_test.shape[0] * patches_per_img]

if model.output_shape[-1] == 2:
	Y_test = Y_test[..., 1]

Y_test = Y_test.reshape((X_test.shape[0], patches_per_axis, patches_per_axis, -1))
Y_test = Y_test.mean(axis=3) > THRESHOLD

# Save predictions
f = open(WEIGHTS_FILE + ".pred.csv", 'w')
print('id,prediction', file=f)
for i in range(1, NUM_TEST_FILES+1):
    for x in range(0, patches_per_axis):
        for y in range(0, patches_per_axis):
            print('%.3d_%d_%d,%d' % (i, x * PATCH_SIZE, y * PATCH_SIZE, Y_test[i - 1, y, x]), file=f)
f.close()
