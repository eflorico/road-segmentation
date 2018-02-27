from config import PATCH_INPUT_SIZE, LEARNING_RATE
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Lambda, AveragePooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.activations import softmax
import metrics

def convnetv2():
	assert(PATCH_INPUT_SIZE == 96)

	x = inp = Input(shape=(PATCH_INPUT_SIZE, PATCH_INPUT_SIZE, 3))

	def conv(num_filters, x, **kwargs):
	    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same', **kwargs)(x)
	    x = BatchNormalization()(x)
	    return Activation('relu')(x)

	# 96*96*3
	x = conv(32, x)
	x = conv(32, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 48*48*32
	x = conv(64, x)
	x = conv(64, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 24*24*64
	x = conv(128, x)
	x = conv(128, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 12*12*128
	x = conv(256, x)
	x = conv(256, x)
	center = Lambda(lambda x: x[:, 4:8, 4:8, :])(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 6*6*256
	x = conv(512, x)
	x = conv(512, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 3*3*512
	x = conv(512, x)
	x = conv(512, x)
	# 3*3*512
	x = Concatenate()([ Flatten()(x),  Flatten()(center) ])
	# 8704
	x = Dense(4096)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	# 4096
	x = Dense(512)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	# 512
	x = Dense(1)(x)
	out = Activation('sigmoid')(x)

	model = Model(inputs=inp, outputs=out)
	model.compile(loss=keras.losses.binary_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=[metrics.acc, metrics.f1])
	return model


def convnet():
	assert(PATCH_INPUT_SIZE == 96)

	model = Sequential()

	def conv(num_filters, **kwargs):
	    model.add(Conv2D(num_filters, kernel_size=(3, 3), padding='same', **kwargs))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))

	# 96*96*3
	conv(32, input_shape=(PATCH_INPUT_SIZE, PATCH_INPUT_SIZE, 3))
	conv(32)
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# 48*48*64
	conv(64)
	conv(64)
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# 24*24*128
	conv(128)
	conv(128)
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# 12*12*256
	conv(256)
	conv(256)
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# 6*6*512
	model.add(Flatten())
	# 2048
	model.add(Dense(2048))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	# 2048
	model.add(Dense(1024))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	# 512
	model.add(Dense(2, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy', metrics.f1])
	return model


def unet():
	assert(PATCH_INPUT_SIZE == 128)

	x = inp = Input(shape=(PATCH_INPUT_SIZE, PATCH_INPUT_SIZE, 3))

	def conv(num_filters, x, **kwargs):
	    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same', **kwargs)(x)
	    x = BatchNormalization()(x)
	    return Activation('relu')(x)

	def deconv(num_filters, x):
	    return Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)

	# 128*128*3
	x = conv(32, x)
	conv1 = x = conv(32, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 64*64*3
	x = conv(64, x)
	conv2 = conv(64, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 32*32*32
	x = conv(128, x)
	x = conv(128, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 16*16*64
	x = conv(256, x)
	x = conv(256, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 8*8*128
	x = conv(512, x)
	x = conv(512, x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# 4*4*256
	x = conv(1024, x)
	x = conv(1024, x)
	x = deconv(512, x)
	# 8*8*256
	x = Concatenate(axis=3)([x, Lambda(lambda x: x[:, 28:36, 28:36, :])(conv2)])
	x = conv(512, x)
	x = conv(512, x)
	x = deconv(256, x)
	# 16*16*128
	x = Concatenate(axis=3)([x, Lambda(lambda x: x[:, 56:72, 56:72, :])(conv1)])
	x = conv(256, x)
	x = conv(256, x)
	x = Conv2D(2, kernel_size=1, padding='same')(x)
	x = Lambda(lambda x: softmax(x, axis=3))(x)

	out = x

	model = Model(inputs=inp, outputs=out)
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
	              metrics=[metrics.acc_pat, metrics.f1_pat, metrics.acc_pix, metrics.f1_pix])
	return model


def two_stream():
	assert(PATCH_INPUT_SIZE == 128)

	inp = Input(shape=(PATCH_INPUT_SIZE, PATCH_INPUT_SIZE, 3))

	def conv(num_filters, x, **kwargs):
	    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same', **kwargs)(x)
	    x = BatchNormalization()(x)
	    return Activation('relu')(x)

	# 128*128*3
	full = AveragePooling2D(pool_size=(2, 2))(inp)
	# 64*64*3
	full = conv(32, full)
	full = conv(32, full)
	full = MaxPooling2D(pool_size=(2, 2))(full)
	# 32*32*32
	full = conv(64, full)
	full = conv(64, full)
	full = MaxPooling2D(pool_size=(2, 2))(full)
	# 16*16*64
	full = conv(128, full)
	full = conv(128, full)
	full = MaxPooling2D(pool_size=(2, 2))(full)
	# 8*8*128
	full = conv(256, full)
	full = conv(256, full)
	full = MaxPooling2D(pool_size=(2, 2))(full)
	# 4*4*256
	full = conv(512, full)
	full = conv(512, full)
	full = MaxPooling2D(pool_size=(2, 2))(full)
	# 4*4*512
	full = conv(512, full)
	full = conv(512, full)
	# 2*2*512

	center = Lambda(lambda x: x[:, 52:-52, 52:-52, :])(inp)
	# 24*24*3
	center = conv(32, center)
	center = conv(32, center)
	center = MaxPooling2D(pool_size=(2, 2))(center)
	# 12*12*32
	center = conv(64, center)
	center = conv(64, center)
	center = MaxPooling2D(pool_size=(2, 2))(center)
	# 6*6*64
	center = conv(128, center)
	center = conv(128, center)
	center = MaxPooling2D(pool_size=(2, 2))(center)
	# 3*3*128
	center = conv(256, center)
	center = conv(256, center)
	# 3*3*256

	x = Concatenate(axis=1)([ Flatten()(full), Flatten()(center) ])
	# 4352
	x = Dense(5000)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	# 5000
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	# 1024
	x = Dense(1)(x)
	x = out = Activation('sigmoid')(x)

	model = Model(inputs=inp, outputs=out)
	model.compile(loss=keras.losses.binary_crossentropy,
	              optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
	              metrics=['accuracy', metrics.f1])
	return model


def get_pipeline_params(model):
	return {
		'two_classes': model.output_shape[-1] == 2, 
		'reduce_to_patches': len(model.output_shape) <= 2
	}
