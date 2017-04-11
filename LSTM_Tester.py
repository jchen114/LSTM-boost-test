from keras.layers import LSTM, Dense, Reshape, Flatten, Masking, Dropout, TimeDistributed
from keras.models import Sequential
import numpy as np

import time


class LSTM_Net():

	def __init__(self, lstm_layers, max_seq_length, num_features, nb_classes):
		self.data_dim = 1
		self.timesteps = 1
		self.nb_classes = nb_classes
		self.batch_size = 1
		print("Build model ----- ")
		self.model = Sequential()

		self.model.add(
			Masking(
				mask_value=-1.0,
				input_shape=(max_seq_length, num_features),
				name='input'
			)
		)

		for layer in lstm_layers:
			self.model.add(
				LSTM(
					output_dim=layer,
					return_sequences=True
				)
			)
			self.model.add(
				Dropout(
					p=0.2
				)
			)

		self.model.add(
			TimeDistributed(
				Dense(
					activation='relu',
					output_dim=64
				)
			)
		)

		# Output Layer
		self.model.add(
			TimeDistributed(
				Dense(
					activation='softmax',
					output_dim=nb_classes
				),
				name='output'
			)
		)

		start = time.time()

		self.model.compile(
			optimizer='rmsprop',
			loss='categorical_crossentropy'
		)

		print("Compilation Time : ", time.time() - start)

		print('model layers: ')
		print(self.model.summary())

		print('model.inputs: ')
		print(self.model.input_shape)

		print('model.outputs: ')
		print(self.model.output_shape)

	def predict_on_seq(self, x, seq_length):
		p = self.model.predict_on_batch(
			x=np.asarray([x])
		)
		preds = p[0][seq_length]
		pred = np.argmax(preds)
		return pred

def generate_seq(max_seq_length):

	np.random.seed(int(time.time()))
	while True:
		sample = -1 * np.ones(shape=(max_seq_length))
		sample_length = np.random.randint(10, max_seq_length)
		for s in range(0, sample_length):
			sample[s] = np.asarray(np.random.randint(0, 20))
		sample = np.asarray(sample, dtype='int32')
		yield sample_length, sample

def hot_encoding(vocab_size, seq):
	new_seq = -1 * np.ones(shape=(seq.shape[0], vocab_size))
	for i, el in enumerate(seq):
		if el == -1:
			break
		vec = np.zeros(shape=(vocab_size))
		vec[el] = 1
		new_seq[i] = vec
	return new_seq


if __name__ == '__main__':
	net = LSTM_Net(
		lstm_layers=[128, 128],
		max_seq_length=30,
		num_features=20,
		nb_classes=20
	)
	seq_gen = generate_seq(30)
	for _ in range(0, 3):
		sample_length, seq = next(seq_gen)
		seq = hot_encoding(20, seq)
		print(net.predict_on_seq(seq, sample_length))
