#!/usr/bin/env python

import os
import pickle
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tf_bijs       = tfp.bijectors
tf_dist       = tfp.distributions
tf_mean_field = tfp.layers.default_mean_field_normal_fn

#==============================================================

class RegressionModel(object):

	NUM_SAMPLES    = 1

	ACT_FUNC       = 'leaky_relu'
	ACT_FUNC_OUT   = 'linear'
	LEARNING_RATE  = 0.75 * 10**-3
	MLP_SIZE       = 48
	REG            = 1e-3
	DROP           = 0.1


	def __init__(self, graph, dataset_details, config, scope, batch_size, max_iter = 10**8):

		self.graph           = graph
		self.scope           = scope
		self.config          = config
		self.batch_size      = batch_size
		self.dataset_details = dataset_details
		self.max_iter        = max_iter
		self.is_graph_constructed = False
		self._read_scaling_details()


	def _generator(self, features, targets, batch_size):
		indices = np.arange(len(features))
		while True:
			np.random.shuffle(indices)
			batch_features = features[indices[:batch_size]]
			batch_targets  = targets[indices[:batch_size]]
			yield (batch_features, batch_targets)


	def _read_scaling_details(self):
		with open(self.dataset_details, 'rb') as content:
			details             = pickle.load(content)
		self.scaling        = {key: details[key] for key in details}
		self.features_shape = self.scaling['features_shape']
		self.targets_shape  = self.scaling['targets_shape']


	def get_scaled_features(self, features):
		if self.config['feature_rescaling'] == 'standardization':
			scaled = (features - self.scaling['mean_features']) / self.scaling['std_features']
		elif self.config['feature_rescaling'] == 'unit_cube':
			scaled = (features - self.scaling['min_features']) / (self.scaling['max_features'] - self.scaling['min_features'])
		return scaled

	def get_scaled_targets(self, targets):
		if self.config['target_rescaling'] == 'standardization':
			scaled = (targets - self.scaling['mean_targets']) / self.scaling['std_targets']
		elif self.config['target_rescaling'] == 'unit_cube':
			scaled = (targets - self.scaling['min_targets']) / (self.scaling['max_targets'] - self.scaling['min_targets'])
		elif self.config['target_rescaling'] == 'mean':
			scaled = targets / self.scaling['mean_targets']
		elif self.config['target_rescaling'] == 'same':
			scaled = targets
		return scaled

	def get_raw_targets(self, targets):
		if self.config['target_rescaling'] == 'standardization':
			raw = targets * self.scaling['std_targets'] + self.scaling['mean_targets']
		elif self.config['target_rescaling'] == 'unit_cube':
			raw = (self.scaling['max_targets'] - self.scaling['min_targets']) * targets + self.scaling['min_targets']
		elif self.config['target_rescaling'] == 'mean':
			raw = targets * self.scaling['mean_targets']
		elif self.config['target_rescaling'] == 'same':
			raw = targets
		return raw


	def set_hyperparameters(self, hyperparam_dict):
		for key, value in hyperparam_dict.items():
			setattr(self, key, value)


	def construct_graph(self):
		act_funcs = {
				'linear':     lambda y: y,
				'leaky_relu': lambda y: tf.nn.leaky_relu(y, 0.2),
				'relu':       lambda y: tf.nn.relu(y),
				'softmax':    lambda y: tf.nn.softmax(y),
				'softplus':   lambda y: tf.nn.softplus(y),
				'softsign':   lambda y: tf.nn.softsign(y),
				'sigmoid':    lambda y: tf.nn.sigmoid(y),
			}

		mlp_activation = act_funcs[self.ACT_FUNC]
		out_activation = act_funcs[self.ACT_FUNC_OUT]

		with self.graph.as_default():
			with tf.name_scope(self.scope):

				self.is_training = tf.compat.v1.placeholder(tf.bool, shape = ())
				self.x_ph        = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.features_shape[1]])
				self.y_ph        = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.targets_shape[1]])

				self.layer_0     = tfp.layers.DenseLocalReparameterization(
									self.MLP_SIZE,
									activation = mlp_activation,
								)
				layer_0_act = self.layer_0(self.x_ph)
				layer_0_out = tf.layers.dropout(layer_0_act, rate = self.DROP, training = self.is_training)

				self.layer_1     = tfp.layers.DenseLocalReparameterization(
									self.MLP_SIZE,
									activation = mlp_activation,
								)
				layer_1_act = self.layer_1(layer_0_out)
				layer_1_out = tf.layers.dropout(layer_1_act, rate = self.DROP, training = self.is_training)

				self.layer_2     = tfp.layers.DenseLocalReparameterization(
									self.MLP_SIZE,
									activation = mlp_activation,
								)
				layer_2_act = self.layer_2(layer_1_out)
				layer_2_out = layer_2_act

				self.layer_3     = tfp.layers.DenseLocalReparameterization(
									self.targets_shape[1],
									activation = out_activation,
								)
				layer_3_out = self.layer_3(layer_2_out)


				self.net_out = layer_3_out

				self.scales  = tf.nn.softplus(tf.Variable(tf.zeros(1)))
				self.y_pred  = tf_dist.Normal(self.net_out, scale = self.scales)


	def construct_inference(self):
		self.is_graph_constructed = True
		with self.graph.as_default():

			self.kl        = sum(self.layer_0.losses) / float(self.batch_size)
			self.kl       += sum(self.layer_1.losses) / float(self.batch_size)
			self.kl       += sum(self.layer_2.losses) / float(self.batch_size)
			self.kl       += sum(self.layer_3.losses) / float(self.batch_size)
			self.reg_loss  = - tf.reduce_mean( self.y_pred.log_prob(self.y_ph) )
			self.loss      = self.reg_loss + self.REG * self.kl

			self.optimizer = tf.compat.v1.train.AdamOptimizer(self.LEARNING_RATE)
			self.train_op  = self.optimizer.minimize(self.loss)
			self.init_op   = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

			self.sess = tf.compat.v1.Session(graph = self.graph)
			with self.sess.as_default():
				self.sess.run(self.init_op)




	def train(self, train_features, train_targets, valid_features, valid_targets, model_path, plot = False, targets = 'same'):

		from sklearn.metrics import r2_score

		if not os.path.isdir(model_path): os.mkdir(model_path)
		logfile = open('%s/logfile.dat' % model_path, 'w')
		logfile.close()

		if not self.is_graph_constructed: self.construct_inference()

		train_feat_scaled = self.get_scaled_features(train_features)
		train_targ_scaled = self.get_scaled_targets(train_targets)
		valid_feat_scaled = self.get_scaled_features(valid_features)
		valid_targ_scaled = self.get_scaled_targets(valid_targets)

		min_target, max_target = np.minimum(np.amin(train_targets, axis = 0), np.amin(valid_targets, axis = 0)), np.maximum(np.amax(train_targets, axis = 0), np.amax(valid_targets, axis = 0))
		if targets == 'probs':
			min_target = 1. / (1. + np.exp( - min_target))
			max_target = 1. / (1. + np.exp( - max_target))

		batch_train_gen = self._generator(train_feat_scaled, train_targ_scaled, self.batch_size)
		batch_valid_gen = self._generator(valid_feat_scaled, valid_targ_scaled, self.batch_size)

		train_errors, valid_errors = [], []

		with self.graph.as_default():
			with self.sess.as_default():

				self.saver = tf.compat.v1.train.Saver()

				if plot:
					import matplotlib.pyplot as plt
					import seaborn as sns
					colors = sns.color_palette('RdYlGn', 4)
					plt.ion()
					plt.style.use('dark_background')
					fig = plt.figure(figsize = (14, 5))
					ax0 = plt.subplot2grid((1, 3), (0, 0))
					ax1 = plt.subplot2grid((1, 3), (0, 1))
					ax2 = plt.subplot2grid((1, 3), (0, 2))

				for epoch in range(self.max_iter):

					train_x, train_y = next(batch_train_gen)
					valid_x, valid_y = next(batch_valid_gen)

					self.sess.run(self.train_op, feed_dict = {self.x_ph: train_x, self.y_ph: train_y, self.is_training: True})

					if epoch % 200 == 0:
						valid_preds = self.sess.run(self.net_out, feed_dict = {self.x_ph: valid_x, self.is_training: False})

						valid_y     = self.get_raw_targets(valid_y)
						valid_preds = self.get_raw_targets(valid_preds)

						if targets == 'probs':
							valid_y     = 1. / (1. + np.exp( - valid_y))
							valid_preds = 1. / (1. + np.exp( - valid_preds))

						try:
							valid_r2 = r2_score(valid_y, valid_preds)
						except:
							valid_r2 = np.nan
						valid_errors.append(valid_r2)

						_1_, _2_ = self.sess.run([self.reg_loss, self.kl], feed_dict = {self.x_ph: train_x, self.y_ph: train_y, self.is_training: False})
						print('...', _1_, _2_)

						train_preds = self.sess.run(self.net_out, feed_dict = {self.x_ph: train_x, self.is_training: False})
						train_y     = self.get_raw_targets(train_y)
						train_preds = self.get_raw_targets(train_preds)

						try:
							train_r2 = r2_score(train_y, train_preds)
						except:
							train_r2 = np.nan
						train_errors.append(train_r2)

						if targets == 'probs':
							train_y     = 1. / (1. + np.exp( - train_y))
							train_preds = 1. / (1. + np.exp( - train_preds))

						logfile = open('%s/logfile.dat' % model_path, 'a')
						logfile.write('%d\t%.5f\t%.5f\n' % (epoch, train_r2, valid_r2))
						logfile.close()


						# define break condition --> last improvement happened more than 100 epochs ago
						max_r2_index = np.argmax(valid_errors)
						if len(valid_errors) - max_r2_index > 100: break

						if max_r2_index == len(valid_errors) - 1:
							self.saver.save(self.sess, '%s/model.ckpt' % model_path)

						new_line = 'EVALUATION: %d (%d)\t%.5f\t%.5f' % ( len(valid_errors) - max_r2_index, len(valid_errors), train_errors[-1], valid_errors[-1])
						print(new_line)

						if plot:

							train_preds_scaled = train_preds
							train_trues_scaled = train_y
							valid_preds_scaled = valid_preds
							valid_trues_scaled = valid_y

							ax0.cla()
							ax1.cla()
							ax2.cla()

							ax0.plot([min_target[0], max_target[0]], [min_target[0], max_target[0]], lw = 3, color = 'w', alpha = 0.5)
							ax0.plot(train_trues_scaled[:, 0], train_preds_scaled[:, 0], marker = '.', ls = '', color = colors[-1], alpha = 0.5)
							ax0.plot(valid_trues_scaled[:, 0], valid_preds_scaled[:, 0], marker = '.', ls = '', color = colors[0], alpha = 0.5)

							if len(min_target) > 1:
								ax1.plot([min_target[1], max_target[1]], [min_target[1], max_target[1]], lw = 3, color = 'w', alpha = 0.5)
								ax1.plot(train_trues_scaled[:, 1], train_preds_scaled[:, 1], marker = '.', ls = '', color = colors[-1], alpha = 0.5)
								ax1.plot(valid_trues_scaled[:, 1], valid_preds_scaled[:, 1], marker = '.', ls = '', color = colors[0], alpha = 0.5)

							RANGE = 50

							ax2.plot(np.arange(len(train_errors[-RANGE:])) + len(train_errors[-RANGE:]), train_errors[-RANGE:], lw = 3, color = colors[-1])
							ax2.plot(np.arange(len(valid_errors[-RANGE:])) + len(valid_errors[-RANGE:]), valid_errors[-RANGE:], lw = 3, color = colors[0])

							plt.pause(0.05)



	def restore(self, model_path):
		if not self.is_graph_constructed: self.construct_inference()
		self.sess  = tf.compat.v1.Session(graph = self.graph)
		self.saver = tf.compat.v1.train.Saver()
		try:
			self.saver.restore(self.sess, model_path)
			return True
		except AttributeError:
			return False



	def predict(self, input_raw):

		input_scaled = self.get_scaled_features(input_raw)

		with self.sess.as_default():
			output_scaled = []
			for _ in range(self.NUM_SAMPLES):
				output_scaled.append(self.sess.run(self.net_out, feed_dict = {self.x_ph: input_scaled, self.is_training: False}))
		output_scaled = np.array(output_scaled)

		output_raw      = self.get_raw_targets(output_scaled)
		output_raw_mean = np.mean(output_raw, axis = 0)
		output_raw_std  = np.std(output_raw, axis = 0)

		return {'samples': output_raw, 'averages': output_raw_mean, 'uncertainties': output_raw_std}


#==============================================================
