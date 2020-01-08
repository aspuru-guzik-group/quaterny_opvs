#!/usr/bin/env python

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('Emulators')


import json
import pickle
import numpy as np
import tensorflow as tf

#=====================================================================

class AbstractEmulator(object):

	def __init__(self, num_folds = 5):
		self.num_folds        = num_folds
		self.dataset          = None
		self.dataset_stats    = None
		self.indices          = None
		self.model_batch_size = None
		self.file_path        = None

	def set_file_path(self, path, file_name):
		self.file_name = file_name
		self.path      = path


	def _load_config(self, path = './'):
		file_name   = '%s/config.json' % path
		with open(file_name) as content:
			self.config = json.loads(content.read())
		self.config['num_params'] = len(self.config['parameters'])
		self.hyperparameters = self.config['emulator_parameters']
		self.path = path

	def set_hyperparameters(self, hyperparameters):
		self.hyperparameters = hyperparameters

	def _load_indices(self, path = './'):
		file_name    = '%s/indices.pkl' % path
		try:
			with open(file_name, 'rb') as content:
				self.indices = pickle.load(content)
		except UnicodeDecodeError:
			with open(file_name, 'rb') as content:
				self.indices = pickle.load(content, encoding = 'latin1')
#		self.indices = pickle.load(open(file_name, 'rb'))
		self.work_indices  = self.indices['work_indices']
		self.test_indices  = self.indices['test_indices']
		self.train_indices = [self.indices['cross_validation_sets'][index]['train_indices'] for index in range(self.num_folds)]
		self.valid_indices = [self.indices['cross_validation_sets'][index]['valid_indices'] for index in range(self.num_folds)]


	def _store_dataset_stats(self, path = './'):

		max_features  = np.amax(self.features, axis = 0)
		min_features  = np.amin(self.features, axis = 0)
		max_targets   = np.amax(self.targets, axis = 0)
		min_targets   = np.amin(self.targets, axis = 0)

		mean_features = np.mean(self.features, axis = 0)
		std_features  = np.std(self.features, axis = 0)
		mean_targets  = np.mean(self.targets, axis = 0)
		std_targets   = np.std(self.targets, axis = 0)

		stats_dict  = {'min_features': min_features, 'max_features': max_features,
						 'min_targets': min_targets, 'max_targets': max_targets,
						 'mean_features': mean_features, 'std_features': std_features,
						 'mean_targets': mean_targets, 'std_targets': std_targets,
						 'features_shape': self.features.shape, 'targets_shape': self.targets.shape}

		with open('%s/dataset_stats.pkl' % path, 'wb') as content:
			pickle.dump(stats_dict, content)
		self.dataset_stats = '%s/dataset_stats.pkl' % path


	def _load_dataset(self, path = './'):
		file_name    = '%s/dataset.pkl' % path
		with open(file_name, 'rb') as content:
			self.dataset = pickle.load(content)

		values     = self.dataset['values']
		raw_values = self.dataset['values']

		if self.config['general']['domain'] == 'simplex':
			params = []
			for element in self.dataset['params']:
				vector = element / np.sum(element)
				params.append(vector[:-1])
			params = np.array(params)
		else:
			params = self.dataset['params']
		raw_params = self.dataset['params']

		self.all_features = raw_params
		self.all_targets  = raw_values

		self.features = params[self.work_indices]
		self.targets  = values[self.work_indices]
		if len(self.targets.shape) == 1:
			self.targets  = np.reshape(self.targets, (len(self.targets), 1))

		self._store_dataset_stats(path = path)

		self.test_features = params[self.test_indices]
		self.test_targets  = values[self.test_indices]
		if len(self.targets.shape) == 1:
			self.test_targets  = np.reshape(self.test_targets, (len(self.test_targets), 1))

		self.train_features, self.train_targets = [], []
		self.valid_features, self.valid_targets = [], []
		for index in range(self.num_folds):

			train_features = params[self.train_indices[index]]
			valid_features = params[self.valid_indices[index]]
			train_targets  = values[self.train_indices[index]]
			if len(train_targets.shape) == 1:
				train_targets  = np.reshape(train_targets, (len(train_targets), 1))
			valid_targets  = values[self.valid_indices[index]]
			if len(valid_targets.shape) == 1:
				valid_targets  = np.reshape(valid_targets, (len(valid_targets), 1))

			self.train_features.append(train_features)
			self.train_targets.append(train_targets)
			self.valid_features.append(np.concatenate([valid_features for i in range(len(train_features) // len(valid_features))]))
			self.valid_targets.append(np.concatenate([valid_targets for i in range(len(train_targets) // len(valid_targets))]))


	def initialize_models(self, batch_size = 1):
		if self.config['general']['model'] == 'probabilistic':
			from model_probabilistic import RegressionModel as Model
		elif self.config['general']['model'] == 'deterministic':
			from model_deterministic import RegressionModel as Model
		else:
			raise NotImplementedError

		self.models = []
		self.graphs = [tf.Graph() for i in range(self.num_folds)]
		for fold_index in range(self.num_folds):
			with self.graphs[fold_index].as_default():
				model = Model(self.graphs[fold_index], self.dataset_stats, self.config['general'], scope = 'fold_%d' % fold_index, batch_size = batch_size)
				self.models.append(model)


	def construct_models(self, hyperparameters = None):
		if hyperparameters is None:
			hyperparameters = self.hyperparameters
		for model_index, model in enumerate(self.models):
			with self.graphs[model_index].as_default():
				model.set_hyperparameters(hyperparameters)
				model.construct_graph()


	def load_models(self, path = None, batch_size = 1):
		print('... loading models')
		if path is None:
			path = self.path
		if self.dataset_stats is None:
			self.dataset_stats = '%s/dataset_stats.pkl' % path

		if self.config['general']['model'] == 'probabilistic':
			from model_probabilistic import RegressionModel as Model
		elif self.config['general']['model'] == 'deterministic':
			from model_deterministic import RegressionModel as Model
		else:
			raise NotImplementedError

		self.models = []
		self.graphs = [tf.Graph() for i in range(self.num_folds)]
		for fold_index in range(self.num_folds):
			with self.graphs[fold_index].as_default():
				model = Model(self.graphs[fold_index], self.dataset_stats, self.config['general'], scope = 'fold_%d' % fold_index, batch_size = batch_size)
				model.set_hyperparameters(self.hyperparameters)
				model.construct_graph()
				if model.restore('%s/Fold_%d/model.ckpt' % (path, fold_index)):
					self.models.append(model)
				else:
					print('could not restore model: ', fold_index)
					break
		else:
			self.model_batch_size = batch_size
			return True
		self.model_batch_size = None
		return False


	def train(self, path = './', plot = False):
		for model_index, model in enumerate(self.models):

			model.train(self.train_features[model_index], self.train_targets[model_index],
						self.valid_features[model_index], self.valid_targets[model_index],
						model_path = '%s/Fold_%d' % (path, model_index), plot = plot, targets = self.config['general']['targets'])


	def predict(self, params, reshaped = False):

		if len(params.shape) == 1:
			params = np.reshape(params, (1, len(params)))

		if self.config['general']['domain'] == 'simplex' and not reshaped:
			features = []
			for element in params:
				vector = element / np.sum(element)
				features.append(vector[:-1])
			features = np.array(features)
		else:
			features = params

		polar_features = []
		for feature_index, feature in enumerate(features):
			polar_feature = []
			for elem_index, element in enumerate(feature):

				if 'rep' in self.config['parameters'][elem_index]:
					polar_feature.extend([np.sin(element), np.cos(element)])
				else:
					polar_feature.append(element)
			polar_features.append(polar_feature)
		features = np.array(polar_features)



		if not len(features) == self.model_batch_size:
			self.load_models(batch_size = len(features))

		samples = []
		for fold_index in range(self.num_folds):
			single_pred_dict = self.models[fold_index].predict(features)	# samples, averages, uncertainties
			samples.append(single_pred_dict['samples'])
		samples = np.array(samples)

#		if self.config['general']['targets'] == 'simplex':
#			samples = 1. / (1. + np.exp( - samples) )

		# shape of samples: (# models, # draws, # features, # dim features)
		averages      = np.mean(np.mean(samples, axis = 0), axis = 0)
		uncertainties = np.std(np.mean(samples, axis = 0), axis = 0)

		pred_dict = {'samples': samples, 'averages': averages, 'uncertainties': uncertainties}
		pred_dict[self.config['objective']['name']] = np.squeeze(pred_dict['averages'])
		return pred_dict

#=====================================================================

if __name__ == '__main__':

	pass
