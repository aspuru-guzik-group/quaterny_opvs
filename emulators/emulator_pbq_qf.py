#!/usr/bin/env python

#===============================================================================

from abstract_emulator import AbstractEmulator

#===============================================================================

class Emulator_PBQ_QF(AbstractEmulator):

	PATH        = 'details_pbq_qf'
	HYPERPARAMS = {
    	'ACT_FUNC': 'leaky_relu',
        'ACT_FUNC_OUT': 'relu',
        'LEARNING_RATE': 0.001,
        'MLP_SIZE': 120,
        'REG': 10**-2.0,
        'DROP': 0.2}

	def __init__(self, home = './'):
		self.home = home
		self.path = f'{self.home}/{self.PATH}'
		AbstractEmulator.__init__(self)
		self._load_config(path = self.path)

	def load_dataset(self):
		self._load_indices(path = self.path)
		self._load_dataset(path = self.path)

	def run_experiment(self, features, full_stats = False):
		prediction = self.predict(features)
		if full_stats:
			return prediction
		else:
			return prediction[self.config['objective']['name']]

#===============================================================================

def train():
	''' train the emulator on the dataset
	'''
	emulator = Emulator_PBQ_QF()
	emulator.load_dataset()
	emulator.initialize_models(batch_size=len(emulator.train_features[0]))
	emulator.set_hyperparameters(emulator.HYPERPARAMS)

	emulator.construct_models()
	emulator.train(path=emulator.path, plot=True)


def predict_test_set(plot=True):
	''' runs predictions on the test set
	'''
	emulator = Emulator_PBQ_QF()
	emulator.load_dataset()
	emulator.initialize_models(batch_size=len(emulator.train_features[0]))
	emulator.set_hyperparameters(emulator.HYPERPARAMS)

	pred_dict = emulator.predict(emulator.test_features, reshaped = True)

	if plot:
		from sklearn.metrics import r2_score
		test_targets = emulator.test_targets
		pred_targets = pred_dict['averages']
		r2 = r2_score(emulator.test_targets, pred_targets)
		print('R2', r2)

		import matplotlib.pyplot as plt
		import seaborn as sns
		fig = plt.figure(figsize = (4, 4))
		plt.plot([0.0, 0.7], [0.0, 0.7], color = 'k')
		plt.plot(test_targets, pred_targets, ls = '', marker = 'o', color = 'k', markersize = 4)
		plt.plot(test_targets, pred_targets, ls = '', marker = 'o', color = 'b', markersize = 2)
		plt.title('PBQ-QF (%.3f)' % r2)
		plt.tight_layout()
		plt.show()

#===============================================================================

if __name__ == '__main__':
	predict_test_set()
