#!/usr/bin/env python 

import numpy as np
import pickle 

#=========================================================

NUM_SPECTRA =  1040
FOLD_SIZE   =  170

np.random.seed(120415)

#=========================================================

indices = np.arange(NUM_SPECTRA)
np.random.shuffle(indices)

test_indices = indices[850:]
work_indices = indices[:850]

#=========================================================

rotation_indices = work_indices.copy()

cross_validation_sets = []

for index in range(len(work_indices) // FOLD_SIZE):

	valid_indices = rotation_indices[:FOLD_SIZE]
	train_indices = rotation_indices[FOLD_SIZE:]

	cross_validation_dict = {'train_indices': train_indices.copy(), 'valid_indices': valid_indices.copy()}
	cross_validation_sets.append(cross_validation_dict)

	rotation_indices = np.roll(rotation_indices, FOLD_SIZE)

print('generated %d folds' % (len(cross_validation_sets)))

data_set = {'test_indices': test_indices, 'work_indices': work_indices, 'cross_validation_sets': cross_validation_sets}
pickle.dump(data_set, open('indices.pkl', 'wb'))
