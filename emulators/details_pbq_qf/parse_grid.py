#!/usr/bin/env python 

import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


file_name = 'photobleaching_mixture00_grid.csv'

features = []
targets  = []

with open(file_name, 'r') as content:

	for line in content:

		linecontent = line.strip().strip('\xef').strip('\xbb').strip('\xbf').strip('\ufeff').split(',')
		print(linecontent)
		feature = np.array([float(element) for element in linecontent[:4]])
		target  = float(linecontent[-1])

		if target > 0.:
			features.append(feature)
			targets.append(target)

features = np.array(features)
targets  = np.array(targets)

d_features = []
for index in range(len(features)):
	for jndex in range(index + 1, len(features)):
		d_features.append(np.linalg.norm(features[index] - features[jndex]))
d_targets = []
for index in range(len(targets)):
	for jndex in range(index + 1, len(targets)):
		d_targets.append(np.abs(targets[index] - targets[jndex]))

plt.scatter(d_features, d_targets)
plt.show()

print(features.shape)

data = {'params': np.array(features), 'values': np.array(targets)}

with open('dataset.pkl', 'wb') as content:
	pickle.dump(data, content)