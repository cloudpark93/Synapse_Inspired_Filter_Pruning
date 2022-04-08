import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.ndimage.filters import gaussian_filter1d

file_name = 'resnet18_DYJS_step_learning_rates'

data = csv.DictReader(open("{}.csv".format(file_name)))

for raw in data:
    loss = np.float64(list(raw.keys()))
    lr = np.float64(list(raw.values()))


lr_angle_1stdev = np.gradient(lr)
lr_angle_2nddev = np.gradient(lr_angle_1stdev)
lr_2nddev_clipped = np.clip(np.abs(np.gradient(lr_angle_2nddev)), 0.0001, 2)
smoothed_signal = gaussian_filter1d(lr_2nddev_clipped, 20)

idx = [np.argmin(lr)]
max_idx = []
max_idx.append(idx)
max_idx.append(argrelmax(smoothed_signal)[0].tolist())
max_idx = np.array(max_idx).flatten()

fig, ax = plt.subplots()
ax.set_title('LR finder')
ax.set_ylabel('loss')
ax.set_xlabel('learning rate [index]')
ax.plot(lr)
ax.scatter(max_idx, lr[max_idx], marker='x', color='red')
fig.savefig('{}.jpg'.format(file_name))
np.savetxt('{}_range.csv'.format(file_name), loss[max_idx], delimiter=',')








