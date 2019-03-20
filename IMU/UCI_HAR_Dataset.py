import numpy as np
import pandas as pd
import os


#### Utility Functions to load UCI_HAR

def read_ACC_data(data_path, split = "train"):
	""" Read data """
	# Fixed params
	n_class = 6
	n_steps = 128
	# Paths
	path_ = os.path.join(data_path, split)
	path_signals = os.path.join(path_, "Inertial_Signals")
	# Read labels and one-hot encode
	label_path = os.path.join(path_, "y_" + split + ".txt")
	labels = pd.read_csv(label_path, header = None)
	# Read time-series data
	channel_files = os.listdir(path_signals)
	channel_files.sort()
	n_channels = len(channel_files)
	posix = len(split) + 5
	# Initiate array
	list_of_channels = []
	X = np.zeros((len(labels), n_steps,6))#,n_channels))
	i_ch = 0
	#print(channel_files)
	#input("pause")
	for fil_ch in channel_files:
		if 'gyro' in fil_ch:#fil_ch not in  ['total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt','total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']:#
			continue
		channel_name = fil_ch[:-posix]
		dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
		#****************************modified**********88
		X[:,:,i_ch] = dat_.values#dat_.as_matrix()
		# Record names
		list_of_channels.append(channel_name)
		# iterate
		i_ch += 1
	# Return
	return X, labels[0].values, list_of_channels

def read_IMU_data(data_path, split = "train"):
	""" Read data """
	# Fixed params
	n_class = 6
	n_steps = 128
	# Paths
	path_ = os.path.join(data_path, split)
	path_signals = os.path.join(path_, "Inertial_Signals")
	# Read labels and one-hot encode
	label_path = os.path.join(path_, "y_" + split + ".txt")
	labels = pd.read_csv(label_path, header = None)
	# Read time-series data
	channel_files = os.listdir(path_signals)
	channel_files.sort()
	n_channels = len(channel_files)
	posix = len(split) + 5
	# Initiate array
	list_of_channels = []
	X = np.zeros((len(labels), n_steps,9))#,n_channels))
	i_ch = 0
	#print(channel_files)
	#input("pause")
	for fil_ch in channel_files:
		channel_name = fil_ch[:-posix]
		dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
		#****************************modified**********88
		X[:,:,i_ch] = dat_.values#dat_.as_matrix()
		# Record names
		list_of_channels.append(channel_name)
		# iterate
		i_ch += 1
	# Return
	return X, labels[0].values, list_of_channels

def standardize(train, test):
	""" Standardize data """
	print("TODO use a scaler")
	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]
	return X_train, X_test
