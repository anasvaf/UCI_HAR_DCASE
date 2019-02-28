import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.utils import to_categorical
import Classifiers
import UCI_HAR_Dataset as UCI_HAR


classes = ["WALKING", "WALK_UPSTAIRS", "WALK_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

#auto features names f1, f2 ..
auto_feats_names = []
for i in range(768):
	auto_feats_names.append("f"+str(i))


def train_CNN_feature_extractor(datapath):
	X_train, labels_train, list_ch_train = UCI_HAR.read_data(data_path=datapath, split="train") # train
	X_test, labels_test, list_ch_test = UCI_HAR.read_data(data_path=datapath, split="test") # test
	assert list_ch_train == list_ch_test, "Mistmatch in channels!"
	X_train, X_test = UCI_HAR.standardize(X_train, X_test)
	print("Data size:", len(X_train), " - ", len(X_train[0]))
	X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, test_size=0.1, stratify = labels_train, random_state = 123)
	lab_tr[:] = [ y -1 for y in lab_tr ]
	lab_vld[:] = [ y -1 for y in lab_vld ]
	labels_test[:] = [ y -1 for y in labels_test ] #labels [1-6] -> [0-5]
	y_tr = to_categorical(lab_tr,num_classes=6)#one_hot(lab_tr)
	y_vld = to_categorical(lab_vld,num_classes=6)#one_hot(lab_vld)
	y_test = to_categorical(labels_test,num_classes=6)#one_hot(labels_test)
	clf = Classifiers.Hybrid_CNN_MLP(patience=25,name="CNN_3_Layers")
	clf.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf.loadBestWeights()
	predictions = clf.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="CNN_classification_report.txt")
	clf.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="CNN_CM.png")
	  
def export_CNN_features(datapath):
	X_train, labels_train, list_ch_train = UCI_HAR.read_data(data_path=datapath, split="train") # train
	X_test, labels_test, list_ch_test = UCI_HAR.read_data(data_path=datapath, split="test") # test
	assert list_ch_train == list_ch_test, "Mistmatch in channels!"
	X_train, X_test = UCI_HAR.standardize(X_train, X_test)
	print("Data size:", len(X_train), " - ", len(X_train[0]))
	clf = Classifiers.Hybrid_CNN_MLP(patience=25,name="CNN_3_Layers")
	clf.loadBestWeights()
	auto_features = clf.get_layer_output(X_train,"automatic_features")
	print("Features shape: ",auto_features.shape)
	auto_feats_df = pd.DataFrame(auto_features,columns=auto_feats_names)
	print(auto_feats_df.head())
	auto_feats_df.to_csv('auto_train_features_CNN3.csv.gz',compression='gzip',index=False,header=None)
	
def plot_features_PCA(datapath):
	cnn = "CNN3"
	train_X_df = pd.read_csv("auto_train_features_"+cnn+".csv.gz",names=auto_feats_names,header=None,sep=",",engine='python',compression='gzip')
	train_y_df = pd.read_csv(datapath+"train/y_train.txt",names=['label'],header=None)
	y = train_y_df['label'].values
	y = [x-1 for x in y]
	pca = PCA(n_components=3, svd_solver='arpack')
	X = train_X_df.values
	reduced_X = pca.fit_transform(X)
	reduced_df = pd.DataFrame(reduced_X,columns=['x','y','z'])
	print(reduced_df.head())
	print(train_y_df.head())
	#fig, (ax0,ax1,ax2) = plt.subplots(nrows=3, figsize=(14, 7))
	

#Simple CLI interface
def mainMenu():
	#change this to point UCI_HAR data path
	ucihar_datapath = "/home/fedecrux/python/data/UCI_HAR_Dataset/"
	print("1. Train CNN feature extractor\n2. Extract CNN Auto Features\n3. Plot features PCA\n\n Press any other key to exit")
	sel = input("")
	if sel == "1":
		train_CNN_feature_extractor(ucihar_datapath)
		return False
	if sel == "2":
		export_CNN_features(ucihar_datapath)
		return False
	if sel == "3":
		plot_features_PCA(ucihar_datapath)
		return False
	else:
		return True

#main CLI application loop
while True:
	if mainMenu():
		break
