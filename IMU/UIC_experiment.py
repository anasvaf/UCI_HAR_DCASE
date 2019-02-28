import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#local imports
import Classifiers
import UCI_HAR_Dataset as UCI_HAR
from os.path import expanduser
#get actual home path for current user
home = expanduser("~")


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
	#do CNN layers exploration - Layers: 1 - 2 - 3 - 4
	#1 CNN layer
	clf_1CNN_k2 = Classifiers.Hybrid_1CNN_k2(patience=25,name="1CNN_k2")
	clf_1CNN_k2.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf_1CNN_k2.loadBestWeights()
	predictions = clf_1CNN_k2.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf_1CNN_k2.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="1CNN_k2_classification_report.txt")
	clf_1CNN_k2.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="1CNN_k2_CM.png")
	#2 layers
	clf_2CNN_k2 = Classifiers.Hybrid_2CNN_k2(patience=25,name="2CNN_k2")
	clf_2CNN_k2.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf_2CNN_k2.loadBestWeights()
	predictions = clf_2CNN_k2.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf_2CNN_k2.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="2CNN_k2_classification_report.txt")
	clf_2CNN_k2.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="2CNN_k2_CM.png")
	#3 layers
	clf_3CNN_k2 = Classifiers.Hybrid_3CNN_k2(patience=25,name="3CNN_k2")
	clf_3CNN_k2.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf_3CNN_k2.loadBestWeights()
	predictions = clf_3CNN_k2.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf_3CNN_k2.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="3CNN_k2_classification_report.txt")
	clf_3CNN_k2.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="3CNN_k2_CM.png")
	#4 layers
	clf_4CNN_k2 = Classifiers.Hybrid_4CNN_k2(patience=25,name="4CNN_k2")
	clf_4CNN_k2.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf_4CNN_k2.loadBestWeights()
	predictions = clf_4CNN_k2.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf_4CNN_k2.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="4CNN_k2_classification_report.txt")
	clf_4CNN_k2.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="4CNN_k2_CM.png")
	##kernel size exploration - Kernels: 2 - 8 - 16 - 32 - 64
	#kernel 8
	clf_3CNN_k8 = Classifiers.Hybrid_3CNN_k8(patience=25,name="3CNN_k8")
	clf_3CNN_k8.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf_3CNN_k8.loadBestWeights()
	predictions = clf_3CNN_k8.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf_3CNN_k8.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="3CNN_k8_classification_report.txt")
	clf_3CNN_k8.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="3CNN_k8_CM.png")
	#kernel 16
	clf_3CNN_k16 = Classifiers.Hybrid_3CNN_k16(patience=25,name="3CNN_k16")
	clf_3CNN_k16.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf_3CNN_k16.loadBestWeights()
	predictions = clf_3CNN_k16.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf_3CNN_k16.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="3CNN_k16_classification_report.txt")
	clf_3CNN_k16.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="3CNN_k16_CM.png")
	#kernel 32
	clf_3CNN_k32 = Classifiers.Hybrid_3CNN_k32(patience=25,name="3CNN_k32")
	clf_3CNN_k32.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf_3CNN_k32.loadBestWeights()
	predictions = clf_3CNN_k32.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf_3CNN_k32.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="3CNN_k32_classification_report.txt")
	clf_3CNN_k32.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="3CNN_k32_CM.png")
	#kernel 64
	clf_3CNN_k64 = Classifiers.Hybrid_3CNN_k64(patience=25,name="3CNN_k64")
	clf_3CNN_k64.fit(X_tr,y_tr,X_vld,y_vld,batch_size=512)
	clf_3CNN_k64.loadBestWeights()
	predictions = clf_3CNN_k64.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	clf_3CNN_k64.printClassificationReport(true=labels_test,pred=predictions_inv,classes=classes,filename="3CNN_k64_classification_report.txt")
	clf_3CNN_k64.plotConfusionMatrix(true=labels_test,pred=predictions_inv,classes=classes,showGraph=False,saveFig=True,filename="3CNN_k64_CM.png")


	
def export_CNN_features(datapath,clf,clf_name):
	X_train, labels_train, list_ch_train = UCI_HAR.read_data(data_path=datapath, split="train") # train
	X_test, labels_test, list_ch_test = UCI_HAR.read_data(data_path=datapath, split="test") # test
	assert list_ch_train == list_ch_test, "Mistmatch in channels!"
	X_train, X_test = UCI_HAR.standardize(X_train, X_test)
	print("Data size:", len(X_train), " - ", len(X_train[0]))
	#clf = Classifiers.Hybrid_CNN_MLP(patience=25,name="CNN_3_Layers")
	clf.loadBestWeights()
	auto_features = clf.get_layer_output(X_train,"automatic_features")
	print("Features shape: ",auto_features.shape)
	auto_feats_df = pd.DataFrame(auto_features,columns=auto_feats_names)
	print(auto_feats_df.head())
	auto_feats_df.to_csv('auto_train_features_'+clf_name+'.csv.gz',compression='gzip',index=False,header=None)
	
def plot_features_PCA(datapath):
	cnn = "CNN3"
	train_X_df = pd.read_csv("auto_train_features_"+cnn+".csv.gz",names=auto_feats_names,header=None,sep=",",engine='python',compression='gzip')
	train_y_df = pd.read_csv(datapath+"train/y_train.txt",names=['label'],header=None)
	pca = PCA(n_components=3, svd_solver='arpack')
	X = train_X_df.values
	reduced_X = pca.fit_transform(X)
	reduced_df = pd.DataFrame(reduced_X,columns=['x','y','z'])
	print(reduced_df.head())
	print(train_y_df.head())
	all_df = pd.merge(reduced_df, train_y_df, on=reduced_df.index, how='inner')
	print(all_df.head())
	print(all_df.tail())
	wal_df = all_df.loc[all_df['label'] == 1]
	wup_df = all_df.loc[all_df['label'] == 2]
	wdo_df = all_df.loc[all_df['label'] == 3]
	sit_df = all_df.loc[all_df['label'] == 4]
	sta_df = all_df.loc[all_df['label'] == 5]
	lay_df = all_df.loc[all_df['label'] == 6]
	wal_df = wal_df[['x','y','z']]
	wup_df = wup_df[['x','y','z']]
	wdo_df = wdo_df[['x','y','z']]
	sit_df = sit_df[['x','y','z']]
	sta_df = sta_df[['x','y','z']]
	lay_df = lay_df[['x','y','z']]
	#fig, (ax0,ax1,ax2) = plt.subplots(nrows=3, figsize=(14, 7))
	fig, (ax0,ax1) = plt.subplots(nrows=2, figsize=(8,4))
	ax0.scatter(sit_df['x'].values,sit_df['y'].values,c="tab:blue",label="Sit",s=4)
	ax0.scatter(sta_df['x'].values,sta_df['y'].values,c="tab:orange",label="Stand",s=4)
	ax0.scatter(wup_df['x'].values,wup_df['y'].values,c="tab:purple",label="W. Upstairs",s=4)
	ax0.scatter(wdo_df['x'].values,wdo_df['y'].values,c="tab:cyan",label="W. Downtairs",s=4)
	#ax0.scatter(lay_df['x'].values,lay_df['y'].values,c="tab:red",label="Lay")
	ax0.scatter(wal_df['x'].values,wal_df['y'].values,c="tab:green",label="Walk",s=4)
	#1st-3rd PCA components
	ax1.scatter(sit_df['x'].values,sit_df['z'].values,c="tab:blue",label="Sit",s=4)
	ax1.scatter(sta_df['x'].values,sta_df['z'].values,c="tab:orange",label="Stand",s=4)
	ax1.scatter(wup_df['x'].values,wup_df['z'].values,c="tab:purple",label="W. Upstairs",s=4)
	ax1.scatter(wdo_df['x'].values,wdo_df['z'].values,c="tab:cyan",label="W. Downtairs",s=4)
	#ax1.scatter(lay_df['x'].values,lay_df['z'].values,c="tab:red",label="Lay")
	ax1.scatter(wal_df['x'].values,wal_df['z'].values,c="tab:green",label="Walk",s=4)	
	plt.title('PCA CNN Auto Features')
	#plt.legend(loc=1)
	ax0.set_title('PCA components 1 and 2')
	ax1.set_title('PCA components 1and 3')
	ax0.legend()
	ax1.legend()
	fig.tight_layout()
	fig.savefig("PCA_CNN3_sit_sta_wal.png",dpi=300)
	
#Simple CLI interface
def mainMenu():
	#change this to point UCI_HAR data path
	ucihar_datapath = home+"/python/data/UCI_HAR_Dataset/"
	print("1. Train CNN feature extractor\n2. Extract CNN Auto Features\n3. Plot features PCA\n\n Press any other key to exit")
	sel = input("")
	if sel == "1":
		train_CNN_feature_extractor(ucihar_datapath)
		return False
	if sel == "2":
		clf_1CNN_k2 = Classifiers.Hybrid_1CNN_k2(name="1CNN_k2")
		export_CNN_features(ucihar_datapath,clf_1CNN_k2,"1CNN_k2")
		clf_2CNN_k2 = Classifiers.Hybrid_2CNN_k2(name="2CNN_k2")
		export_CNN_features(ucihar_datapath,clf_2CNN_k2,"2CNN_k2")
		clf_3CNN_k2 = Classifiers.Hybrid_3CNN_k2(name="3CNN_k2")
		export_CNN_features(ucihar_datapath,clf_3CNN_k2,"3CNN_k2")
		clf_4CNN_k2 = Classifiers.Hybrid_4CNN_k2(name="4CNN_k2")
		export_CNN_features(ucihar_datapath,clf_4CNN_k2,"4CNN_k2")
		#gen featurs kernel size
		clf_3CNN_k8 = Classifiers.Hybrid_3CNN_k8(name="3CNN_k8")
		export_CNN_features(ucihar_datapath,clf_3CNN_k8,"3CNN_k8")
		clf_3CNN_k16 = Classifiers.Hybrid_3CNN_k16(name="3CNN_k16")
		export_CNN_features(ucihar_datapath,clf_3CNN_k16,"3CNN_k16")
		clf_3CNN_k32 = Classifiers.Hybrid_3CNN_k32(name="3CNN_k32")
		export_CNN_features(ucihar_datapath,clf_3CNN_k32,"3CNN_k32")
		clf_3CNN_k64 = Classifiers.Hybrid_3CNN_k64(name="3CNN_k64")
		export_CNN_features(ucihar_datapath,clf_3CNN_k64,"3CNN_k64")
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
