import numpy as np
import Classifiers
import UCI_HAR_Dataset as UCI_HAR

classes = ["WALKING", "WALK_UPSTAIRS", "WALK_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

#change this to point UCI_HAR data path
ucihar_datapath = "/home/fedecrux/python/data/UCI_HAR_Dataset/"

def train_CNN_feature_extractor():
	X_train, labels_train, list_ch_train = UCI_HAR.read_data(ucihar_data_path=datapath, split="train") # train
	X_test, labels_test, list_ch_test = UCI_HAR.read_data(ucihar_data_path=datapath, split="test") # test
	assert list_ch_train == list_ch_test, "Mistmatch in channels!"
	X_train, X_test = standardize(X_train, X_test)
	print("Data size:", len(X_train), " - ", len(X_train[0]))

	X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, random_state = 123)
	lab_tr[:] = [ y -1 for y in lab_tr ]
	lab_vld[:] = [ y -1 for y in lab_vld ]

	labels_test[:] = [ y -1 for y in labels_test ]

	print(np.unique( np.array(lab_tr)  ))
	print(np.unique( np.array(labels_test)  ))

	y_tr = to_categorical(lab_tr,num_classes=6)#one_hot(lab_tr)
	y_vld = to_categorical(lab_vld,num_classes=6)#one_hot(lab_vld)
	y_test = to_categorical(labels_test,num_classes=6)#one_hot(labels_test)

	clf = Classifiers.Hybrid_CNN_LSTM(patience=1,name="CNN_LSTM_original_")
	clf.loadBestWeights()
	predictions = clf.predict(X_test,batch_size=1)
	predictions_inv = [ [np.argmax(x)] for x in predictions]
	
	clf.printClassificationReport(true=y_test,pred=predictions_inv,classes=classes,filename="CNN_LSTM_original_report.txt")
	clf.plotConfusionMatrix(true=y_test,pred=predictions_inv,classes=classes,showGraph=True,saveFig=True,filename="CNN_LSTM_original_CM.png")
	  
	  

def mainMenu():
	print("1. Train CNN feature extractor\n2. Selection 2\n\n Press any other key to exit")
	sel = input("")
	if sel == "1":
		train_CNN_feature_extractor()
		return False
	else:
		return True





while True:
	if mainMenu():
		break
