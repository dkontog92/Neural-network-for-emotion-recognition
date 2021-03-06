import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

def test_fer_model(img_folder,model):
	'''Function to import the test images and load the trained neural network
       
       Arguments: img_folder: path to directory of the test images
                  model: path to the pickle object
    
	'''
    #The mean of our training data. It needs to be subtracted from the test
    #data to make predictions
	mean = 135.1788
	with open(model, 'rb') as handle:
		network = pickle.load(handle)      
    
	directory_files = os.listdir(img_folder)
	directory_files = sorted(directory_files,key=str.lower)
	xdata = []
	for file in directory_files:
		if file[-3:] == 'jpg':
			xdata.append(plt.imread(img_folder+'\\'+file)[:, :, 0])
         
	xdata = np.array(xdata)
	xdata = xdata - mean
	network.loss(xdata)
    
	preds = np.argmax(network.loss(xdata),1)
	print(preds)
	return preds


if __name__ == '__main__':

# USER INPUT - START -------------------------------------------------- #
    # Specify the full path to test data
    img_folder = 'C:\\Users\\dimit\\Desktop\\assignment2_advanced\\datasets\\public\\Test'

    # Specify the full path to model
    # Our selected model is in the pickles subdirectory in the src directory 
    modeldir ='pickles'

    # USER INPUT - END ---------------------------------------------------- #

    model = modeldir + '\\net.pickle'
    preds = test_fer_model(img_folder, model)
