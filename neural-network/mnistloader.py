import struct as st
import sys, os
import numpy as np

def import_data():
    # Obtain the current directory for the python file
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    filename = {'training_images' : 'train-images-idx3-ubyte' ,
                'training_labels' : 'train-labels-idx1-ubyte',
                'test_images'     : 't10k-images-idx3-ubyte',
                'test_labels'     : 't10k-labels-idx1-ubyte'}

    # Extract the test images
    test_images_file = open(os.path.join(__location__, filename['test_images']),'rb')
    test_images_file.seek(4)
    # Find information regarding images
    nImg_test = st.unpack('>I',test_images_file.read(4))[0] # Number of images
    nR_test = st.unpack('>I',test_images_file.read(4))[0] # Number of rows
    nC_test = st.unpack('>I',test_images_file.read(4))[0] # Numbe of columns
    nBytesTotal = nImg_test*nR_test*nC_test*1 # Each pixel data is 1 byte
    # Organise images into 784 pixel long row vectors
    test_images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal, 
                                        test_images_file.read(nBytesTotal))).reshape((nImg_test,nR_test*nC_test)) / 255

    # Extract the training labels
    test_labels_file = open(os.path.join(__location__, filename['test_labels']),'rb')
    # As before, skip the first 4 bytes
    test_labels_file.seek(4)
    nLbls_test = st.unpack('>I',test_labels_file.read(4))[0] # Number of labels
    nBytesTotal = nLbls_test*1 # Each label is 1 byte
    test_labels_array = np.array(st.unpack('>'+'B'*nBytesTotal,test_labels_file.read(nBytesTotal))).reshape(nLbls_test)

    # Extract the training images
    train_images_file = open(os.path.join(__location__, filename['training_images']),'rb')
    # Skip first 4 bytes
    train_images_file.seek(4)
    # Find information regarding number and size of images
    nImg_train = st.unpack('>I',train_images_file.read(4))[0] # Number of images
    nR_train = st.unpack('>I',train_images_file.read(4))[0] # Number of rows
    nC_train = st.unpack('>I',train_images_file.read(4))[0] # Numbe of columns
    nBytesTotal = nImg_train*nR_train*nC_train*1 # Each pixel data is 1 byte
    # Organise images into 784 pixel long row vectors
    train_images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,
                                            train_images_file.read(nBytesTotal))).reshape((nImg_train,nR_train*nC_train)) / 255

    # Extract the training labels
    train_labels_file = open(os.path.join(__location__, filename['training_labels']),'rb')
    # Skip first 4 bytes again
    train_labels_file.seek(4)
    nLbls_train = st.unpack('>I',train_labels_file.read(4))[0] # Number of labels
    nBytesTotal = nLbls_train*1 # Each label is 1 byte
    train_labels_array = np.array(st.unpack('>'+'B'*nBytesTotal,
                                    train_labels_file.read(nBytesTotal))).reshape(nLbls_train)

    validation_set_size = 10000

    return zip_data(train_images_array[:-validation_set_size], train_labels_array[:-validation_set_size], 
                    train_images_array[-validation_set_size:], train_labels_array[-validation_set_size:],
                    test_images_array, test_labels_array)

def zip_data(train_images_array, train_labels_array,
             validation_images_array, validation_labels_array, 
             test_images_array, test_labels_array):
    # Create zip objects containing training images and vectorised inputs based on training labels
    training_inputs = [np.reshape(x, (784, 1)) for x in train_images_array]
    training_results = [label_to_vector(y) for y in train_labels_array]
    training_data = list(zip(training_inputs, training_results))
    # Create validation data containing validation images and regular labels
    validation_inputs = [np.reshape(x, (784, 1)) for x in train_images_array]
    validation_data = list(zip(validation_inputs, validation_labels_array))
    # Create test data containing images and regular labels
    test_inputs = [np.reshape(x, (784, 1)) for x in test_images_array]
    test_data = list(zip(test_inputs, test_labels_array))

    return training_data, validation_data, test_data
    
def label_to_vector(label):
    # Set classification vector according to the labels
    Y = np.zeros((10,1))
    Y[label] = 1
    return Y

if __name__ == "__main__":
    training_data, validation_data, test_data = import_data()
    print(len(test_data))