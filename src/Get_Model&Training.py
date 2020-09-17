import os
import pickle

import tensorflow as tf

import matplotlib.pyplot as plt

path_samples = 'samples/samples_200k_MAXMIN.smp'
path_models = 'models/conv_model'


def Load_samples():
    """
    
    Loads prepared samples for training 
    Returns:
    ---------
    Returns the two matrices X and Y prepared for the model, which are saved in 
    the'samples' folder,otherwise nothing. 
    
    """    
    if os.path.isfile(path_samples) :
        print('Extracting features from audio')
        with open(path_samples,'rb') as handle:
            samp = pickle.load(handle)
            return samp[0],samp[1],samp[2]
    else:
        return None

def Get_Model(input_shape,nb_classes): 
    """
    
    Returns the CNN model architecture
    Arguments:
    ---------
    input_shape : shape of NN input data
    nb_classes : the number of training classes         
    Returns:
    ---------
    Returns the builded model
    """ 
    model = tf.keras.Sequential()


    model.add(tf.keras.layers.Conv2D(16,(5,5),activation='relu',strides=(1,1),padding='same', input_shape=input_shape))

    model.add(tf.keras.layers.Conv2D(32,(5,5),activation='relu',strides=(1,1),padding='same'))

    model.add(tf.keras.layers.Conv2D(64, (5,5),activation='relu',strides=(1,1),padding='same'))

    model.add(tf.keras.layers.Conv2D(128,(5,5),activation='relu',strides=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D((5,4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(64,activation='relu'))    
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())


    model.add(tf.keras.layers.Dense(nb_classes,activation='softmax'))
 
    model.summary()

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
   
    return model


def Training(X , y,nb_classes):
    """
    
    Trains the CNN model
    Arguments:
    ---------
    X , Y : learning matrices
    nb_classe : the number of training classes      
    Returns:
    ---------
    Plot the accuracy and loss curves of the trained model 

    """ 
    
    #the form of the input data of a CNN
    input_shape = (X.shape[1],X.shape[2], 1 )
    #retrieve the CNN model
    model = Get_Model(input_shape,nb_classes)
    
    #Used to calculate accuracy and save the last best model based on the monitored quantity
    checkpoint =tf.keras.callbacks.ModelCheckpoint(path_models,monitor='val_acc', verbose =2, mode ='max',
                             save_best_only=True)

    #Training the model for 250 epochs and a batch size of 64
    history = model.fit(X, y , epochs=250,batch_size=64,shuffle =True, validation_split=0.1 ,callbacks = [checkpoint])
 
    
    #Saves the trained model to a file to make predictions on new data
    model.save(path_models)

    #Plotting loss and accuracy of validation set and training set 
    print(history.history.keys())
    plt.figure(1)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
  
print("[INFO] Creation of learning samples ...")
X , y , nb_classes=Load_samples()

print("[INFO] Training begins ..")

Training(X,y,nb_classes)
