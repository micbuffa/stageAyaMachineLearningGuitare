import os
import pickle

from scipy.io import wavfile
import pandas as pd
import numpy as np

import tensorflow as tf
from tqdm import tqdm
from python_speech_features import mfcc


from Configuration_Class import config

import warnings
warnings.filterwarnings("ignore")

#An instance of the configuration class
config = config()


wavfiles_namedir = 'Dataset'

def Data_Loading(wavfiles_namedir):
    """
    
    Collects data that will be used in the learning process
    Arguments:
    ---------
    wavefiles_namedir: the path of the folder where the audiofiles are stored          
    Returns:
    ---------
    df : dataframe contains information about the dataset
    classes : contains the names of the classes that will be used in the learning process. 
    class_dist : contains the labels of the classes and the average length of each of them
    n_samples :the number of samples used
    prob_dist :The probability associated for each entry (class)
        ex : Chorus+TubeAmp-Clean      0.083333          
    nb_classes : the number of classes used in learning
        
    """   
    #Recover dataset
    dir_ = os.listdir(os.path.join(wavfiles_namedir))
    fname =[]#the list of wav filenames
    length_wav =[]#the list of wav file lengths
    label =[]#the list of wav file labels
    data ={}#dataframe data

    #Gather information about each wav file
    for f in dir_:#f : filename
        #recovers wavfiles and their sampling rate
        rate, signal = wavfile.read(wavfiles_namedir+'/'+f)
        
        #retrieve information about the loaded wav file
        (file, ext) = os.path.splitext(f)
        fname.append(f)
        label.append(file)
        length_wav.append(signal.shape[0]/rate)
      
    #the construction of dataframe   
    data['label']=label
    data['fname']=fname
    data['length']=length_wav
    df = pd.DataFrame(data)

    #recovers the names of existing classes without repetition
    classes = list(df.label)
    #calculates the average length of each track  
    class_dist = df.groupby(['label'])['length'].mean()
    #the number of training classes 
    nb_classes=len(classes) 
    #the number of samples to be generated
    n_samples =200000
    #The probability associated with each entry (class)
    prob_dist = class_dist / class_dist.sum()

    return df, classes, class_dist, n_samples, prob_dist, nb_classes
        





def Preprocessing(wavfiles_namedir,threshold):
    """
    
    prepares the loaded data for the learning phase
    Arguments:
    ---------
    wavfiles_namedir : the path of the folder where the audiofiles are stored
    threshold : the minimum threshold that a signal can reach        
    Returns:
    ---------
    X : Matrix contains the prepared samples
    y : Matrix contains label indexes of the true predictions
    nb_classes : the number of training classes
    
    """ 
    
    #retrieve the loaded and prepared dataset
    df, classes , class_dist , n_samples , prob_dist, nb_classes = Data_Loading(wavfiles_namedir)

    # The X and Y matrices 
    X=[]
    y=[]
    
    #the number of samples that are above and below the threshold
    sample_ok =0
    sample_nok=0    
    
    #To normalize samples
    _min,_max = float('inf'), -float('inf') 
    
    for _ in tqdm(range(n_samples)):
        
        #the choice of the class is random each iteration
        label_class= np.random.choice(class_dist.index,p=prob_dist)
        #recover the filename corresponding to the previously generated label_class
        file = np.random.choice(df[df.label==label_class].fname)
        #retrieve the wavfile that corresponds to "file" and its sample rate
        rate , wav= wavfile.read(wavfiles_namedir+'/'+file)

        # retrieve samples from the wavfile recovered starting at a random_index
        rand_index=np.random.randint(0,wav.shape[0]-config.step )
        sample = wav[rand_index:rand_index+config.step]

        # Normalize sample between [-1,1] before MFCC featuring 
        sample = sample /pd.Series(sample).apply(np.abs).max()
   
        #threshold verification
        if pd.Series(sample).apply(np.abs).mean() > threshold:
            
            #Using MFCC to extract features
            X_sample = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft,winlen=config.winlen,winstep=config.winstep)
                    
            
            #the minimum and maximum value of loss and accuracy obtained at each epoch
            _min=min(np.amin(X_sample), _min)
            _max=max(np.amax(X_sample), _max)
            #Normalize samples using MaxMin norm 
            X_sample = (X_sample - _min) / (_max - _min)
           
            X.append(X_sample) 
            y.append(classes.index(label_class))
            
            sample_ok = sample_ok + 1
        else : 
            sample_nok = sample_nok + 1
    
    #Save the minimum loss value and the maximum accuracy value as 
    # an attributes of the configuration class.
    config.min = _min 
    config.max = _max 

    X ,y = np.array(X), np.array(y)

    #remodel X without modifying its data to adapt it to the convolutional model
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2], 1) 
    #Converts the class vector (0-nb_classes-1) to binary class matrix
    y = tf.keras.utils.to_categorical(y,num_classes=nb_classes)
    
    #Save the configuration used in sample preparation in the pickles folder for later use.
    with open(config.p_path , 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
        
    #save X and Y in the samples folder for training
    config.data = (X , y, nb_classes)
    with open(config.samples_path , 'wb') as handle:
        pickle.dump(config.data, handle, protocol=2)   
        
    print(str(sample_ok)+" samples generated, "+str(sample_nok)+" samples rejected")        
      
    return X,y ,nb_classes



X , y , nb_classes= Preprocessing(wavfiles_namedir,0.02) #récupérer les Matrices X et Y préparés par la fonction build_rand_feat

