import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt

from Configuration_Class import config

config = config()

# class config:
#     def __init__(self,mode='conv',nfilt=50,nfeat=25,nfft=512,rate=16000,winlen=0.032,winstep=0.02):
#         self.mode=mode
#         self.nfilt=nfilt
#         self.nfeat=nfeat
#         self.nfft = nfft
#         self.rate=rate
#         self.winlen =winlen
#         self.winstep =winstep

#         self.step=10112#5072 #int(rate/10)
#         self.model_path = os.path.join('models', mode + '.model')
#         # self.p_path = os.path.join('pickles4', mode + '.p')
#         # self.samples_path = os.path.join('samples4', 'samples' + '.smp')
#         # self.weight_path = os.path.join('rn-1', 'rn-1'+ '.poid')
#         # self.kfold_path = os.path.join('kfold', 'kfold'+ '.kf')

#"Test-6-JV&GR-Presets+Ozone"       
Track_name ="Test-6-JV&GR-Presets"

wavfiles_namedir ='Dataset' #le chemin de dossier des wavfiles des classes du RN
model_filename = 'models/conv_model'
#test_wavfile = 'Test/Test-All-Classes.wav'
test_wavfile = 'Test/'+Track_name+'.wav'
#test_wavfile = 'Dataset/TubeAmp-Metal.wav'
#test_wavfile = path_+'Test/Test-6-JV&GR-Presets-stereo-2.wav'



def Data_Loading(wavfiles_namedir):
    """
    
    Collects data that will be used in the Testing process
    Arguments:
    ---------
    wavefiles_namedir: the path of the folder where the audiofiles are stored          
    Returns:
    ---------
    classes : contains the names of the classes that will be used in the learning process 
    model : the model formed and saved in the folder 'models' 

    """ 
       
    #Recover the labels of the waves used in learning 
    dir_ = os.listdir(os.path.join(wavfiles_namedir))
    classes =[]
    for f in dir_:
        (file, ext) = os.path.splitext(f)
        classes.append(file)
        
    
    model = tf.keras.models.load_model(model_filename)   
    return classes , model


threshold = 0.02 #  0< threshold <1

def Build_Predictions(test_wavfile,classes,model):
    """
    
    Builds and plots output predictions for the test wavfile.   
    Arguments:
    ---------
    wavfiles_namedir : the path of the folder where the audiofiles are stored 
    classes : contains the names of the classes that will be used in the learning process. 
    model : the recovered formed model                 
    Returns:
    ---------
    Curves
        
    """    
    #dictionary: each class has a list of pairs (time, probability)
    index_prob = {}
    #dictionary: every 100ms has the class index with the highest probability
    index_class ={}
    #dictionary contains only samples above the threshold ,key: time and value: mfcc of sample
    cl={}
    
    #Retrieve the wav test file
    rate, wav =  wavfile.read(test_wavfile)
    print(rate)
    #Time(ms)
    t=0
    #Number of rejected and accepted samples 
    samples_nb = 0
    rejected_samples = 0
    
    _min,_max = float('inf'), -float('inf') 

    #Features extraction from the recovered wavefile
    for i in tqdm(range(0,wav.shape[0]-config.step, config.step)):
        
        sample = wav[i:i+config.step]
        # Normalize sample between [-1,1] before MFCC featuring 
        sample = sample /pd.Series(sample).apply(np.abs).max()
        
        #threshold verification
        if pd.Series(sample).apply(np.abs).mean() > threshold: 
        
          #Using MFCC to extract features
          X_sample =mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt, 
                         nfft = config.nfft,winlen=config.winlen,winstep=config.winstep)
          
          #the minimum and maximum value of loss and accuracy 
          _min=min(np.amin(X_sample), _min)
          _max=max(np.amax(X_sample), _max)
          #Normalize samples using MaxMin norm 
          X_sample = (X_sample - _min) / (_max - _min)

          #Save results for every t
          cl[t] = X_sample.reshape(1,X_sample.shape[0],X_sample.shape[1], 1)
          
          samples_nb = samples_nb + 1  
        else:
         rejected_samples = rejected_samples + 1

  
        t +=100    
    print("\n"+str(samples_nb)+" samples generated, "+str(rejected_samples)+" samples rejected")        
        
    #the prediction generation process
    for key , val in tqdm(cl.items()):
          
        y_hat = model.predict(val)                   
        for i in range(len(classes)):
            if classes[i] in index_prob:
                 index_prob[classes[i]].append((key,y_hat[0][i]))
            else:                   
                 index_prob[classes[i]] = [(key,y_hat[0][i])]
              
        index_class[key]=classes[np.argmax(y_hat)]
        
    #Plotting results
    Plot_Probabilities(index_prob)
    Plot_classes(index_class)
    Plot_EMA(index_prob,5)

def Plot_Probabilities(index_prob):
    """
    
    Plots probabilities.   
    Arguments:
    ---------
    index_prob : contains the plotting data (each class has a list of pairs (time, probability))                 
    Returns:
    ---------
    Probabilities curve
        
    """     
    #dictionary used in the datastore, key: label of the class, 
    #value: list of probabilities corresponding to the class. 
    Sav_Y={}
    
    plt.figure(figsize=(50,5))    

    #Plotting the curve of each class
    for ind , val in index_prob.items():
        #list of probabilities of a class for each 100 ms of the test file
        y= []
        #list of time (step = 100ms )
        x= []
    
        for i in range(len(index_prob[ind])):
            x.append(val[i][0])
            y.append(val[i][1])
        
        Sav_Y[ind]=y    

        colors = np.random.rand(len(index_prob),3)
        plt.plot(x,y,label= ind,color=colors[0],linewidth = 2,markersize=10)

    plt.legend(prop={"size":10},loc='upper left')
    plt.xlim(0,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('probabilities') 
    plt.title('la variation des prédictions du RN (test)')       
    plt.show()

    #Excel file generation process 
    data={}
    data['Temps(ms)']= x 
    for ind , val in Sav_Y.items():
        data[ind] =val

    pred_df = pd.DataFrame(data)   
    pred_df.to_csv('Test/Probabilite_predictions_'+Track_name+'.csv',index = False)    

def Plot_classes(index_class):
    """
    
    Plots classes.   
    Arguments:
    ---------
    index_class : contains the plotting data (every 100ms has the class index with the highest probability)                 
    Returns:
    ---------
    Classes curve
        
    """         
    #list of time(step = 100ms)    
    x=[]
    #list of class labels
    classe=[]

    #Plotting the curve of classes
    for ind , val in index_class.items():        
        x.append(ind)
        classe.append(val)
    
    plt.figure(2,figsize=(50,5))
    plt.plot(x, classe, color='black',linewidth = 2,marker='.',markerfacecolor='red', markersize=20) 

    plt.xlim(0 ,len(x)*100)
    plt.xlabel('Time(ms)') 
    plt.ylabel('classes') 
    plt.title('la variation des prédictions du RN (test)')
    plt.show()

    #Excel file generation process 
    data = {'Temps(ms)' : x ,'Output_pred' : classe}
    pred_df = pd.DataFrame(data)   
    pred_df.to_csv('Test/Classes_predictions_'+Track_name+'.csv',index = False)
 
def Exponential_moving_average(classe_probs,window):
    """
    calculates the exponential moving average of a list of probabilities of a class. 
    An exponential moving average consists in giving more value to the most recent data, 
    while smoothing the lines.
    Arguments:
    ---------    
    class_probs : list of probabilities for a learning class
    window : the number of samples to be considere
    Returns:
    ---------
    EMA calculation
    """ 
    #the ewm function of pandas cannot be applied directly to a list, 
    #only a data frame can use it, so I created a data frame 
    #which contains the classes_probs only to compute the ewm and I used
    #as window size = 5, moving average per 1 s
    data = {'classe_probs' : classe_probs}
    ema_df = pd.DataFrame(data)   
    ema=ema_df.ewm(span = window).mean()
    return ema
     
def Plot_EMA(index_prob,ema_window_size):
    """
    
    Plots exponentielle moving average.   
    Arguments:
    ---------
    index_prob : contains the plotting data (each class has a list of pairs (time, probability))                 
    Returns:
    ---------
    Ema curves for every class
        
    """  
    #dictionary used in the datastore, key: label of the class, 
    #value: list of EMA(probabilities) corresponding to the class.  
    Sav_Y={}
    plt.figure(3,figsize=(50,5))       

    #Plotting the curve of each class
    for ind , val in index_prob.items():
        x=[]
        y=[]
    
        for i in range(len(index_prob[ind])):
            x.append(val[i][0])
            y.append(val[i][1])
     
        #The calculation of EMA on the list of probabilities for each class             
        EMA =Exponential_moving_average(y,ema_window_size) 
        Sav_Y[ind]=EMA
        colors = np.random.rand(len(index_prob),3)      
        plt.plot(x,EMA,label= ind,color=colors[0],linewidth = 2,markersize=10)
        
    plt.legend(prop={"size":10},loc='upper left')
    plt.xlim(0,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('probabilities') 
    plt.title('la variation des prédictions du RN (test)')       
    plt.show()

    #Excel file generation process 
    data={}
    data['Temps(ms)']= x 
    for ind , val in Sav_Y.items():
        data[ind] =val['classe_probs']
            
    pred_df = pd.DataFrame(data) 
    pred_df.to_csv('Test/EMA_predictions_'+Track_name+'.csv',index = False)


print("\n[INFO] Data Loading ..")
classes ,  model = Data_Loading(wavfiles_namedir)

print("\n[INFO] Building predictions .. ")

Build_Predictions(test_wavfile,classes,model)

#Uncomment this to hear the test wavfile 
#import winsound
#winsound.PlaySound(test_wavfile, winsound.SND_FILENAME)