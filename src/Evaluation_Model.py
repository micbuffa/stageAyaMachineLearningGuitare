import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc

from sklearn.metrics import accuracy_score , confusion_matrix

import matplotlib.pyplot as plt
from Configuration_Class import config

import warnings
warnings.filterwarnings("ignore")


prediction_namecsv ='predictions_pistes.csv'
wavfiles_namedir = 'Dataset' 
path_model ="models/conv_model"

# class config:
#     def __init__(self,mode='conv',nfilt=50,nfeat=25,nfft=512,rate=16000):
#         self.mode=mode
#         self.nfilt=nfilt
#         self.nfeat=nfeat
#         self.nfft = nfft
#         self.rate=rate

#         self.step=10112#5072 #int(rate/10)
#         self.model_path = os.path.join('models', mode + '.model')
#         #self.p_path = os.path.join('pickles', mode + '.p')
#         #self.samples_path = os.path.join('samples', 'samples' + '.smp')
#         #self.weight_path = os.path.join('rn-1', 'rn-1'+ '.poid')
#         #self.kfold_path = os.path.join('kfold', 'kfold'+ '.kf')
config= config()

def Data_Loading(wavfiles_namedir):
    """
    
    Collects data that will be used in the Testing process
    Arguments:
    ---------
    wavefiles_namedir: the path of the folder where the audiofiles are stored          
    Returns:
    ---------
    df : dataframe contains information about the dataset
    classes : contains the names of the classes that will be used in the learning process 
    model : the model formed and saved in the folder 'models' 
       
    """ 
   
    #Recover dataset
    dir_ = os.listdir(os.path.join(wavfiles_namedir))
    fname =[]#the list of wav filenames
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
      
    #the construction of dataframe   
    data['label']=label
    data['fname']=fname
    df = pd.DataFrame(data)
    
    #recovers the names of existing classes without repetition
    classes = list(df.label)
   
    #load the formed model from the models folder
    model = tf.keras.models.load_model(path_model)

    return df , classes ,model


def Build_Predictions(wavfiles_namedir,classes,model):
    """
    
    Generates output predictions for input samples.   
    Arguments:
    ---------
    wavfiles_namedir : the path of the folder where the audiofiles are stored 
    classes : contains the names of the classes that will be used in the learning process. 
    model : the recovered formed model                 
    Returns:
    ---------
    y_true : the true data
    y_pred : the predicted data
    fn_prob : dictionary : key : filenames , value : the average of the probabilities 
              for the same learning class ,exemple : TubeAmp-Clean : [0.82,0.002,0.12,0.2,..]
        
    """ 
    #list of inputs (before interpretation)
    y_true = [] 
    #list of ouputs (after interpretation)
    y_pred = [] 
    fn_prob = {}
    
    
    print('Extracting features from audio')
    _min,_max = float('inf'), -float('inf')
    for fn in tqdm(os.listdir(wavfiles_namedir)):
        #retrieve the wavfile
        rate, wav =  wavfile.read(os.path.join(wavfiles_namedir,fn))
        #retrieve the label of the class that corresponds to the file
        label = df.loc[df['fname'] == fn, 'label'].iloc[0] 
        #recover the collected label index  
        c = classes.index(label)
        #List of probabilities for a wavfile
        y_prob = []
        
        #Features extraction from the recovered wavefile
        for i in range(0,wav.shape[0]-config.step, config.step):
           
            sample = wav[i:i+config.step]
            # Normalize sample between [-1,1] before MFCC featuring 
            sample = sample /pd.Series(sample).apply(np.abs).max()

            #threshold verification
            if pd.Series(sample).apply(np.abs).mean() > 0.02:
                
                #Using MFCC to extract features
                x = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft,winlen=0.032,winstep=0.02)
                
                #the minimum and maximum value of loss and accuracy 
                _min=min(np.amin(x), _min)
                _max=max(np.amax(x), _max)
                #Normalize samples using MaxMin norm 
                x =( x - _min) / (_max - _min) 
                
                #remodel X without modifying its data to adapt it to the convolutional model
                x = x.reshape(1,x.shape[0],x.shape[1], 1)
                #Gives the prediction of X
                y_hat = model.predict(x)
               
                #gather the probabilities of the same wavfile 
                y_prob.append(y_hat)
                #Contains the indices of the predicted classes where the probability was the highest
                y_pred.append(np.argmax(y_hat))
                #Contains the index of the right class to expect for each wav file
                y_true.append(c)
                
        # dictionary : key : filenames , value : the average of the probabilities 
        #for the same learning class ,exemple : TubeAmp-Clean : [0.82,0.002,0.12,0.2,..]    
        fn_prob[fn] = np.mean(y_prob , axis = 0).flatten()
        
    return y_true, y_pred , fn_prob 
  
          
def Prediction_tocsv(df,classes,y_true,y_pred,fn_prob):
    """
    
    transforms the interpretation of the results as an Excel file    
    Arguments:
    ---------
    df : dataframe contains information about the dataset 
    classes : contains the names of the classes that will be used in the learning process
    y_true : the true data
    y_pred : the predicted data
    fn_prob : dictionary : key : filenames , value : the average of the probabilities 
              for the same learning class ,exemple : TubeAmp-Clean : [0.82,0.002,0.12,0.2,..]                
    Returns:
    ---------
    predictions_pistes.csv : generates an Excel file of the result
    Displays the accuracy score in %.     
            
    """ 
        
    #Accuracy Score
    acc_score = accuracy_score(y_true= y_true, y_pred = y_pred)
    print("\n[INFO] Accuracy score: {:.2f}%".format(acc_score * 100))
    
    #Excel file generation process    
    #list of probabilities    
    y_probs =[]
    for i, row in df.iterrows():
        y_prob = fn_prob[row.fname] 
        y_probs.append(y_prob) 
        
        #for the same wav file there are nb_classes probabilities, each corresponding to a class
        for c , p in zip(classes,y_prob):
            df.at[i, c] =p 
   
    #retrieve the class name corresponds to the highest probability for a wavfile
    y_predicted_class = [classes[np.argmax(y)] for y in y_probs] 
    #Add a column in DF "Output_prediction" where there is the final prediction for each wav file
    df['Output_prediction'] = y_predicted_class 

    #Excel file
    df.to_csv(prediction_namecsv,index = False,sep =';') 
 
    
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Greys):
    """
    
    draws the confusion matrix (predefined function)    
    Arguments:
    ---------        
    cm : contains the result of confusion_matrix of y_true and y_pred
    classes : contains the names of the classes that will be used in the learning process
    normalize : if True then the matrix will be plotted with 1 and 0 , otherwise with the real values.
    cmap : the representation color of the matrix
    Returns:
    ---------              
    Confusion matrix plotted
            
    """ 
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def confusion_matrix_fct(y_true, y_pred,classes): 
    """
    
    Calculates the confusion matrix for the plotting function 
    Arguments:
    ---------        
    y_true : the true data
    y_pred : the predicted data
    classes : contains the names of the classes that will be used in the learning process
    Returns:
    ---------        
    Display the confusion matrix
            
    """ 
    #List of labels to index the matrix
    labels=[]
    for i in range(len(classes)):
        labels.append(i)
     
    #Calculates the confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    np.set_printoptions(precision=2)
    #Plotting ..
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix')


print("[INFO] Data Loading ..")
df , classes ,  model = Data_Loading(wavfiles_namedir)

print("\n[INFO ] Building predictions ..")

y_true , y_pred , fn_prob = Build_Predictions(wavfiles_namedir,classes,model)

print("\n[INFO] Excel file generation ..")
Prediction_tocsv(df,classes,y_true,y_pred,fn_prob)

print("\n[INFO] Confusion Matrix ..")

confusion_matrix_fct(y_true,y_pred,classes)
    





