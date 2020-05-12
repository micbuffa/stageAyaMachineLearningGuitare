import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from cfg4 import config

#Initialiser les vaiables utilisées dans les fonctions
csv_namefile = 'effets_guitare4.csv'   
clean_namedir = 'clean4'
prediction_namecsv ='predictions_4pistes.csv'

# La fonction qui intitialise les variables 
def Init (csv_namefile):
    df = pd.read_csv(csv_namefile)
    classes = list(np.unique(df.label))
    fn2class = dict(zip(df.fname, df.label))
    p_path = os.path.join('pickles4','conv.p')

    with open(p_path,'rb') as handle:
        config = pickle.load(handle)
    
    # Téléchargement du modele si existe    
    model = load_model(config.model_path)

    return df , classes , fn2class,model,config

# Initialiser les variables à l'aide de la fonction Init 
df , classes , fn2class , model,config = Init(csv_namefile)

# La fonction qui nous retourne la prédiciton 
def build_predictions(audio_dir):

    y_true = [] 
    y_pred = []
    fn_prob = {}
    
    print('Extracting features from audio')
    
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav =  wavfile.read(os.path.join(audio_dir,fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
    
        for i in range(0,wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft )
            x =( x - config.min) / (config.max - config.min) 
            x = x.reshape(1,x.shape[0],x.shape[1], 1)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
   
            y_true.append(c)
            
            
        fn_prob[fn] = np.mean(y_prob , axis = 0).flatten()
    return y_true, y_pred , fn_prob
  
# La fonction qui rend l'interpretation de la prédiction calculée            
def Prediction(clean_namedir,prediction_namecsv):
    # Construction des prédictions
    y_true , y_pred , fn_prob = build_predictions(clean_namedir)

    # Calcul d' Accuracy score
    acc_score = accuracy_score(y_true= y_true, y_pred = y_pred)

    print("Accuracy score =", acc_score*100 ,"%")
    print("\n")

    # Enregistrement de résulat dans DF 
    y_probs =[]
    for i, row in df.iterrows():
        y_prob = fn_prob[row.fname]
        y_probs.append(y_prob)
        for c , p in zip(classes,y_prob):
            df.at[i, c] =p
   
    y_predict =y_pred
    y_pred = [classes[np.argmax(y)] for y in y_probs]
    df['Output_prediction'] = y_pred

    # Enregistrer les prédictions en tant que fichier Excel
    df.to_csv(prediction_namecsv,index = False) 
    
    return y_true , y_predict



# Tracage de la matrice de confusion
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Calcule de la matrice de confusion
def confusion_matrix_fct(y_true, y_predict):
    cnf_matrix = confusion_matrix(y_true, y_predict, labels=[0,1,2,3])
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix for multi-class')


#Récuperer y_predict et y_true et calculer la précision  
y_true , y_predict = Prediction(clean_namedir,prediction_namecsv)
confusion_matrix_fct(y_true,y_predict)
    





