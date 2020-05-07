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
                    
    

df = pd.read_csv('effets_guitare4.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles4','conv.p')

with open(p_path,'rb') as handle:
    config = pickle.load(handle)
    
# Téléchargement du modele si existe    
model = load_model(config.model_path)


# Construction des prédictions
y_true , y_pred , fn_prob = build_predictions('clean4')

# Calcul d' Accuracy score
acc_score = accuracy_score(y_true= y_true, y_pred = y_pred)
print("\n")
print("Accuracy score = ", acc_score*100 ,"%")
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
df.to_csv('predictions_4pistes.csv',index = False) 
    

# Tracage de la matrice de confusion
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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


cnf_matrix = confusion_matrix(y_true, y_predict, labels=[0,1,2,3])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix for multi-class')




