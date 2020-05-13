import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from cfg4 import config


#Initialiser les vaiables utilisées dans les fonctions
csv_namefile2 = 'effets_guitare4.csv'
csv_namefile = 'LaGrange-Guitars.csv'  
clean_namedir =  'Test/clean_test' 
prediction_namecsv ='predictions_test.csv'

# La fonction qui intitialise les variables 
def Init (csv_namefile):
    df = pd.read_csv(csv_namefile)
    df2 = pd.read_csv(csv_namefile2)

    classes = list(np.unique(df2.label))
    fn2class = dict(zip(df.fname, df.label))
    p_path = os.path.join('pickles4','conv.p')

    with open(p_path,'rb') as handle:
        config = pickle.load(handle)
    
    # Téléchargement du modele si existe    
    model = load_model(config.model_path)

    return df , classes , fn2class,model,config

# Initialiser les variables à l'aide de la fonction Init 
df , classes , fn2class , model, config = Init(csv_namefile)

# La fonction qui nous retourne la prédiciton 
def build_predictions(audio_dir):

    y_pred = []
    fn_prob = {}
    
    
    print('Extracting features from audio')
    
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav =  wavfile.read(os.path.join(audio_dir,fn))
        # label = fn2class[fn]
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
   
            # y_true.append(c)
            
            
        fn_prob[fn] = np.mean(y_prob , axis = 0).flatten()
   
    return y_pred , fn_prob
  
# La fonction qui rend l'interpretation de la prédiction calculée            
def Prediction(clean_namedir,prediction_namecsv):
    # Construction des prédictions
    y_pred , fn_prob = build_predictions(clean_namedir)


    # Enregistrement de résulat dans DF 
    y_probs =[]
    for i, row in df.iterrows():
        y_prob = fn_prob[row.fname]
        y_probs.append(y_prob)
        for c , p in zip(classes,y_prob):
            df.at[i, c] =p
   
    y_pred = [classes[np.argmax(y)] for y in y_probs]
    df['Output_prediction'] = y_pred

    # Enregistrer les prédictions en tant que fichier Excel
    df.to_csv(prediction_namecsv,index = False) 
    return y_pred
    
 
def plot_prediction(y_pred):
    
    # les valeurs de l'axe des abscisses
    x =['00:00','00:10','00:20','00:30','00:40','00:50','01:00',
        '01:10','01:20','01:30','01:40','01:50','02:00',
        '02:10','02:20','02:30','02:40','02:50','03:00','03:10',
        '03:20','03:30','03:40']
   
    #Les valeurs del’axe des ordonnées
    y = y_pred 
    plt.xticks(rotation=90, ha='right')
    # plotting the points  
    plt.plot(x, y, color='black', linewidth = 2,marker='.', 
             markerfacecolor='red', markersize=10) 
     
    #Le pas utilisées dans le graphe 
    plt.xlim(0.0,22.5)
    plt.xlabel('Temps (minute)') 
    plt.ylabel('Classes') 
    plt.title('la variation des prédictions du RN (test)') 

    plt.show()

    




    

#Récuperer y_predict et y_true et calculer la précision  
y_pred = Prediction(clean_namedir,prediction_namecsv)

#Tracage de la variation de la prédiction
plot_prediction(y_pred)
    
