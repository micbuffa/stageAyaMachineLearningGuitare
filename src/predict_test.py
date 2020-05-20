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
csv_namefile2 = 'effets_guitare.csv'#le fichier excel des classes utilisées pour former le modèle
csv_namefile = 'LaGrange-Guitars.csv' #le fichier excel de wavfile de test 
clean_namedir =  'Test/clean_test' #le chemin de dossier de wavfile nettoyé de la classe de test

# La fonction qui intitialise les variables 
def Init (csv_namefile,csv_namefile2):
    df = pd.read_csv(csv_namefile)# Téléchargement du fichier Excel qui contient le nom de la piste de test avec label qui le correspond    
    df2 = pd.read_csv(csv_namefile2)# Téléchargement du fichier Excel qui contient le nom des pistes d'apprentissage avec label qui le correspond    

    classes = list(np.unique(df2.label))#récupérer les labels des classes apartir de df sans répitition de ces labels
    
    p_path = os.path.join('pickles4','conv.p')#récupérer le chemin de dossier pickles
    with open(p_path,'rb') as handle:
        config = pickle.load(handle)
    
    #charger le modele formé à partir le dossier models4
    model = load_model(config.model_path)

    return df , classes , model,config




# La fonction qui nous calcule la prédiciton 
def build_predictions(clean_namedir,config,model):

    index_prob = {}#dictionnaire : chaque 1/10s a sa probabilité pour les 4 classes
    index_class ={}#dictionnaire : chaque 1/10s a l'indice de classe qui a la plus forte probabilité

    
    
    print('Extracting features from audio')
    
    for fn in tqdm(os.listdir(clean_namedir)):#loop sur les filennames des morceaux de wavfile de test nettoyé
        rate, wav =  wavfile.read(os.path.join(clean_namedir,fn))#récupérer le wavfile
        t=0# t : temps en ms
    
        for i in range(0,wav.shape[0]-config.step, config.step):#loop sur la longueur de wavfile avec un pas de 1/10s
            sample = wav[i:i+config.step]#à chaque iteration , on récupére 1/10s de wavfile
            x = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft )#préparation de l'échantillon en utilisant la formule mfccs
            x =( x - config.min) / (config.max - config.min) #normaliser le X avec les valeurs min et max qui sont déjà calculées a la phase de l'apprentissage 
            x = x.reshape(1,x.shape[0],x.shape[1], 1)#remodeler X sans modifier ses données pour l'adapter au modèle convolutionnel
            y_hat = model.predict(x)#la probabilité de X  d'etre chaque classe , sa forme : [0.8954995 , 0,0256264, 0,1684461,0.2556898] ,chaque valeur correspond a une classe
            
   

            index_prob[t] = y_hat #t : temps en ms , chaque 100ms = 1/10s, y_hat : la probabilité de chaque classes pour la meme 1/10s
            index_class[t]=np.argmax(y_hat) #t : temps en ms , chaque 100ms = 1/10s,np.argmax(y_hat) : renvoie l'indice de classe qui a la plus forte probabilité pour ce 1 / 10s
            t +=100# 100 ms = 1/10s = 0.1s


    return  index_prob,index_class
  
    
  
# La fonction qui rend l'interpretation de la prédiction calculée sous forme des graphes           
def Prediction(clean_namedir,config,df,classes,model):
    
    # Construction des prédictions : retourne les dictionnaires pour le tracage
    index_prob ,index_class= build_predictions(clean_namedir,config,model)
    # index_prob : clé: temps en ms , valeur : les 4 probabilités pour chaque classe
    # index_class : clé : temps en ms , valeur : l'indice de classe qui a la plus forte probabilité 

    #Tracage de la variation de la prédiction
    #le pas de l'axe des abscisses : 10000ms = 10 s 

    plot_prediction_probabilities(index_prob,10000)
    plot_prediction_classes(index_class, classes,10000,df)
    
    
    
    
    

    
#Le tracage des résultats de la prédiction (probabilités)
def plot_prediction_probabilities(index_prob,pas):#Fonction à modifier
    Chorus =[]#liste des probabilités pour Chorus tout au long la piste
    Reverb =[]#liste des probabilités pour Reverb tout au long la piste
    Phaser_ =[]#liste des probabilités pour Phaser_ tout au long la piste
    Nickel_Power = []#liste des probabilités pour Nickel-Power tout au long la piste
    x= []#liste de temps ( 1/10s )
    temps_pas = pas # le pas de l'axe des abscisse (10000ms = 10 s)

    # les valeurs de l'axe des abscisses et des ordonnées
    # ind : temps 
    # val : list des probabilités ,va[0][0] : prob Chorus , va[0][1] : prob Nickel-Power 
    # , va[0][2]: prob Phaser_ ,va[0][3] : prob Reverb
    for ind , val in index_prob.items():
        Chorus.append(val[0][0])
        Reverb.append(val[0][3])
        Phaser_.append(val[0][2])
        Nickel_Power.append(val[0][1])
        x.append(ind)

    # La taille de graphe 
    plt.figure(figsize=(200,10))
    # Tracage de chaque courbe => courbe : les probabilités d'une classe
    plt.plot(x, Chorus, color='black', label="Chorus",linewidth = 2,
              markersize=10) 
    plt.plot(x, Nickel_Power,label="Nickel-Power", color='red', linewidth = 2,
             markersize=10) 
    
    plt.plot(x, Phaser_, color='blue',label="Phaser_", linewidth = 2,
            markersize=10)
    
    plt.plot(x, Reverb, color='green',label="Reverb", linewidth = 2,
              markersize=10) 
 
    #Le pas utilisées dans le graphe  = 10000ms = 10s
    plt.xticks(np.arange(0, len(x)*100, temps_pas))
    plt.xlim(0 ,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('probabilities') 
    plt.title('la variation des prédictions du RN (test)') 
    plt.legend(prop={"size":10},loc='upper left')
    plt.show()

#Le tracage des résultats de la prédiction (classes)
def plot_prediction_classes(index_class,classes,pas,df):#Fonction à modifier
    x=[]#liste de temps (1/10s)
    classe=[]#liste des libellés de classes correspondant aux probabilités
    temps_pas = pas # le pas de l'axe des abscisse (10000ms = 10 s)
    
    
    # les valeurs de l'axe des abscisses et des ordonnées
    # ind : temps 
    # val : list des indices des classes , 0 : Chorus , 1 : Nickel-Power , 2 : Phaser_ , 3 : Reverb
    for ind , val in index_class.items():
        #récupérer le libellé de la classe
        for c  in classes:
            if classes.index(c) == val:
                classe.append(c)

        x.append(ind)
    
    #Sauvegarde des résulats en tant que fichier excel (la prédiction pour chaque 1/10s de 
    #la piste de test)  
    #100ms = 1/10s
    
    data = {'Temps (1/10s=100ms)' : x ,'Output_pred ' : classe}
    pred_df = pd.DataFrame(data)   
    pred_df.to_csv(df.label[0] +'_predictions.csv',index = False) 
     

    # plt.xticks(rotation=90, ha='right')
    #la taille de graphe
    plt.figure(figsize=(200,10))
    plt.plot(x, classe, color='black',linewidth = 2,marker='.', 
             markerfacecolor='red', markersize=20) 
    #len(x) : nombre de 1 / 10s dans la piste
    #len(x)*100 : pour récupérer la derniere valeur dans la liste de temps
    plt.xticks(np.arange(0, len(x)*100, temps_pas))
    plt.xlim(0 ,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('classes') 
    plt.title('la variation des prédictions du RN (test)') 
    plt.show()
 
# Initialiser les variables à l'aide de la fonction Init 
df , classes , model, config = Init(csv_namefile,csv_namefile2)   

#Récuperer y_pred pour tracer les resultats 
Prediction(clean_namedir,config,df,classes,model)


    
