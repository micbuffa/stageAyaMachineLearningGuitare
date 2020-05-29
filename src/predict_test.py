import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt,mpld3
from cfg4 import config


#Initialiser les vaiables utilisées dans les fonctions
csv_namefile2 = 'effets_guitare.csv'#le fichier excel des classes utilisées pour former le modèle
csv_namefile = 'LaGrange-Guitars.csv' #le fichier excel de wavfile de test 
clean_namedir =  'Test/clean_test' #le chemin de dossier de wavfile nettoyé de la classe de test

def Init (csv_namefile,csv_namefile2):
    
    """Initialise les variables du programme
    Args:
        csv_namefile: le nom du fichier excel où il y a la liste des noms de fichiers audio de test avec le libellé de la classe qui leur correspond           
        csv_namefile2: le nom du fichier excel où il y a la liste des noms de fichiers audio d'apprentissage avec le libellé de la classe qui leur correspond           

    Returns:
        
        renvoie 4 variables qui seront utilisées dans les autres fonctions       
        df : dataframe contient les données de test dans le fichier excel 
        classes : contient les noms des classes qui ont été utilisées dans l'apprentissage
        model : le modele formé et enregistré dans le dossier 'models4'
        config : les configurations utilisées dans la formation du modele
   
    """   
    df = pd.read_csv(csv_namefile)# Téléchargement du fichier Excel qui contient le nom de la piste de test avec label qui le correspond    
    df2 = pd.read_csv(csv_namefile2)# Téléchargement du fichier Excel qui contient le nom des pistes d'apprentissage avec label qui le correspond    

    classes = list(np.unique(df2.label))#récupérer les labels des classes apartir de df sans répitition de ces labels
    
    p_path = os.path.join('pickles4','conv.p')#récupérer le chemin de dossier pickles
    with open(p_path,'rb') as handle:
        config = pickle.load(handle)
    
    #charger le modele formé à partir le dossier models4
    model = load_model(config.model_path)

    return df , classes , model,config

def build_predictions(clean_namedir,config,model):
    """fonction qui génère des prédictions de sortie pour les échantillons d'entrée.
    Args:
        clean_namedir : le nom du dossier où nous enregistrons les pistes de test nettoyées
        config : les configuration récupérées dans la fonction INIT
        model : le modele formé récupéré
                  
    Returns:
        renvoie 
            index_prob : dictionnaire : clé : chaque 1/10s a sa probabilité pour les 4 classes
            exemple : 100ms =1/10s [0.85,0.25,0.003,0.90]
            ...
            index_class : dictionnaire : clé :chaque 1/10s a l'indice de classe qui a la plus forte probabilité
            exemple : 100ms : 3(l'indice de la probabilité la plus élevé(0.90) -> 3 = Reverb)
            
    """ 
    index_prob = {}#dictionnaire : chaque 1/10s a sa probabilité pour les 4 classes
    index_class ={}#dictionnaire : chaque 1/10s a l'indice de classe qui a la plus forte probabilité
    print('Extracting features from audio')
    
    for fn in tqdm(os.listdir(clean_namedir)):#loop sur les filennames des morceaux de wavfile de test nettoyé
        rate, wav =  wavfile.read(os.path.join(clean_namedir,fn))#récupérer le wavfile
        t=0# t : temps en ms
    
        for i in range(0,wav.shape[0]-config.step, config.step):#loop sur la longueur de wavfile avec un pas de 1/10s
            sample = wav[i:i+config.step]#à chaque iteration , on récupére 1/10s de wavfile
            x = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft,
                       winlen=0.032,winstep=0.015)#préparation de l'échantillon en utilisant la formule mfccs
            #x =( x - config.min) / (config.max - config.min) #normaliser le X avec les valeurs min et max qui sont déjà calculées a la phase de l'apprentissage 
            smin = np.amin(x)
            smax = np.amax(x)
            x = (x - np.mean(x)) / (np.std(x)+1e-8)
            
            x = x.reshape(1,x.shape[0],x.shape[1], 1)#remodeler X sans modifier ses données pour l'adapter au modèle convolutionnel
            y_hat = model.predict(x)#la probabilité que X soit chaque classe, sa forme : [0.8954995 , 0,0256264, 0,1684461,0.2556898] ,chaque valeur correspond a une classe
            
            index_prob[t] = y_hat #t : temps en ms , chaque 100ms = 1/10s, y_hat : la probabilité de chaque classes pour la meme 1/10s
            index_class[t]=np.argmax(y_hat) #t : temps en ms , chaque 100ms = 1/10s,np.argmax(y_hat) : renvoie l'indice de classe qui a la plus forte probabilité pour ce 1 / 10s
            t +=100# 100 ms = 1/10s = 0.1s


    return  index_prob,index_class

def Prediction(clean_namedir,config,df,classes,model):
    
    """fonction rend l'interpretation de la prédiction calculée sous forme des graphes
    Args:
        clean_namedir : le nom du dossier où nous enregistrons les pistes de test nettoyées
        config : les configuration récupérées dans la fonction INIT
        df: Trame de données de test précédemment initialisée à l'aide de la fonction Init  
        classes : contient les noms des classes utilisés dans l'apprentissage du modele
        model : le modele formé récupéré
                          
    Returns:
       Affiche les graphes de prédictions pour la classe de test 
                    
    """ 
        # Construction des prédictions : retourne les dictionnaires pour le tracage
    index_prob ,index_class= build_predictions(clean_namedir,config,model)
    # index_prob : clé: temps en ms , valeur : les 4 probabilités pour chaque classe
    # index_class : clé : temps en ms , valeur : l'indice de classe qui a la plus forte probabilité 

    #Tracage des graphes de resultat
    plot_prediction_probabilities(index_prob)
    plot_prediction_classes(index_class, classes,df)
    plot_EMA(index_prob,1000)
 

    
def plot_prediction_probabilities(index_prob):

    """fonction qui trace les résultats de la prédiction en fonction des probabilités
    Args:
        index_prob : dictionnaire : clé : chaque 1/10s de piste de test, valeur: sa probabilité pour les 4 classes
    Returns:
         
        trace le graphe de variation des prédictions pour chaque classe d'apprentissage (4 courbes meme figure)
            
    """ 

    Chorus =[]#liste des probabilités pour Chorus tout au long la piste
    Reverb =[]#liste des probabilités pour Reverb tout au long la piste
    Phaser_ =[]#liste des probabilités pour Phaser_ tout au long la piste
    Nickel_Power = []#liste des probabilités pour Nickel-Power tout au long la piste
    x= []#liste de temps ( 1/10s )

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
    p=plt.figure(1,figsize=(50,5))
    # Tracage de chaque courbe => courbe : les probabilités d'une classe
    plt.plot(x, Chorus, color='black', label="Chorus",linewidth = 2,
              markersize=10) 
   
    plt.plot(x,Nickel_Power,label="Nickel-Power", color='red', linewidth = 2,
             markersize=10) 
    
    
    plt.plot(x, Phaser_, color='blue',label="Phaser_", linewidth = 2,
            markersize=10)
   
    
    plt.plot(x, Reverb, color='green',label="Reverb", linewidth = 2,
              markersize=10) 
    
    
    data = {'Temps (1/10s=100ms)' : x ,'Chorus ' : Chorus,
            'Nickel-Power ' : Nickel_Power,'Phaser ' : Phaser_,
            'Reverb ' : Reverb}
    pred_df = pd.DataFrame(data)   
    pred_df.to_csv('Probabilité_classe_predictions.csv',index = False, sep =';')
    pred_df.to_json('Probabilité_classe_predictions.json', orient='split') 

    
    #len(x) : nombre de 1 / 10s dans la piste
    #len(x)*100 : pour récupérer la derniere valeur dans la liste de temps
    plt.xlim(0 ,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('probabilities') 
    plt.title('la variation des prédictions du RN (test)') 
    plt.legend(prop={"size":10},loc='upper left')
    mpld3.save_html(p,'plot_prediction_probabilities.html')#mpld3.save_html : permet de sauvegarder le graphe en tant que page html
    #la page est sauvegardée dans src
    # plt.show()
    #mpld3.show() #si vous voulez visualiser le graphe une fois exécuter le programme , le navigateur ouvre automatiquement

def plot_prediction_classes(index_class,classes,df):
    """fonction qui trace les résultats de la prédiction en fonction des classes
    Args:
        index_class : dictionnaire : clé : chaque 1/10s de piste de test, valeur: l'indice de la classe qui a la plus forte probabilité 
        classes : contient les noms des classes qui ont été utilisées dans l'apprentissage
        df : dataframe contient les données de test dans fichier excel 

    Returns:
        trace le graphe de variation des prédictions pour la piste de test (en fonction des classes)
    """ 
    x=[]#liste de temps (1/10s)
    classe=[]#liste des libellés de classes correspondant aux probabilités
    
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
    # pred_df.to_csv(df.label[0] +'_predictions.csv',index = False) 
    pred_df.to_csv('Classes_predictions.csv',index = False, sep =';')
    pred_df.to_json('Classes_predictions.json', orient='split') 


    # plt.xticks(rotation=90, ha='right')
    #la taille de graphe
    p1=plt.figure(2,figsize=(50,5))
    plt.plot(x, classe, color='black',linewidth = 2,marker='.', 
             markerfacecolor='red', markersize=20) 
    #len(x) : nombre de 1 / 10s dans la piste
    #len(x)*100 : pour récupérer la derniere valeur dans la liste de temps
    plt.xlim(0 ,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('classes') 
    plt.title('la variation des prédictions du RN (test)') 
    mpld3.save_html(p1,'plot_prediction_classes.html')#mpld3.save_html : permet de sauvegarder le graphe en tant que page html
    #la page est sauvegardée dans src
    # plt.show()
 
    
def plot_EMA(index_prob,window):
    """fonction qui trace la moyenne mobile exponentielle pour chaque classe d'apprentissage
    Args:
        index_prob : dictionnaire : clé : chaque 1/10s de piste de test, valeur: sa probabilité pour les 4 classes
    Returns:
        trace le graphe d'EMA pour chaque classe d'apprentissage (4 courbes meme figure)
    """ 
    Chorus =[]#liste des probabilités pour Chorus tout au long la piste
    Reverb =[]#liste des probabilités pour Reverb tout au long la piste
    Phaser_ =[]#liste des probabilités pour Phaser_ tout au long la piste
    Nickel_Power = []#liste des probabilités pour Nickel-Power tout au long la piste
    x= []#liste de temps ( 1/10s )

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
    p2=plt.figure(3,figsize=(100,5))
    # Tracage de chaque courbe => courbe : EMA d'une classe
    EMA_C =Exponential_moving_average(Chorus,window)
    plt.plot( x,EMA_C, color='black', label="Chorus_EMA",linewidth = 2,
              markersize=10) 

    EMA_NP =Exponential_moving_average(Nickel_Power,window)
    plt.plot( x,EMA_NP, color='red', label="Nickel-Power_EMA",linewidth = 2,
              markersize=10) 
    
    EMA_P =Exponential_moving_average(Phaser_,window)
    plt.plot( x,EMA_P, color='blue', label="Phaser_EMA",linewidth = 2,
              markersize=10) 
    
    EMA_R =Exponential_moving_average(Reverb,window)
    plt.plot( x,EMA_R, color='green', label="Reverb_EMA",linewidth = 2,
              markersize=10) 
 
    data = {'Temps (1/10s=100ms)' : x ,
            'Chorus_EMA ' : EMA_C['classe_probs'],
            'Nickel-Power_EMA ' : EMA_NP['classe_probs'],
            'Phaser_EMA ' : EMA_P['classe_probs'],
            'Reverb_EMA ' : EMA_R['classe_probs']}
    pred_df = pd.DataFrame(data) 
    pred_df.to_csv('EMA_predictions.csv',index = False,sep =';')
    pred_df.to_json('EMA_predictions.json', orient='split')
    
    
    #len(x) : nombre de 1 / 10s dans la piste
    #len(x)*100 : pour récupérer la derniere valeur dans la liste de temps
    plt.xlim(0 ,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('probabilities') 
    plt.title('la variation des prédictions du RN (test)') 
    plt.legend(prop={"size":10},loc='upper left')
    mpld3.save_html(p2,' plot_EMA.html')#mpld3.save_html : permet de sauvegarder le graphe en tant que page html
    #la page est sauvegardée dans src
    # plt.show() 


def Exponential_moving_average(classe_probs,window):
    """fonction qui calcule la moyenne mobile expo d'une liste de probabilités d'une classe
    une moyenne mobile exponentielle consiste à valoriser davantage les données 
    les plus récentes, tout en lissant les lignes
    Args:
        class_probs : liste des probabilités pour une classe d'apprentissage tout au long la piste de test
        window : est le nombre d'échantillons à considérer
    Returns:
        Calcule de EMA de la liste 
    """ 
    #la fonction ewm de pandas ne peut pas être appliquée directement sur une liste, 
    #seule une trame de données peut l'utiliser, donc j'ai créé une trame de données 
    #qui contient les classes_probs uniquement pour en calculer la ewm et j'ai utilisé
    #comme taille de fenêtre = 1000, moyenne mobile par 1 s
    data = {'classe_probs' : classe_probs}
    ema_df = pd.DataFrame(data)   
    ema=ema_df.ewm(span = window).mean()
    return ema
 
    
# Initialiser les variables à l'aide de la fonction Init 



df , classes , model, config = Init(csv_namefile,csv_namefile2)   
#Récuperer y_pred pour tracer les resulTemps (1/10s=100ms)classe_probstats 
Prediction(clean_namedir,config,df,classes,model)


    
