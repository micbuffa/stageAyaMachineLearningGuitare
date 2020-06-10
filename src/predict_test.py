import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt,mpld3
from matplotlib import cm
from cfg4 import config


#Initialiser les vaiables utilisées dans les fonctions
csv_namefile2 = 'effets_guitare.csv'#le fichier excel des classes utilisées pour former le modèle
csv_namefile = 'LaGrange-Guitars.csv' #le fichier excel de wavfile de test 
# clean_namedir =  'Test/clean_test' #le chemin de dossier de wavfile nettoyé de la classe de test
wavfile_namedir =  'Test/wavfiles_test' #le chemin de dossier de wavfile de la classe de test


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
    df2 = pd.read_csv(csv_namefile2)# Téléchargement du fichier Excel qui contient les noms des pistes d'apprentissage avec label qui les correspond    

    classes = list(df2.label)#récupérer les labels des classes apartir de df sans répitition de ces labels
    
    p_path = os.path.join('pickles4','conv.p')#récupérer le chemin de dossier pickles
    with open(p_path,'rb') as handle:
        config = pickle.load(handle)
    
    #charger le modele formé à partir le dossier models4
    model = load_model(config.model_path)

    return df , classes , model,config

def Cleaning(y, rate, threshold):
    
    """Le calcule l'enveloppe du signal
    Args:
        y: le signal à nettoyer
        rate : Le débit du signal considéré définit le nombre de millions de transitions par seconde.
        threshold : le seuil minimal qu'un signal peut atteindre
           
    Returns:
        
        renvoie l'enveloppe du signal considéré pour qu'il soit appliqué au signal initial afin d'éliminer les amplitudes mortes( mask = enveloppe )
    """   
    mask=[]#liste des true et false depend du seuil 
       
    y=pd.Series(y).apply(np.abs)#Transforme le signal en serie entre 0 et 1 
    y_mean= y.rolling(window=int(rate/100),min_periods=1, center=True).mean()#(Provide rolling window calculations on every 1/100s of signal) 
  
    for mean in y_mean: #si la valeur du signal > le seuil , donc elle est acceptée sinon supprimée
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask 


def build_predictions(wavfile_namedir,config,model,classes):
    """fonction qui génère des prédictions de sortie pour les échantillons d'entrée.
    Args:
        wavfile_namedir : le nom du dossier de la piste de test sans nettoyage
        config : les configuration récupérées dans la fonction INIT
        model : le modele formé récupéré
        classes : contient les noms des classes qui ont été utilisées dans l'apprentissage
                  
    Returns:
        renvoie 
            index_prob : dictionnaire : clé : libellé d'une classe d'apprentissage , valeurs :liste des couples (temps,probabilité(classe))
            exemple : Chorus = [(0,005),(100,0.015),(200,0.213)..]
            ...
            index_class : dictionnaire : clé :chaque 1/10s a libellé de la classe qui a la plus forte probabilité
            exemple : 100ms : Chorus            
    """ 
    index_prob = {}#dictionnaire : chaque classe a une liste des couples (temps , probabilité)
    index_class ={}#dictionnaire : chaque 1/10s a l'indice de classe qui a la plus forte probabilité
    cl={}#dictionnaire contient que les échantillons superieur au seuil ,clé : temps et valeur : mfcc de l'échantillon
    print('Extracting features from audio')
    
    for fn in tqdm(os.listdir(wavfile_namedir)):#loop sur les filennames des morceaux de wavfile de test nettoyé
        rate, wav =  wavfile.read(os.path.join(wavfile_namedir,fn))#récupérer le wavfile
        t=0# t : temps en ms
    
        for i in range(0,wav.shape[0]-config.step, config.step):#loop sur la longueur de wavfile avec un pas de 1/10s
            sample = wav[i:i+config.step]#à chaque iteration , on récupére 1/10s de wavfile
            
             #Le calcul du mask :
            s=sample.flatten()#pour rendre l'échantillon unidimensionnel
            mask = Cleaning(s,rate,0.02) #Le calcule du l'enveloppe
            
            if mask.count(True) == len(mask):#si True donc l'échantillon est supérieur au seuil
                
                cl[t]= mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft,
                       winlen=0.032,winstep=0.015)#préparation de l'échantillon en utilisant la formule mfccs
            t +=100    
            
            #x =( x - config.min) / (config.max - config.min) #normaliser le X avec les valeurs min et max qui sont déjà calculées a la phase de l'apprentissage 
  
    for key , val in cl.items():#boucle sur le disctionnaire des échaantillons nettoyées (le suivi du trvail normal de cette fonction)
               
       smin = np.amin(val)
       smax = np.amax(val)
       x = (val - np.mean(val)) / (np.std(val)+1e-8)
            
       x = x.reshape(1,x.shape[0],x.shape[1], 1)#remodeler X sans modifier ses données pour l'adapter au modèle convolutionnel
       y_hat = model.predict(x)#la probabilité que X soit chaque classe, sa forme : [0.8954995 , 0,0256264, 0,1684461,0.2556898] ,chaque valeur correspond a une classe
            
            
                        
          #remplissage de index_prob   
          #emplacement = une classe
       for i in range(len(classes)):#boucle sur les classes d'apprentissage existantes 
             if classes[i] in index_prob:
                    # ajouter le nouveau couple (temps,prob) au tableau existant à cet emplacement
                 index_prob[classes[i]].append((key,y_hat[0][i]))
             else:                   
                    # créer un nouveau tableau dans cet emplacement
                 index_prob[classes[i]] = [(key,y_hat[0][i])]
                    
            #remplissage de index_class
            #np.argmax(y_hat): rend l'indice de la classe dont la probabilité est la plus forte
            #classes[np.argmax(y_hat)] : récupéré le libellé qui correspond à l'indice de la classe
       index_class[key]=classes[np.argmax(y_hat)]
             
       


    return  index_prob,index_class

def Prediction(wavfile_namedir,config,df,classes,model):
    
    """fonction rend l'interpretation de la prédiction calculée sous forme des graphes
    Args:
        wavfile_namedir : le nom du dossier où nous enregistrons la piste de test 
        config : les configuration récupérées dans la fonction INIT
        df: Trame de données de test précédemment initialisée à l'aide de la fonction Init  
        classes : contient les noms des classes utilisés dans l'apprentissage du modele
        model : le modele formé récupéré
                          
    Returns:
       Affiche les graphes de prédictions pour la classe de test 
                    
    """ 
    # Construction des prédictions : retourne les dictionnaires pour le tracage
    index_prob ,index_class= build_predictions(wavfile_namedir,config,model,classes)
    # index_prob : dictionnaire : clé : libellé d'une classe d'apprentissage , valeurs :liste des couples (temps,probabilité(classe))
    # index_class : clé : temps en ms , valeur : libellé de la classe dont la probabilité est la plus forte

    #Tracage des graphes de resultat
    plot_prediction_probabilities(index_prob)
    plot_prediction_classes(index_class)
    plot_EMA(index_prob,1000)
 

    
def plot_prediction_probabilities(index_prob):

    """fonction qui trace les résultats de la prédiction en fonction des probabilités
    Args:
        index_prob : dictionnaire : clé : libellé d'une classe d'apprentissage , valeurs :liste des couples (temps,probabilité(classe))
    Returns:
         
        trace le graphe de variation des prédictions pour chaque classe d'apprentissage (N courbes meme figure)
            
    """ 
    Sav_Y={}#dictionnaiere utilisé dans le sauvegarde, clé : libellé de la classe , valeur : liste des probabilités correspondante à la classe 
    plt.figure(figsize=(50,5))    
    #Tracage de la courbe de chaque classe
    # les valeurs de l'axe des abscisses et des ordonnées
    #ind : Le libellé de la classe
    #val : la liste des couples (temps, probabilité) correspondante à la classe
    for ind , val in index_prob.items():
        y= []#liste des probabilités d'une classe tout au long la piste de test
        x= []#liste de temps ( 1/10s )
        
        for i in range(len(index_prob[ind])):
            x.append(val[i][0])#Val[i][0]: rend le temps
            y.append(val[i][1])#Val[i][0]: rend la probabilité

        colors = np.random.rand(len(index_prob),3)#colors génère des couleurs arbitraireq pour le tracage 
        plt.plot(x,y,label= ind,color=colors[0],linewidth = 2,markersize=10)
        Sav_Y[ind]=y    

    plt.legend(prop={"size":10},loc='upper left')
    plt.xlim(0,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('probabilities') 
    plt.title('la variation des prédictions du RN (test)')       
    plt.show()

    #Sauvegarde des resultast format : json et csv
    data={}
    data['Temps(1/10s=100ms)']= x 
    for ind , val in Sav_Y.items():
      data[ind] =val
      
    pred_df = pd.DataFrame(data)   
    pred_df.to_csv('Probabilité_classe_predictions.csv',index = False, sep =';')
    pred_df.to_json('Probabilité_classe_predictions.json', orient='records') 


def plot_prediction_classes(index_class):
    """fonction qui trace les résultats de la prédiction en fonction des classes
    Args:
        index_class : dictionnaire : clé : chaque 1/10s de piste de test, valeur: libellé de la classe qui a la plus forte probabilité 

    Returns:
        trace le graphe de variation des prédictions pour la piste de test (en fonction des classes)
    """ 
    x=[]#liste de temps (1/10s)
    classe=[]#liste de libellés des classes
    
    # les valeurs de l'axe des abscisses et des ordonnées
    # ind : temps 
    # val : le libellé de la classe
    for ind , val in index_class.items():        
        x.append(ind)
        classe.append(val)
        
    p=plt.figure(2,figsize=(50,5))
    plt.plot(x, classe, color='black',linewidth = 2,marker='.',markerfacecolor='red', markersize=20) 
    #len(x) : nombre de 1 / 10s dans la piste
    #len(x)*100 : pour récupérer la derniere valeur dans la liste de temps
    plt.xlim(0 ,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('classes') 
    plt.title('la variation des prédictions du RN (test)') 
    # plt.show()
    mpld3.save_html(p,'predict_classe.html')
 
    #Sauvegarde des résulats en tant que fichier excel (la prédiction pour chaque 1/10s de 
    #la piste de test)  
    #100ms = 1/10s
    data = {'Temps(1/10s=100ms)' : x ,'Output_pred ' : classe}
    pred_df = pd.DataFrame(data)   
    pred_df.to_csv('Classes_predictions.csv',index = False, sep =';')
    pred_df.to_json('Classes_predictions.json', orient='records') 

  
    
def plot_EMA(index_prob,window):
    """fonction qui trace la moyenne mobile exponentielle pour chaque classe d'apprentissage
    Args:
        index_prob : dictionnaire : clé : chaque 1/10s de piste de test, valeur: sa probabilité pour chaque classe
        window : est le nombre d'échantillons à considérer
    Returns:
        trace le graphe d'EMA pour chaque classe d'apprentissage (N courbes meme figure)
    """ 
   
    Sav_Y={}#dictionnaiere utilisé dans le sauvegarde, clé : libellé de la classe , valeur : liste des EMA(probabilités) correspondante à la classe 
    plt.figure(3,figsize=(50,5))    
    # ind: contient le libellé de la classe
    # val : contient liste de (temp, probabilité)
    for ind , val in index_prob.items():
        x=[]#liste de temps (1/10s)
        y=[]#liste des probabilités pour chaque classe d'apprentissage
        
        for i in range(len(index_prob[ind])):
            x.append(val[i][0])#Val[i][0]: rend le temps
            y.append(val[i][1])#Val[i][1]: rend la probabilité
            
        colors = np.random.rand(len(index_prob),3) # colors génère des couleurs arbitraireq pour le tracage
        EMA =Exponential_moving_average(y,window) #Le calcul de EMA sur la liste des probbilité pour chaque classe 
        plt.plot(x,EMA,label= ind,color=colors[0],linewidth = 2,markersize=10)
        Sav_Y[ind]=EMA
        
    plt.legend(prop={"size":10},loc='upper left')
    #len(x) : nombre de 1 / 10s dans la piste
    #len(x)*100 : pour récupérer la derniere valeur dans la liste de temps
    plt.xlim(0,len(x)*100)
    plt.xlabel('Time (ms)') 
    plt.ylabel('probabilities') 
    plt.title('la variation des prédictions du RN (test)')       
    plt.show()
    
    #Sauvegarde des resultast format : json et csv
    data={}
    data['Temps(1/10s=100ms)']= x 
    for ind , val in Sav_Y.items():
      data[ind] =val['classe_probs']
            
    pred_df = pd.DataFrame(data) 
    pred_df.to_csv('EMA_predictions.csv',index = False,sep =';')
    y=pred_df.to_json('EMA_predictions.json', orient='records')
    


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
Prediction(wavfile_namedir,config,df,classes,model)


    
