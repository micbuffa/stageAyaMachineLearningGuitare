import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from cfg4 import config


from pydub import AudioSegment
from pydub.effects import normalize

from sklearn.preprocessing import MinMaxScaler

wavfiles_namedir = 'Samples-AYA' #le dossier des wavfiles avant nettoyage
norm_namedir = 'norm' #le dossier des wavfiles avant nettoyage


config = config()#instance la classe de configuration 

def Init(wavfiles_namedir):
       
    """Initialise les variables du programme
    Args:
        wavefiles_namedir: le nom de dossier où il y les audiofiles
           
    Returns:
        
        renvoie 5 variables qui seront utilisées dans les autres fonctions       
        df : dataframe contient les données dans le fichier excel 
        classes : contient les noms des classes qui seront utilisés dans l'apprentissage 
        class_dist : contient les libellés des classes et la longueur moyenne de chacune d'elles
        n_samples :le nombre des échantillons de 1/10s dans les wavfiles qui sont en fait possibles dans les signaux en se basant sur 'length'
        prob_dist :La probabilité associée à chaque entrée (classe)
        exemple : 
            Chorus          0.249651
            Nickel-Power    0.250399
        nb_classe : le nombre des classes utilisées dans l'entrainement
        
    """   
    
    dir_ = os.listdir(os.path.join(wavfiles_namedir))#Recupérer les fichiers audios
    #Ces variables sont utilisées pour créer dataframe
    fname =[]#le nom du fichier wav
    length_wav =[]#la longueur 
    label =[]#le libellé du fichier wav
    data ={}#les données de dataframe

    # Récuperer des informations sur chaque fichier wav
    for f in dir_:#f : filename
        rate, signal = wavfile.read(wavfiles_namedir+'/'+f) #recupére les wavfiles nettoyés
        
        (file, ext) = os.path.splitext(f)#récupérer le libellé du fichier wav
        fname.append(f)
        label.append(file)
        length_wav.append(signal.shape[0]/rate)
        
        
    #la construction de dataframe    
    data['label']=label
    data['fname']=fname
    data['length']=length_wav
       
    df = pd.DataFrame(data)

    classes = list(df.label)#recupere les noms des classes existants sans répétition
    class_dist = df.groupby(['label'])['length'].mean()#calcule da la longueur moyenne de les pistes regroupées par nom de classe
    
    nb_classe=len(classes)#le nombre des classes de l'entrainement   
        

    # Création des N sample , la probabilité de distribution et les choices en se basant sur prob_dist
    n_samples =200000#int(df['length'].sum()/0.1) #le nombre des échantillons de 1/10s dans les wavfiles qui sont en fait possibles dans les signaux
    prob_dist = class_dist / class_dist.sum()#La probabilité associée à chaque entrée (classe)
    return df, classes , class_dist , n_samples , prob_dist,nb_classe# Init initilise les varibeles qui seront utilisées dans les autres fonctions
        

def check_samples(config):
    """fonction de vérification :Verifier si il existe déjà des échantillons préparées pour éviter la répétition du travail 
    Args:
        config : une instance de la classes configuration 
          
    Returns:
        renvoie les deux matrices X et Y préparées pour le modèle ,qui sont enregistrées dans le dossier 'samples4'
        sinon rien 
    """    
    if os.path.isfile(config.samples_path) :#verifier si le dossier samples4(contient X et Y du modele) est vide ou non
        print('Loading existing samples {} for model'.format(config.mode))
        with open(config.samples_path,'rb') as handle:
            samp = pickle.load(handle)
            return samp
    else:
        return None
      


threshold =0.02
def build_rand_feat(wavfiles_namedir,config):
    
    """fonction pour la préparation des échantillons  
    Args:
        wavfiles_namedir : le nom de dossier où il y les audiofiles
        config : une instance de la classes configuration 
          
    Returns:
        renvoie les deux matrices X et Y
        et le nombre des classes de l'entrainement
    """ 
    
    #Dans la classe de configuration on un champ "data" qui contient X et Y du 
    #modele si ces dérniers est déja passés par la la fonction build_rand_feat
    # tmp = check_config(config)
    # if tmp:
    #    return tmp.data[0], tmp.data[1]
    
    #Nous utilisons check_samples () car il ne retourne X et Y que s'ils sont déjà préparés (optimisation de la mémoire)
    # samp = check_samples(config)
    # if samp :
    #     return samp[0], samp[1],samp[2] #samp[0] : X , samp[1]: Y, samp[2] : nombre de classes
    
    # Initialiser les varibales qui seront utilisés dans cette fonction seulement
    df, classes , class_dist , n_samples , prob_dist,nb_classe = Init(wavfiles_namedir)
    
    X=[]
    y=[]
    
    
    sample_ok =0
    sample_nok=0    
    
    
    
    _min,_max = float('inf'), -float('inf') #pour comprendre la mise à l'échelle pour normaliser les valeurs de loss et acc
    for _ in tqdm(range(n_samples)):#boucle sur les n échantillons qu'on a calculé déjà 
    
        label_class= np.random.choice(class_dist.index,p=prob_dist)#le choix de la classe est aléatoire chaque itération

        file = np.random.choice(df[df.label==label_class].fname)#recupération de le filename correspondant à la label_class que nous avons généré
      
        # sound = AudioSegment.from_file(wavfiles_namedir+'/'+file, "wav")
        #wav = normalize(sound)
        # wav.export(out_f=norm_namedir+'/'+file, format="wav")

        rate , wav= wavfile.read(norm_namedir+'/'+file)#récupération de wavfile qui correspond au file

        
        rand_index=np.random.randint(0,wav.shape[0]-config.step )#Prendre une valeur du signal basée sur la longueur du audio file, échantillonner directement à cet index et prendre 1/10 s
        sample = wav[rand_index:rand_index+config.step]#l'échantillon est tiré du fichier wav, le début de l'échantillon est rand_index, la longeur de l'échantillon est step = 1/10s 

        # pre_emphasis = 0.97

        # emphasized_signal = np.append(sample[0], sample[1:] - pre_emphasis * sample[:-1])
        sample = sample /pd.Series(sample).apply(np.abs).max()

        #afin que le modèle puisse discerner très rapidement une classification différente 
        #rien d'autre que ces 1 / 10s est supprimé
        
        #Le calcul du mask :

        #s=sample.flatten()#pour rendre l'échantillon unidimensionnel
        

        if pd.Series(sample).apply(np.abs).mean() > threshold: #si True donc l'échantillon est supérieur au seuil
        
            X_sample = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft,
                       winlen=0.032,winstep=0.02)#préparation de l'échantillon en utilisant la formule mfcc
            # MWI norm
            #smin = np.amin(X_sample)
            #smax = np.amax(X_sample)
            #X_sample = (X_sample - np.mean(X_sample)) / (np.std(X_sample)+1e-8)
            
            
            _min=min(np.amin(X_sample), _min)#la valeur minimal de loss obtenue a chaque entrainement
            _max=max(np.amax(X_sample), _max)#la valeur maximal d'accuracy obtenue a chaque entrainement
            X_sample = (X_sample - _min) / (_max - _min)
           
            X.append(X_sample) #la matrice X contient les échantillons préparées
            y.append(classes.index(label_class))#la matrice Y contient les indices des labels récupérés au début de la loop
            sample_ok = sample_ok + 1
        else : 
            sample_nok = sample_nok + 1
    
    
    config.min = _min #sauvegarde de la valeur minimal de loss comme attribut de la classe config
    config.max = _max #sauvegarde de la valeur maximal d'accuracy comme attribut de la classe config
    
    X ,y = np.array(X), np.array(y)#tourner x et y en arrays pour garder une trace de min et lmax
    #X = (X - smin) / (smax - smin)#Pour normaliser X 
    
    
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2], 1) #remodeler X sans modifier ses données pour l'adapter au modèle convolutionnel
    y = to_categorical(y,num_classes=nb_classe)#to_categorical : Convertit un vecteur de classe (entiers)(0-3) en matrice de classe binaire.
    
    config.data = (X , y, nb_classe)#sauenregistrer les x et y en tant qu'attribut data dans la classe de configuration et nb_classes pour une utilisation ultérieure
    with open(config.p_path , 'wb') as handle:
        pickle.dump(config, handle, protocol=2)#sauvegarde de la configuration utilisées dans la préparation des échantillons dans le dossier pickles4 pour une utilisation ultérieure
        
    with open(config.samples_path , 'wb') as handle:#sauvegarde de X et Y dans le dossier samples4 pour une utilisation ultérieure
        pickle.dump(config.data, handle, protocol=2)    
    print(str(sample_ok)+" samples generated, "+str(sample_nok)+" samples rejected")        
      
    return X,y ,nb_classe



X , y , nb_classe= build_rand_feat(wavfiles_namedir,config) #récupérer les Matrices X et Y préparés par la fonction build_rand_feat
