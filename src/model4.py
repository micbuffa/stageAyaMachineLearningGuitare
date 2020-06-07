import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg4 import config
import matplotlib.pyplot as plt

#Initialiser les vaiables utilisées dans les fonctions
csv_namefile = 'effets_guitare.csv' #le fichier excel 
clean_namedir = 'clean4' #Le dossier des wavfile nettoyés
config = config()#instance la classe de configuration 


def Init(csv_namefile,clean_namedir):
    """Initialise les variables du programme
    Args:
        csv_namefile: le nom du fichier excel où il y a la liste des noms de fichiers audio avec le libellé de la classe qui leur correspond
        clean_namedir: le nom de dossier où il y les audiofiles nettoyés 
           
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
            Phaser_         0.249551
            Reverb          0.250399
        nb_classe : le nombre des classes utilisées dans l'entrainement

        
    """   
    #Téléchargement du fichier Excel qui contient le nom de la piste avec label qui le correspond       
    df = pd.read_csv(csv_namefile)
    df.set_index('fname', inplace=True)#df.set_index : Défini fname dans DataFrame à l'aide des colonnes existantes.
    
    nb_classe=len(df)#le nombre des classes de l'entrainement
    # Récuperer les échantions nettoyées et le calcul de la longeur de chaque piste
    for f in df.index:#indice de 0 à 123 (nombre de wavfiles dans le fichier excel)
        rate, signal = wavfile.read(clean_namedir+'/'+f) #recupére les wavfiles nettoyés
        df.at[f, 'length'] = signal.shape[0]/rate #pour chaque wavfile nettoyé , on calcule la longeur par la formule 

    # Récupérer les labelles des classes : Chorus , Nickel-Power , Reverb - Phaser_ 
    classes = list(np.unique(df.label))#recupere les noms des classes existants sans répétition
    class_dist = df.groupby(['label'])['length'].mean()#calcule da la longueur moyenne de les pistes regroupées par nom de classe
    
    # Création des N sample , la probabilité de distribution et les choices en se basant sur prob_dist
    n_samples = 4 * int(df['length'].sum()/0.1) #le nombre des échantillons de 1/10s dans les wavfiles qui sont en fait possibles dans les signaux
    prob_dist = class_dist / class_dist.sum()#La probabilité associée à chaque entrée (classe)
    return df, classes , class_dist , n_samples , prob_dist,nb_classe# Init initilise les varibeles qui seront utilisées dans les autres fonctions
        

def check_config(config):
    """fonction de vérification : Verifier si il existe déja une configuration pour le modele pour éviter la répétition du travail 
    Args:
        config : une instance de la classes configuration 
          
    Returns:
                renvoie une configuration existante pour le modèle ,qui est enregistrée dans le dossier 'pickles4'
        sinon rien 
        
    """   
    if os.path.isfile(config.p_path) :#verifier si le dossier pickles4(contient la configuration) est vide ou non
        print('Loading existing data {} for model'.format(config.mode))
        with open(config.p_path,'rb') as handle:
            tmp = pickle.load(handle)#tmp contient le fichier conv.p où on sauvegarde la configuration
            
            return tmp
    else:
        return None


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
      

def build_rand_feat(csv_namefile,clean_namedir,config):
    
    """fonction pour la préparation des échantillons  
    Args:
        csv_namefile: le nom du fichier excel où il y a la liste des noms de fichiers audio avec le libellé de la classe qui leur correspond
        clean_namedir : le nom du dossier où nous enregistrons les pistes nettoyées
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
    samp = check_samples(config)
    if samp :
        return samp[0], samp[1] #samp[0] : X et samp[1]: Y
    
    # Initialiser les varibales qui seront utilisés dans cette fonction seulement
    df, classes , class_dist , n_samples , prob_dist,nb_classe = Init(csv_namefile,clean_namedir)
    
    X=[]
    y=[]
    
    _min,_max = float('inf'), -float('inf') #pour comprendre la mise à l'échelle pour normaliser les valeurs de loss et acc
    for _ in tqdm(range(n_samples)):#boucle sur les n échantillons qu'on a calculé déjà 
    
        rand_class= np.random.choice(class_dist.index,p=prob_dist)#le choix de la classe est aléatoire chaque itération
        file = np.random.choice(df[df.label==rand_class].index)#recupération de le filename correspondant à la rand_class que nous avons généré
        rate , wav= wavfile.read(clean_namedir+'/'+file)#récupération de wavfile qui correspond au file
        label = df.at[file,'label']#récupération le label de la classe qui correspand au file récupéré
        rand_index=np.random.randint(0,wav.shape[0]-config.step )#Prendre une valeur du signal basée sur la longueur du audio file, échantillonner directement à cet index et prendre 1/10 s
        sample = wav[rand_index:rand_index+config.step]#l'échantillon est tiré du fichier wav, le début de l'échantillon est rand_index, la longeur de l'échantillon est step = 1/10s 
        #afin que le modèle puisse discerner très rapidement une classification différente 
        #rien d'autre que ces 1 / 10s est supprimé
        X_sample = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft,
                       winlen=0.032,winstep=0.015)#préparation de l'échantillon en utilisant la formule mfccs
        
        # MWI norm
        smin = np.amin(X_sample)
        smax = np.amax(X_sample)
        X_sample = (X_sample - np.mean(X_sample)) / (np.std(X_sample)+1e-8)
        
        
        _min=min(np.amin(X_sample), _min)#la valeur minimal de loss obtenue a chaque entrainement
        _max=max(np.amax(X_sample), _max)#la valeur maximal d'accuracy obtenue a chaque entrainement
        X.append(X_sample) #la matrice X contient les échantillons préparées
        y.append(classes.index(label))#la matrice Y contient les indices des labels récupérés au début de la loop
        
    config.min = _min #sauvegarde de la valeur minimal de loss comme attribut de la classe config
    config.max = _max #sauvegarde de la valeur maximal d'accuracy comme attribut de la classe config
    
    X ,y = np.array(X), np.array(y)#tourner x et y en arrays pour garder une trace de min et lmax
    # X = (X - _min) / (_max - _min)#Pour normaliser X 
    
    
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2], 1) #remodeler X sans modifier ses données pour l'adapter au modèle convolutionnel
    y = to_categorical(y,num_classes=nb_classe)#to_categorical : Convertit un vecteur de classe (entiers)(0-3) en matrice de classe binaire.
    
    config.data = (X , y)#sauenregistrer les x et y en tant qu'attribut data dans la classe de configuration pour une utilisation ultérieure
    with open(config.p_path , 'wb') as handle:
        pickle.dump(config, handle, protocol=2)#sauvegarde de la configuration utilisées dans la préparation des échantillons dans le dossier pickles4 pour une utilisation ultérieure
        
    with open(config.samples_path , 'wb') as handle:#sauvegarde de X et Y dans le dossier samples4 pour une utilisation ultérieure
        pickle.dump(config.data, handle, protocol=2)    
        
    return X,y ,nb_classe


def get_conv_model(input_shape,nb_classe): 

    """fonction pour la préparation de modele convolutionnel
    Args:
        input_shape : forme des données d'entrées de RN
        nb_classe : le nombre des classes utilisées dans l'entrainement

          
    Returns:
        renvoie le modele
    """ 
    model = Sequential()#Un modèle séquentiel convient à une pile de couches simples
    model.add(Conv2D(16,(3,3),activation='relu',strides=(1,1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32,(3,3),activation='relu',strides=(1,1),
                     padding='same'))
    model.add(Conv2D(64, (3,3),activation='relu',strides=(1,1),
                      padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',strides=(1,1),
                      padding='same'))
    
    model.add(MaxPool2D((4,4)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classe,activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    #categorical_crossentropy : Il s'agit de la classe métrique de crossentropie à utiliser 
                               #lorsqu'il existe plusieurs classes d'étiquettes (2 ou plus).
    #optimizer = Adam : efficace en termes de calcul, a peu de mémoire requise, est invariante 
                      #à la mise à l'échelle diagonale des gradients et convient bien aux problèmes importants en 
                      #termes de données / paramètres
    return model


def Train(model_path,X , y ,csv_namefile,clean_namedir,nb_classe):
    
    """fonction d'apprentissage : nous permet de former notre RN 
    Args:
        model_path :le chemin du dossier 'models4' où nous enregistrerons notre modèle formé
        X , Y : les matrices d'apprentissage
        csv_namefile: le nom du fichier excel où il y a la liste des noms de fichiers audio avec le libellé de la classe qui leur correspond
        clean_namedir : le nom du dossier où nous enregistrons les pistes nettoyées
        nb_classe : le nombre des classes utilisées dans l'entrainement
        
    Returns:
        Trace les courbes d'accuracy et loss de notre modele formé 
    """ 
    
    y_flat = np.argmax(y, axis=1)#np.argmax : Renvoie les indices des valeurs
    #maximales le long d'un axe , l'interet de y _flat c'est de récupéré les indices 
    #des classes : 0 1 2 3 , puisque y contient les indices des labels pour chaque échantillon donc les indices sont répété 
    # y = [[1. 0. 0. 0.] , y_flat = [0 3 2 ... 3 3 0] 
    #     [0. 0. 0. 1.]
    #     [0. 0. 1. 0.]
    #      ...


    input_shape = (X.shape[1],X.shape[2], 1 )#la forme des données d'entrées d"un CNN
    model = get_conv_model(input_shape,nb_classe) #récupéré le modele conv
    class_weight = compute_class_weight('balanced',np.unique(y_flat),y_flat)
    #copute_class_weight : Estimer les poids de classe pour les ensembles de données 
    #np.unique(y_flat) : pour récupérer les indices des classes dans y_flat sans répétition [0,1,2,3] 
    
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose =1, mode ='max',
                             save_best_only=True, save_weights_only=False, period=1)
    #Modelcheckpoint :Callback pour enregistrer le modèle Keras ou les poids de modèle à une certaine
    #fréquence.Dans ce cas , il est utilisé pour calculer l'accuracy et sauvegarder le dernier meilleur modèle en fonction de la quantité surveillée
    history = model.fit(X, y , epochs=20,batch_size=32,
                        shuffle =True, class_weight=class_weight, validation_split=0.1 , 
                        callbacks = [checkpoint])
 
    
    #history : Forme le modèle pour un nombre fixe d'époques avec une validation automatique 
    model.save(model_path)#Enregistre le modèle formé dans un fichier pour une utilisation ultérieure

    # Affichage graphiquement de L’évolution de l’erreur (loss function) 
    # et de l’erreur de classification ‘accuracy’

    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



# La phase d'apprentissage 
X , y , nb_classe= build_rand_feat(csv_namefile,clean_namedir,config) #récupérer les Matrices X et Y préparés par la fonction build_rand_feat
#appeler la fonction de formation du modèle
Train(config.model_path,X,y,csv_namefile,clean_namedir,nb_classe)














