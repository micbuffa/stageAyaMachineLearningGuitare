import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg4 import config
import matplotlib.pyplot as plt

#Initialiser les vaiables utilisées dans les fonctions
csv_namefile = 'effets_guitare.csv'#le fichier excel 
clean_namedir = 'clean4' #Le dossier des wavfile nettoyés
config = config()

def Init (csv_namefile,clean_namedir):
          
    # Téléchargement du fichier Excel qui contient le nom de la piste avec label qui le correspond       
    df = pd.read_csv(csv_namefile)
    df.set_index('fname', inplace=True)#df.set_index : Défini fname dans DataFrame à l'aide des colonnes existantes.
    
    # Récuperer les échantions nettoyées et le calcul de la longeur de chaque piste
    for f in df.index:#indice de 0 à 123 (nombre de wavfiles dans le fichier excel)
        rate, signal = wavfile.read(clean_namedir+'/'+f)#recupére les wavfiles nettoyés
        df.at[f, 'length'] = signal.shape[0]/rate #pour chaque wavfile nettoyé , on calcule la longeur par la formule
        
    # Récupérer les labelles des classes : Chorus , Nickel-Power , Reverb - Phaser_ 
    classes = list(np.unique(df.label))#recupere les noms des classes existants sans répétition
    class_dist = df.groupby(['label'])['length'].mean()#calcule da la longueur moyenne de les pistes regroupées par nom de classe
    
    # Création des N sample , la probabilité de distribution 
    n_samples = 2* int(df['length'].sum()/0.1)#le nombre des échantillons de 1/10s dans les wavfiles qui sont en fait possibles dans les signaux
    prob_dist = class_dist / class_dist.sum()#Les probabilités associées à chaque entrée (classe) 

    
    return df, classes , class_dist , n_samples , prob_dist #Init initilise les varibeles qui seront utilisées dans les autres fonctions



    
# Verifier si il existe déjà des échantillons préparées pour éviter la répétition du travail 
def check_samples(config):
    if os.path.isfile(config.samples_path) :#verifier si le dossier samples4(contient X et Y du modele) est vide ou non
        print('Loading existing samples {} for model'.format(config.mode))
        with open(config.samples_path,'rb') as handle:
            samp = pickle.load(handle)
            return samp
    else:
        return None
    
    

# Creation des échantillons    
def build_rand_feat(csv_namefile,clean_namedir,config):
    
    # Initialiser les varibales qui seront utilisés dans cette fonction seulement
    df, classes , class_dist , n_samples , prob_dist = Init(csv_namefile,clean_namedir)
         
    # Nous utilisons check_samples () car il ne retourne X et Y que s'ils sont déjà préparés (optimisation de la mémoire)
    samp = check_samples(config)
    if samp :
        X= samp[0]
        y= samp[1]
        #S'ils existent, nous coupons les X et Y en ensemble d'apprentissage et ensemble de validation (20%)
        train_X,valid_X , train_y , valid_y = train_test_split(X,y,test_size=0.2)
        return train_X,train_y , valid_X , valid_y 
    
    
    #sinon, nous préparons le X et le Y, puis divisés en ensemble de formation et ensemble de validation  
    X=[]
    y=[]
    train_X=[]
    train_y=[]
    valid_y=[]
    valid_X=[]
 
    # Construction des échantillons
    _min,_max = float('inf'), -float('inf')#pour comprendre la mise à l'échelle pour normaliser les valeurs de loss et acc
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
                       nfft = config.nfft )#préparation de l'échantillon en utilisant la formule mfccs
        _min=min(np.amin(X_sample), _min)#la valeur minimal de loss obtenue a chaque entrainement
        _max=max(np.amax(X_sample), _max)#la valeur maximal d'accuracy obtenue a chaque entrainement
        X.append(X_sample) #la matrice X contient les échantillons préparées
        y.append(classes.index(label))#la matrice Y contient les indices des labels récupérés au début de la loop
        
    config.min = _min #sauvegarde de la valeur minimal de loss comme attribut de la classe config
    config.max = _max #sauvegarde de la valeur maximal d'accuracy comme attribut de la classe config
    
    X ,y = np.array(X), np.array(y)#tourner x et y en arrays pour garder une trace de min et lmax
    X = (X - _min) / (_max - _min)#Pour normaliser X 
    
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2], 1)#remodeler X sans modifier ses données pour l'adapter au modèle convolutionnel
    y = to_categorical(y,num_classes=4)#to_categorical : Convertit un vecteur de classe (entiers)(0-3) en matrice de classe binaire.
    
    #Aprés préparartion des matrices , nous coupons les X et Y en ensemble d'apprentissage et ensemble de validation (20%)  
    train_X,valid_X , train_y , valid_y = train_test_split(X,y,test_size=0.2)
 
   # Le sauvegarde  de la configuration et les échantillons preéparées pour une utilisation ultérieure
    config.data = (X , y) #en tant que attribut data dans la classe de configuration dans le dossier pickles4
    samples = (X, y) #♦en tant que matrices dans le dossier samples4
    with open(config.p_path , 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
        
    with open(config.samples_path , 'wb') as handle:
        pickle.dump(samples, handle, protocol=2)    
    
       
    return train_X,train_y , valid_X , valid_y 




# Modele convolutionnel
def get_conv_model( input_shape): #input_shape : forme des données d'entrées de RN
    model = Sequential()#Un modèle séquentiel convient à une pile de couches simples
    model.add(Conv2D(16,(3,3),activation='relu',strides=(1,1),
                      padding='same', input_shape=input_shape))
    model.add(Conv2D(32,(3,3),activation='relu',strides=(1,1),
                      padding='same'))
    model.add(Conv2D(64, (3,3),activation='relu',strides=(1,1),
                      padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',strides=(1,1),
                      padding='same'))
    model.add(Conv2D(256,(3,3),activation='relu',strides=(1,1),
                      padding='same'))
    
    
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(4,activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    #categorical_crossentropy : Il s'agit de la classe métrique de crossentropie à utiliser 
                               #lorsqu'il existe plusieurs classes d'étiquettes (2 ou plus).
    #optimizer = Adam : efficace en termes de calcul, a peu de mémoire requise, est invariante 
                      #à la mise à l'échelle diagonale des gradients et convient bien aux problèmes importants en 
                      #termes de données / paramètres
    return model


# La fonction qui nous permet de former le RN en utilisant une set de validation manuelle
def Train(model_path,train_X,train_y,valid_X,valid_y,rndir_path):
    #rndir_path : le chemin (dans le fichier ‘rn-1’)  où on sauvegarde les poids à chaque itération si l’erreur en validation est inférieure à la plus petite déjà calculée

    callbacks_list= []#C'est une liste des checkpoint , Il ya deux : une pour  accuracy et l'autre pour loss
    y_flat = np.argmax(train_y, axis=1)#np.argmax : Renvoie les indices des valeurs
    #maximales le long d'un axe , l'interet de y _flat c'est de récupéré les indices 
    #des classes : 0 1 2 3 , puisque y_flat contient les indices des labels pour chaque échantillon donc les indices sont répété 
    # y = [[1. 0. 0. 0.] , y_flat = [0 3 2 ... 3 3 0] 
    #     [0. 0. 0. 1.]
    #     [0. 0. 1. 0.]
    #      ...
    
    input_shape = (train_X.shape[1],train_X.shape[2], 1 )#la forme des données d'entrées d"un CNN
    model = get_conv_model(input_shape) #récupéré le modele conv
    class_weight = compute_class_weight('balanced',np.unique(y_flat),y_flat)
    #copute_class_weight : Estimer les poids de classe pour les ensembles de données 
    #np.unique(y_flat) : pour récupérer les indices des classes dans y_flat sans répétition [0,1,2,3] 
    
    checkpoint_acc = ModelCheckpoint(model_path, monitor='val_acc', verbose =1, mode ='max',
      
                                     save_best_only=True, save_weights_only=False, period=1)
    # sauvegarde les poids dans le fichier ‘rn-1’ à chaque itération si l’erreur en validation est inférieure à la plus petite déjà calculée
    checkpoint_loss = ModelCheckpoint(model_path, monitor='val_loss', verbose =1, mode ='min',
                              save_best_only=True, save_weights_only=True, period=1)
    
    #Modelcheckpoint :Callback pour enregistrer le modèle Keras ou les poids de modèle à une certaine
    #fréquence.Dans checkpoint_acc , il est utilisé pour calculer l'accuracy et sauvegarder le dernier meilleur modèle en fonction de la quantité surveillée
    #Dans checkpoint_loss , il est utilisé pour calculer le poids d'erreur et sauvegarder le dernier plus petit erreur
    
    #sauvegarde les poids dans le fichier ‘rn-1’
    model.save_weights(rndir_path)
  
    #Ajouter les deux checkpoint dans la liste des callbacks pour qu'on puisse les passées ensemble dans model.fit
    callbacks_list.append(checkpoint_acc)
    callbacks_list.append(checkpoint_loss)

    history = model.fit(train_X, train_y , epochs=8,batch_size=32,
          shuffle =True, class_weight=class_weight, validation_data = (valid_X, valid_y), 
          callbacks = callbacks_list)
    #history : Forme le modèle pour un nombre fixe d'époques avec une validation manuelle
    
    model.save(model_path)#Enregistre le modèle formé dans un fichier pour une utilisation ultérieure
    
    #Evaluation de score sur notre validation set et training set
    print('\nEvaluation des scores :' )

    score_validation = model.evaluate(valid_X,valid_y,verbose=True)
    score_training = model.evaluate(train_X,train_y,verbose=True)
    print('Validation score :' )
    print(score_validation)
    print('Training score :' )
    print(score_training)

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
train_X, train_y , valid_X , valid_y = build_rand_feat(csv_namefile,clean_namedir,config)
#appeler la fonction de formation du modèle
Train(config.model_path,train_X,train_y,valid_X,valid_y,config.weight_path)





















