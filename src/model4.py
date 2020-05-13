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
csv_namefile = 'effets_guitare4.csv'  
clean_namedir =  'clean4' 
config = config()
def Init(csv_namefile,clean_namedir):
           
    #Téléchargement du fichier Excel qui contient le nom de la piste avec label qui le correspond       

    df = pd.read_csv(csv_namefile)
    df.set_index('fname', inplace=True)
    
    # Récuperer les échantions nettoyées et le calcul de la longeur de chaque piste
    for f in df.index:
        rate, signal = wavfile.read(clean_namedir+'/'+f)
        df.at[f, 'length'] = signal.shape[0]/rate

    # Récupérer les labelles des classes : Chorus , Nickel-Power , Reverb - Phaser_ 
    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()
    
    # Création des N sample , la probabilité de distribution et les choices en se basant sur prob_dist
    n_samples = 2* int(df['length'].sum()/0.1)
    prob_dist = class_dist / class_dist.sum()
    # choices = np.random.choice(class_dist.index, p=prob_dist)
    
    return df, classes , class_dist , n_samples , prob_dist
        

# Verifier si un modele existe déja pour éviter la répétition du travail 
def check_config():
    if os.path.isfile(config.p_path) :
        print('Loading existing data {} for model'.format(config.mode))
        with open(config.p_path,'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

# Verifier si il existe déjà des échantillons préparées ia pour éviter la répétition du travail 
def check_samples():
    if os.path.isfile(config.samples_path) :
        print('Loading existing samples {} for model'.format(config.mode))
        with open(config.samples_path,'rb') as handle:
            samp = pickle.load(handle)
            return samp
    else:
        return None
    
    
# Creation des échantillons    
def build_rand_feat(csv_namefile):
    
    # tmp = check_config()
    # if tmp:
    #    return tmp.data[0], tmp.data[1]
    
    samp = check_samples()
    if samp :
        return samp[0], samp[1]
    
    # Initialiser les varibales utilisées
    df, classes , class_dist , n_samples , prob_dist = Init(csv_namefile)
    
    X=[]
    y=[]
    _min,_max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class= np.random.choice(class_dist.index,p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate , wav= wavfile.read(clean_namedir+'/'+file)
        label = df.at[file,'label']
        rand_index=np.random.randint(0,wav.shape[0]-config.step )
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft )
        _min=min(np.amin(X_sample), _min)
        _max=max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
        
    config.min = _min
    config.max = _max 
    X ,y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2], 1)
    y = to_categorical(y,num_classes=4)
    
    config.data = (X , y)
    with open(config.p_path , 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
        
    with open(config.samples_path , 'wb') as handle:
        pickle.dump(config.data, handle, protocol=2)    
        
    return X,y


# Modele convolutionnel
def get_conv_model(input_shape):
    model = Sequential()
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
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(4,activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['acc'])
    return model


# La fonction qui nous permet de former notre RN 
def Train(csv_namefile):

    # La phase d'apprentissage 
    X , y = build_rand_feat(csv_namefile)
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1],X.shape[2], 1 )
    model = get_conv_model(input_shape)
    class_weight = compute_class_weight('balanced',np.unique(y_flat),y_flat)
    
    checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose =1, mode ='max',
                             save_best_only=True, save_weights_only=False, period=1)

    history = model.fit(X, y , epochs=10,batch_size=32,
                        shuffle =True, class_weight=class_weight, validation_split=0.1 , 
                        callbacks = [checkpoint])
    
   

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


    model.save(config.model_path)

Train(csv_namefile)














