import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


# Initialiser les variables du programmes
csv_namefile = 'LaGrange-Guitars.csv' #le fichier excel de wavfile de test 
clean_namedir =  'clean_test'  #Le dossier des wavfile de test nettoyés
wavfiles_namedir = 'wavfiles_test' #le dossier des wavfiles de test avant nettoyage
class_de_test = 'LaGrange-Guitars'

def Init (csv_namefile,wavfiles_namedir):
    
    # Récupération de fichier Excel ou il y a le file name avec label correspond
    df = pd.read_csv(csv_namefile)#index de 0 à 23 (nombre de wavfiles dans le fichier excel)
    df.set_index('fname',inplace=True)#df.set_index : Défini l'index DataFrame à l'aide des colonnes existantes.

    # Récupération des pistes et le calcul de leurs longueur
    for f in df.index :#index de 0 à 24 (nombre de wavfiles dans le fichier excel)
        rate, signal = wavfile.read(wavfiles_namedir +'/'+f)#Récupérer le wavfile
        df.at[f,'length'] = signal.shape[0]/rate#pour chaque wavfile , on calcule la longeur par la formule 


    #calcule da la longueur moyenne de les pistes regroupées par nom de classe
    class_dist = df.groupby(['label'])['length'].mean()

    return df, class_dist #ces 2 varibales sont utilisées dans les autres fonctions

# Tracage des diffents fonctions : mfcc , fft, ..  (les fonctions sont prédéfinie)     
def plot_signals(signals):
    
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,squeeze =False,
                             sharey=True, figsize=(20,10))
    fig.suptitle('Time Series', size=20)
    i = 0
    for x in range(1):
        for y in range(1):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            
def plot_fft(fft):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,squeeze =False,
                              sharey=True, figsize=(20,10))
    fig.suptitle('Fourier Transforms', size=20)
    i = 0
    for x in range(1):
        for y in range(1):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            
def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,squeeze =False,
                              sharey=True, figsize=(20,10))
    fig.suptitle('Filter Bank Coefficients', size=20)
    i = 0
    for x in range(1):
        for y in range(1):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            
def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,squeeze =False,
                              sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=20)
    i = 0
    for x in range(1):
        for y in range(1):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


# Le nettoyage des échantillons: calcule l'enveloppe du signal
def Cleaning(y, rate, threshold):#y = signal à nettouyer , threshold = le seuil minimal qu'un signal peut atteindre
    mask=[]#liste des true et false depend du seuil 
       
    y=pd.Series(y).apply(np.abs)#Transforme le signal en serie entre 0 et 1 
    y_mean= y.rolling(window=int(rate/10),min_periods=1, center=True).mean()#(Provide rolling window calculations on every 1/10s of signal) 
 
    for mean in y_mean:#si la valeur du signal > le seuil , donc elle est acceptée sinon supprimée
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# Fonction du calcul pour fft , elle retourne le signal en fonction de freq
def calc_fft(y, rate):
    n=len(y)# y = signal 
    n=len(y) # la longeur du signal
    freq =  np.fft.rfftfreq(n, d=1/rate) #fft.rfftfreq : Renvoie les fréquences d'échantillonnage de la transformée de Fourier discrète (pour une utilisation avec rfft, irfft).
    Y = abs(np.fft.rfft(y)/n) #fft.rfft : Calcule la transformée de Fourier discrète unidimensionnelle pour une entrée réelle.
    return [Y,freq] #retourne le couple Y et freq de chaque signal pour tracer le fft

# Tracage de pie_chart des pistes
def pie_chart(class_dist,df):# df : dataframe , class_dist , c'est deux varibales sont déjà initialisées à l'aide de la fonction Init
    
    # Tracage de pie chart
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dist,labels=class_dist.index , autopct='%1.1f%%',shadow=False 
           , startangle=90)

    ax.axis('equal')
    plt.show()
    df.reset_index(inplace=True)

# Le calcule et le tracage des fonctions ; fft , mfccs, fbank ..
def built_plot_signal(wavfiles_namedir,df,class_de_test):#df et class_de_test: le nom de la classe de test, sont 2 varibales déjà initialiser 
    # Initialisation des varibale pour le tracage 
 
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

   
    wav_file= df[df.label == class_de_test].iloc[0,0]# Puisque on a une seule classe donc on recupere le filename de col = 0 et row = 0
    signal, rate = librosa.load(wavfiles_namedir+'/'+wav_file, sr = 44100 )#on charge le wavfile correspondant au filename
    
    mask = Cleaning(signal, rate, 0.005)#Calculer le mask de wavfile récupéré
    signal = signal[mask]#Envelopper le signal par le mask calculé 

    #faire le tracage en utilisant le siganl nettoyé
    signals[class_de_test] = signal
    fft[class_de_test] = calc_fft(signal, rate)
    bank = logfbank(signal[:rate],rate , nfilt=26 ,nfft=1103).T
    fbank[class_de_test] = bank
    mel = mfcc(signal[:rate],rate,numcep=13, nfilt=26 ,nfft=1103).T
    mfccs[class_de_test] = mel

  
    # Tracage des fonctions
    plot_signals(signals)
    plt.show()

    plot_fft(fft)
    plt.show()

    plot_fbank(fbank)
    plt.show()

    plot_mfccs(mfccs)
    plt.show()

# Enregistrement des pistes néttoyées dans le dossier 'clean_test' 
def save_clean_wavfiles(clean_namedir, wavfiles_namedir,df):

    if len(os.listdir(clean_namedir)) == 0 :#si le dossier Clean_test est vide nous procédons au nettoyage
        for f in tqdm(df.fname):#Boucler sur les morceaux de la classe de test
            signal,rate = librosa.load(wavfiles_namedir +'/'+f,sr=16000)#recupéré le wavfile qui le correspond
            mask=Cleaning(signal,rate,0.005) #calcul du mask du signal récupéré
            wavfile.write(filename=clean_namedir +'/'+f,rate=rate,data=signal[mask]) #sauvegarder le wavfile nettoyé dans le dossier Clean_test
   
# Initialiser les varibales      
df,class_dist = Init(csv_namefile,wavfiles_namedir)

# Taracage de pie chart
pie_chart(class_dist,df)

# Tracage des fonctions fft mfccs ..
built_plot_signal(wavfiles_namedir,df,class_de_test)

# sauvegarde des wavfiles nettoyés
save_clean_wavfiles(clean_namedir, wavfiles_namedir,df)






