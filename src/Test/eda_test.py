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
    """Initialise les variables du programme
    Args:
        csv_namefile: le nom du fichier excel où il y a la liste des noms de fichiers audio de test avec le libellé de la classe qui leur correspond
        wavfiles_namedir: le nom de dossier où il y l'audiofile de test
           
    Returns:
        
        renvoie une variable qui sera utilisée dans les autres fonctions       
        df : dataframe contient les données de test dans le fichier excel 
   
    """   
        # Récupération de fichier Excel ou il y a le file name avec label correspond
    df = pd.read_csv(csv_namefile)#index de 0 à 23 (nombre de wavfiles dans le fichier excel)
    df.set_index('fname',inplace=True)#df.set_index : Défini l'index DataFrame à l'aide des colonnes existantes.

    # Récupération des pistes et le calcul de leurs longueur
    for f in df.index :#index de 0 à 24 (nombre de wavfiles dans le fichier excel)
        rate, signal = wavfile.read(wavfiles_namedir +'/'+f)#Récupérer le wavfile
        df.at[f,'length'] = signal.shape[0]/rate#pour chaque wavfile , on calcule la longeur par la formule 


    return df #la varibale est utilisée dans les autres fonctions

# Tracage des diffents fonctions : mfcc , fft, ..  (les fonctions sont prédéfinie)     
def plot_signals(signals):
    """Tracage de Time series
    Args:
        signals : le signal à tracer 
           
    Returns:
        trace Time series de chaque piste
    """   
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
    """Tracage de Fourier Transforms
    Args:
        fft : le signal généré par la fonction fft à tracer
           
    Returns:
        trace Fourier Transforms de chaque piste
    """
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
    """Tracage Filter Bank Coefficients
    Args:
        fbank : le signal généré par la fonction fbank à tracer
           
    Returns:
        trace Filter Bank Coefficients de chaque piste
    """
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
    """Tracage FMel Frequency Cepstrum Coefficients
    Args:
        mfccs : le signal généré par la fonction mfccs à tracer
           
    Returns:
        trace Mel Frequency Cepstrum Coefficients de chaque piste
    """
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

def Cleaning(y, rate, threshold):
    """Le nettoyage des échantillons: calcule l'enveloppe du signal de test
    Args:
        y: le signal de test à nettoyer
        rate : Le débit du signal considéré définit le nombre de millions de transitions par seconde.
        threshold : le seuil minimal qu'un signal peut atteindre
           
    Returns:
        renvoie l'enveloppe du signal considéré pour qu'il soit appliqué au signal initial afin d'éliminer les amplitudes mortes( mask = enveloppe )
    """   
    
    mask=[]#liste des true et false depend du seuil 
       
    y=pd.Series(y).apply(np.abs)#Transforme le signal en serie entre 0 et 1 
    y_mean= y.rolling(window=int(rate/10),min_periods=1, center=True).mean()#(Provide rolling window calculations on every 1/10s of signal) 
 
    for mean in y_mean:#si la valeur du signal > le seuil , donc elle est acceptée sinon supprimée
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(y, rate):
    """Fonction du calcul pour la fonction fft 
    Args:
        y: le signal de test à tracer
        rate : Le débit du signal considéré définit le nombre de millions de transitions par seconde.
           
    Returns:
        renvoie le signal en fonction de frequence
    """   
    n=len(y)# y = signal 
    n=len(y) # la longeur du signal
    freq =  np.fft.rfftfreq(n, d=1/rate) #fft.rfftfreq : Renvoie les fréquences d'échantillonnage de la transformée de Fourier discrète (pour une utilisation avec rfft, irfft).
    Y = abs(np.fft.rfft(y)/n) #fft.rfft : Calcule la transformée de Fourier discrète unidimensionnelle pour une entrée réelle.
    return [Y,freq] #retourne le couple Y et freq de chaque signal pour tracer le fft

def built_plot_signal(wavfiles_namedir,df,class_de_test):
    """Fonction du calcule et du tracage des fonctions ; fft , mfccs, fbank ..
        wavfiles_namedir : le nom de dossier où il y l'audiofile de test
        df: Trame de données de test précédemment initialisée à l'aide de la fonction Init    
        class_de_test : le noms de la classe de test
    Returns:
        Trace fft , TS , MFCC, Fbank de la classe de test
    """   
 
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

   
    wav_file= df[df.label == class_de_test].iloc[0,0]# Puisque on a une seule classe donc on recupere le filename de col = 0 et row = 0
    signal, rate = librosa.load(wavfiles_namedir+'/'+wav_file, sr = 44100 )#on charge le wavfile correspondant au filename
    
    mask = Cleaning(signal, rate, 0.0005)#Calculer le mask de wavfile récupéré
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

def save_clean_wavfiles(clean_namedir, wavfiles_namedir,df):
    """Fonction qui pernet de nettoyer et enregistrer la piste de test néttoyées dans le dossier 'clean_test' 
        df: Trame de données de test précédemment initialisée à l'aide de la fonction Init    
        clean_namedir : le nom du dossier où nous enregistrons la piste de test nettoyées
        wavfiles_namedir : le nom de dossier où il y l'audiofile de test
    Returns:
       le fichier audio est  enregistré dans le clean_namedir
    """   

    if len(os.listdir(clean_namedir)) == 0 :#si le dossier Clean_test est vide nous procédons au nettoyage
        for f in tqdm(df.fname):#Boucler sur les morceaux de la classe de test
            signal,rate = librosa.load(wavfiles_namedir +'/'+f,sr=16000)#recupéré le wavfile qui le correspond
            mask=Cleaning(signal,rate,0.0005) #calcul du mask du signal récupéré
            wavfile.write(filename=clean_namedir +'/'+f,rate=rate,data=signal[mask]) #sauvegarder le wavfile nettoyé dans le dossier Clean_test
   
# Initialiser les varibales      
df,class_dist = Init(csv_namefile,wavfiles_namedir)

# Tracage des fonctions fft mfccs ..
built_plot_signal(wavfiles_namedir,df,class_de_test)

# sauvegarde des wavfiles nettoyés
save_clean_wavfiles(clean_namedir, wavfiles_namedir,df)






