import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

#Initialiser les vaiables utilisées dans les fonctions
csv_namefile = 'effets_guitare.csv' #le fichier excel 
clean_namedir = 'clean' #Le dossier des wavfile nettoyés
wavfiles_namedir = 'wavfiles' #le dossier des wavfiles avant nettoyage

 
def Init (csv_namefile,wavfiles_namedir):
    """Initialise les variables du programme
    Args:
        csv_namefile: le nom du fichier excel où il y a la liste des noms de fichiers audio avec le libellé de la classe qui leur correspond
        wavfiles_namedir: le nom de dossier où il y les audiofiles
           
    Returns:
        
        renvoie 3 variables qui seront utilisées dans les autres fonctions       
        df : dataframe contient les données dans le fichier excel 
        classes : contient les noms des classes qui seront utilisés dans l'apprentissage 
        nb_classe : le nombre des classes utilisées dans l'entrainement

    """   
    # Récupération de fichier Excel ou il y a le file name avec label correspond
    df = pd.read_csv(csv_namefile)
    df.set_index('fname',inplace=True)#df.set_index : Défini l'index DataFrame à l'aide des colonnes existantes.
    
    # Récupération des pistes et le calcul de leurs longueur
    for f in df.index : #index de 0 à 123 (nombre de wavfiles dans le fichier excel)
        rate, signal = wavfile.read(wavfiles_namedir +'/'+f)#Récupérer le wavfile
        df.at[f,'length'] = signal.shape[0]/rate#pour chaque wavfile , on calcule la longeur par la formule 
    
    
    #Récupération du labelle des pistes sans répition : Chorus , Nickel-Power , Phaser_,Reverb
    classes = list(df.label) #recupere les noms des classes existants sans répétition
    nb_classe = len(classes) #le nombre des classes utilisées

    
    
    return df, classes , nb_classe #ces 3 varibales sont utilisées dans les autres fonctions

# Tracage des diffents fonctions : mfcc , fft, ..  (les fonctions sont prédéfinie)     
def plot_signals(signals,nb_classe):
    """Tracage de Time series
    Args:
        signals : le signal à tracer 
        nb_classe : le nombre des classes utilisées dans l'entrainement
           
    Returns:
        trace Time series de chaque piste
    """   
    rows =(nb_classe//2) if (nb_classe%2==0) else (nb_classe//2 +1)
    #le calcule du nombre des lignes nécessaires pour présenter les classes 
    # en fonction du nombre des classes(2 figures par ligne)
    fig, axes = plt.subplots(nrows=rows, ncols=2, sharex=False,
                             sharey=True, figsize=(20,10))
    fig.suptitle('Time Series', size=20)
    i = 0
    for x in range(rows):
        for y in range(2):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
def plot_fft(fft,nb_classe):
    """Tracage de Fourier Transforms
    Args:
        fft : le signal généré par la fonction fft à tracer
        nb_classe : le nombre des classes utilisées dans l'entrainement

    Returns:
        trace Fourier Transforms de chaque piste
    """
    rows =(nb_classe//2) if (nb_classe%2==0) else (nb_classe//2 +1)
    #le calcule du nombre des lignes nécessaires pour présenter les classes 
    # en fonction du nombre des classes(2 figures par ligne)
    fig, axes = plt.subplots(nrows=rows, ncols=2, sharex=False,
                              sharey=True, figsize=(20,10))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(rows):
        for y in range(2):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
def plot_fbank(fbank,nb_classe):
    """Tracage Filter Bank Coefficients
    Args:
        fbank : le signal généré par la fonction fbank à tracer
        nb_classe : le nombre des classes utilisées dans l'entrainement
    
    Returns:
        trace Filter Bank Coefficients de chaque piste
    """
    rows =(nb_classe//2) if (nb_classe%2==0) else (nb_classe//2 +1)
    #le calcule du nombre des lignes nécessaires pour présenter les classes 
    # en fonction du nombre des classes(2 figures par ligne)
    fig, axes = plt.subplots(nrows=rows, ncols=2, sharex=False,
                              sharey=True, figsize=(20,10))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(rows):
        for y in range(2):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
def plot_mfccs(mfccs,nb_classe):
    """Tracage FMel Frequency Cepstrum Coefficients
    Args:
        mfccs : le signal généré par la fonction mfccs à tracer
        nb_classe : le nombre des classes utilisées dans l'entrainement
          
    Returns:
        trace Mel Frequency Cepstrum Coefficients de chaque piste
    """
    rows =(nb_classe//2) if (nb_classe%2==0) else (nb_classe//2 +1)
    #le calcule du nombre des lignes nécessaires pour présenter les classes 
    # en fonction du nombre des classes(2 figures par ligne)    
    fig, axes = plt.subplots(nrows=rows, ncols=2, sharex=False,
                              sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(rows):
        for y in range(2):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


def Cleaning(y, rate, threshold):
    """Le nettoyage des échantillons: calcule l'enveloppe du signal
    Args:
        y: le signal à nettoyer
        rate : Le débit du signal considéré définit le nombre de millions de transitions par seconde.
        threshold : le seuil minimal qu'un signal peut atteindre
           
    Returns:
        
        renvoie l'enveloppe du signal considéré pour qu'il soit appliqué au signal initial afin d'éliminer les amplitudes mortes( mask = enveloppe )
    """   
    mask=[]#liste des true et false depend du seuil 
       
    y=pd.Series(y).apply(np.abs)#Transforme le signal en serie entre 0 et 1 
    y_mean= y.rolling(window=int(rate/100),min_periods=1, center=True).mean()#(Provide rolling window calculations on every 1/10s of signal) 
  
    for mean in y_mean: #si la valeur du signal > le seuil , donc elle est acceptée sinon supprimée
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask #

def calc_fft(y, rate):
    """Fonction du calcul pour la fonction fft 
    Args:
        y: le signal à tracer
        rate : Le débit du signal considéré définit le nombre de millions de transitions par seconde.
           
    Returns:
        
        renvoie le signal en fonction de frequence
    """   
    n=len(y) # la longeur du signal
    freq =  np.fft.rfftfreq(n, d=1/rate) #fft.rfftfreq : Renvoie les fréquences d'échantillonnage de la transformée de Fourier discrète (pour une utilisation avec rfft, irfft).
    Y = abs(np.fft.rfft(y)/n) #fft.rfft : Calcule la transformée de Fourier discrète unidimensionnelle pour une entrée réelle.
    return [Y,freq] #retourne le couple Y et freq de chaque signal pour tracer le fft

def pie_chart(df):
    
    """Fonction du Tracage de pie_chart des pistes
        df: Trame de données précédemment initialisée à l'aide de la fonction Init         
    Returns:
        Trace la pie_chart des pistes existantes dans DF    
    """   
    class_dist = df.groupby(['label'])['length'].mean() #calcule da la longueur moyenne de les pistes regroupées par nom de classe utilisé dans le tracage de pie chart
    # Tracage de pie chart
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dist,labels=class_dist.index , autopct='%1.1f%%',shadow=False 
           , startangle=90)

    ax.axis('equal')
    plt.show()
    df.reset_index(inplace=True)

def built_plot_signal(df,classes,wavfiles_namedir,nb_classe):
    """Fonction du calcule et du tracage des fonctions ; fft , mfccs, fbank ..
        df: Trame de données précédemment initialisée à l'aide de la fonction Init    
        classes : les noms des classes précédemment récupérées de DF dans la fonction Init
        wavfiles_namedir: le nom de dossier où il y les audiofiles 
        nb_classe : le nombre des classes utilisées dans l'entrainement

    Returns:
        Trace fft , TS , MFCC, Fbank des classes.
    """   
    
# Initialisation des varibale pour le tracage 
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    
    for c in classes: #loop sur les noms des classes : Chorus , Nickel-Power , Phaser_ , Reverb
        wav_file= df[df.label == c].iloc[0,0]#on verifie si le label sélectionné par la loop est le meme que dans DF alors on recupere la premiere de col = 0 et row = 0
        signal, rate = librosa.load(wavfiles_namedir+'/'+wav_file, sr = 44100 )#on charge le wavfile correspondant au filename
    
        mask = Cleaning(signal, rate, 0.02) #Calculer le mask de wavfile récupéré
        signal = signal[mask] #Envelopper le signal par le mask calculé 

        #faire le tracage en utilisant le siganl nettoyé 
        signals[c] = signal
        fft[c] = calc_fft(signal, rate)

        bank = logfbank(signal[:rate],rate , nfilt=26 ,nfft=1103).T
        fbank[c] = bank
        mel = mfcc(signal[:rate],rate,numcep=13, nfilt=26 ,nfft=1103).T
        mfccs[c] = mel

  
    # Tracage des fonctions
    plot_signals(signals,nb_classe)
    plt.show()

    plot_fft(fft,nb_classe)
    plt.show()

    plot_fbank(fbank,nb_classe)
    plt.show()

    plot_mfccs(mfccs,nb_classe)
    plt.show()
 
def save_clean_wavfiles(df,clean_namedir, wavfiles_namedir):
    """Fonction qui pernet de nettoyer et enregistrer les pistes néttoyées dans le dossier 'clean4' 
        df: Trame de données précédemment initialisée à l'aide de la fonction Init    
        clean_namedir : le nom du dossier où nous enregistrons les pistes nettoyées
        wavfiles_namedir : le nom de dossier où il y les audiofiles
    Returns:
       les fichiers audio sont enregistrés dans le clean_namedir
    """   
    if len(os.listdir(clean_namedir)) == 0 :#si le dossier Clean est vide nous procédons au nettoyage
        for f in tqdm(df.fname):#Boucler sur les morceaux de chaque classe
            signal,rate = librosa.load(wavfiles_namedir +'/'+f,sr=16000)#recupéré le wavfile qui le correspond
            mask=Cleaning(signal,rate,0.02) #calcul du mask du signal récupéré
            wavfile.write(filename=clean_namedir +'/'+f,rate=rate,data=signal[mask]) #sauvegarder le wavfile nettoyé dans le dossier Clean4
 
# Initialiser les varibales    
df,classes ,nb_classe= Init(csv_namefile,wavfiles_namedir)
# Taracage de pie chart
pie_chart(df)
# Tracage des fonctions fft mfccs ..
built_plot_signal(df,classes,wavfiles_namedir,nb_classe)
# sauvegarde des wavfiles nettoyés
save_clean_wavfiles(df,clean_namedir, wavfiles_namedir)






