import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

#Initialiser les vaiables utilisées dans les fonctions
csv_namefile = 'effets_guitare4.csv'  
clean_namedir =  'clean4' 
wavfiles_namedir = 'wavfiles4'

# Initialiser les variables du programmes
def Init (csv_namefile,wavfiles_namedir):
    # Récupération de fichier Excel ou il y a le file name avec label correspond
    df = pd.read_csv(csv_namefile)
    df.set_index('fname',inplace=True)
    # Récupération des pistes et le calcul de leurs longueur
    for f in df.index :
        rate, signal = wavfile.read(wavfiles_namedir +'/'+f)
        df.at[f,'length'] = signal.shape[0]/rate


# Récupération du labelle des pistes sans répition : Chorus , Nickel-Power , Phaser_,Reverb
    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()

    
    
    return df, classes , class_dist 
df,classes,class_dist = Init(csv_namefile,wavfiles_namedir)

# Tracage des diffents fonctions : mfcc , fft, ..       
def plot_signals(signals):
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False,
                             sharey=True, figsize=(20,10))
    fig.suptitle('Time Series', size=20)
    i = 0
    for x in range(2):
        for y in range(2):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False,
                              sharey=True, figsize=(20,10))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(2):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False,
                              sharey=True, figsize=(20,10))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(2):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False,
                              sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(2):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# Le nettoyage des échantillons
def Cleaning(y, rate, threshold):
    mask=[]
       
    y=pd.Series(y).apply(np.abs)
    y_mean= y.rolling(window=int(rate/10),min_periods=1, center=True).mean()
    # Tracage de l'envelope    
    # plt.plot(y,color='blue' , label = c)
    # plt.plot(y_mean,color='red')
    # plt.legend()
    # plt.show()

    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# Fonction du calcul pour fft , elle retourne le signal en fonction de freq
def calc_fft(y, rate):
    n=len(y)
    freq =  np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return [Y,freq]

# Tracage de pie_chart des pistes
def pie_chart():
    
    # Tracage de pie chart
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dist,labels=class_dist.index , autopct='%1.1f%%',shadow=False 
           , startangle=90)

    ax.axis('equal')
    plt.show()
    df.reset_index(inplace=True)

# Le calcule et le tracage des fonctions ; fft , mfccs, fbank ..
def built_plot_signal(wavfiles_namedir):
    # Initialisation des varibale pour le tracage 
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

    for c in classes:
        wav_file= df[df.label == c].iloc[0,0]
        signal, rate = librosa.load(wavfiles_namedir+'/'+wav_file, sr = 44100 )
    
        mask = Cleaning(signal, rate, 0.005)
        signal = signal[mask]

        signals[c] = signal
        fft[c] = calc_fft(signal, rate)

        bank = logfbank(signal[:rate],rate , nfilt=26 ,nfft=1103).T
        fbank[c] = bank
        mel = mfcc(signal[:rate],rate,numcep=13, nfilt=26 ,nfft=1103).T
        mfccs[c] = mel

  
    # Tracage des fonctions
    plot_signals(signals)
    plt.show()

    plot_fft(fft)
    plt.show()

    plot_fbank(fbank)
    plt.show()

    plot_mfccs(mfccs)
    plt.show()

# Enregistrement des pistes néttoyées dans le dossier 'clean' 
def save_clean_wavfiles(clean_namedir, wavfiles_namedir):
    # Enregistrement des pistes nettoyées dans une nv dossier "Clean"
    if len(os.listdir(clean_namedir)) == 0 :
        for f in tqdm(df.fname):
            signal,rate = librosa.load(wavfiles_namedir +'/'+f,sr=16000)
            mask=Cleaning(signal,rate,0.005)
            wavfile.write(filename=clean_namedir +'/'+f,rate=rate,data=signal[mask])
    

pie_chart()
built_plot_signal(wavfiles_namedir)
save_clean_wavfiles(clean_namedir, wavfiles_namedir)






