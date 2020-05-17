import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from cfg4 import config


#Initialiser les vaiables utilisées dans les fonctions
csv_namefile2 = 'effets_guitare4.csv'#le fichier excel des classes utilisées pour former le modèle
csv_namefile = 'LaGrange-Guitars.csv'#le fichier excel de la classe de test  
clean_namedir =  'Test/clean_test' #le chemin de dossier de wavfile nettoyé de la classe de test
prediction_namecsv ='predictions_test.csv' #Le nom de fichier excel résultant aprés la prédiction

# La fonction qui intitialise les variables 
def Init (csv_namefile):
    df = pd.read_csv(csv_namefile)# Téléchargement du fichier Excel qui contient le nom de la piste de test avec label qui le correspond    
    df2 = pd.read_csv(csv_namefile2)# Téléchargement du fichier Excel qui contient le nom des pistes d'apprentissage avec label qui le correspond    

    classes = list(np.unique(df2.label))#récupérer les labels des classes apartir de df sans répitition de ces labels
    fn2class = dict(zip(df.fname, df.label))#création d'un dictionnaire où clé : filename et valeur : label
    
    p_path = os.path.join('pickles4','conv.p')#récupérer le chemin de dossier pickles
    with open(p_path,'rb') as handle:
        config = pickle.load(handle)
    
    #charger le modele formé à partir le dossier models4
    model = load_model(config.model_path)

    return df , classes , fn2class,model,config




# La fonction qui nous retourne la prédiciton 
def build_predictions(clean_namedir,config,df,classes,model):

    y_pred = []# liste des ouputs (apres l'interpretation)
    fn_prob = {} #ensemble de filenames avec la moyenne de leurs probabilités q'ils les correspond (Apres prediction)
    
    
    print('Extracting features from audio')
    
    for fn in tqdm(os.listdir(clean_namedir)):#loop sur les filennames des morceaux de wavfile de test nettoyé
        rate, wav =  wavfile.read(os.path.join(clean_namedir,fn))#récupérer le wavfile
        y_prob = []#liste des probabilités pour un wavfile
    
        for i in range(0,wav.shape[0]-config.step, config.step):#loop sur la longueur de wavfile avec un pas de 1/10s
            sample = wav[i:i+config.step]#à chaque iteration , on récupére 1/10s de wavfile
            x = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft )#préparation de l'échantillon en utilisant la formule mfccs
            x =( x - config.min) / (config.max - config.min) #normaliser le X avec les valeurs min et max qui sont déjà calculées a la phase de l'apprentissage 
            x = x.reshape(1,x.shape[0],x.shape[1], 1)#remodeler X sans modifier ses données pour l'adapter au modèle convolutionnel
            y_hat = model.predict(x)#la probabilité de X  d'etre chaque classe , sa forme : [0.8954995 , 0,0256264, 0,1684461,0.2556898] ,chaque valeur correspond a une classe
            y_prob.append(y_hat)#rassembler les probabilités du même wavfile (chaque prob correspond au 1/10s de wavfiles encours)
            y_pred.append(np.argmax(y_hat))#y_pred : contient les indices des classes prédites ou la probabilité etait la plus élévé
   
                        
        fn_prob[fn] = np.mean(y_prob , axis = 0).flatten()
        #chaque fichier wav de 8s nous l'avons coupé en échantillons de 1 / 10s, chaque échantillon nous avons calculé la probabilité d'être l'une des classes
        # nous avons rassemblé ces probabilités dans une liste que nous avons appelée y_prob
        # donc en conclusion pour chaque fichier wav il y a une liste de probabilités
        # pour obtenir la probabilité moyenne du fichier wav pour chaque classe, on applique np.mean sur chaque colonne de probabilités correspondant à la même classe
   
    return y_pred , fn_prob
  
# La fonction qui rend l'interpretation de la prédiction calculée            
def Prediction(clean_namedir,prediction_namecsv,config,df,classes,model):
    # Construction des prédictions
    y_pred , fn_prob = build_predictions(clean_namedir,config,df,classes,model)
    #y_pred contient l'indice de la classe prédite pour chaque morceau de le wav de test


    # Enregistrement de résulat dans DF 
    y_probs =[]#la liste des probabilités
    for i, row in df.iterrows():#loop sur les lignes des DF (24 lignes)
        y_prob = fn_prob[row.fname]#On récupére la ligne des probilités pour chaque file name de morceau qui le correspond
        #fn_prob : disctionnaire clé/ valeur , filename/probabilités  ex : LaGrange-Guitars_0 : [0.82,0.01,0.16,0.23]
        y_probs.append(y_prob)#ajouer à la liste des probabilités ([val, val , val , val])
        for c , p in zip(classes,y_prob):# c boucle sur classes(Chorus , Nickel-Power , ..) , et p sur y_prob qu'on récupéré
            df.at[i, c] =p#pour le même fichier wav il y a 4 probabilités, chacune correspondant à une classe
   
    y_predicted_class = [classes[np.argmax(y)] for y in y_probs]#On récupére le nom de la classe correspond à la probabilité la plus élevé pour un morceau de wavfile
    df['Output_prediction'] = y_predicted_class#Ajouter un column dans DF "Output_prediction" où nous mettons la prédiction finale pour chaque fichier wav

    #Transformer DataFrame en un fichier csv pour visualiser les résultats
    df.to_csv(prediction_namecsv,index = False) 
    return y_predicted_class
    
#Le tracage des résultats de la prédiction
def plot_prediction(y_predicted_class):#Fonction à modifier
    
    # les valeurs de l'axe des abscisses
    x =['00:00','00:10','00:20','00:30','00:40','00:50','01:00',
        '01:10','01:20','01:30','01:40','01:50','02:00',
        '02:10','02:20','02:30','02:40','02:50','03:00','03:10',
        '03:20','03:30','03:40']
   
    #Les valeurs del’axe des ordonnées
    y = y_pred 
    plt.xticks(rotation=90, ha='right')
    # plotting the points  
    plt.plot(x, y, color='black', linewidth = 2,marker='.', 
             markerfacecolor='red', markersize=10) 
     
    #Le pas utilisées dans le graphe 
    plt.xlim(0.0,22.5)
    plt.xlabel('Temps (minute)') 
    plt.ylabel('Classes') 
    plt.title('la variation des prédictions du RN (test)') 

    plt.show()

    




 
# Initialiser les variables à l'aide de la fonction Init 
df , classes , fn2class , model, config = Init(csv_namefile)   

#Récuperer y_pred pour tracer les resultats 
y_pred = Prediction(clean_namedir,prediction_namecsv,config,df,classes,model)

#Tracage de la variation de la prédiction
plot_prediction(y_pred)
    
