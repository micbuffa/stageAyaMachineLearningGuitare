import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from cfg4 import config

#Initialiser les vaiables utilisées dans les fonctions
csv_namefile = 'effets_guitare4.csv'#le fichier excel 
clean_namedir = 'clean4' #Le dossier des wavfile nettoyés
prediction_namecsv ='predictions_4pistes.csv' #Le nom de fichier excel résultant aprés la prédiction

# La fonction qui intitialise les variables 
def Init (csv_namefile):
    df = pd.read_csv(csv_namefile) # Téléchargement du fichier Excel qui contient le nom de la piste avec label qui le correspond       

    classes = list(np.unique(df.label))#récupérer les labels des classes apartir de df sans répitition de ces labels
    fn2class = dict(zip(df.fname, df.label))#création d'un dictionnaire où clé : filename et valeur : label
   
    p_path = os.path.join('pickles4','conv.p')#récupérer le chemin de dossier pickles 
    with open(p_path,'rb') as handle:
        config = pickle.load(handle)
    
    #charger le modele formé à partir le dossier models4
    model = load_model(config.model_path)

    return df , classes , fn2class,model,config



# La fonction qui nous retourne la prédiciton 
def build_predictions(clean_namedir,config,df,classes,model):

    y_true = [] #liste des inputs (avant l'interpretation)
    y_pred = [] # liste des ouputs (apres l'interpretation)
    fn_prob = {} #ensemble de filenames avec la moyenne de leurs probabilités q'ils les correspond (Apres prediction)
    
    print('Extracting features from audio')
    
    for fn in tqdm(os.listdir(clean_namedir)):#loop sur les filennames des wavfiles nettoyés
        rate, wav =  wavfile.read(os.path.join(clean_namedir,fn))#récupérer le wavfile
        label = fn2class[fn] #récupérer le label de la classe qui le correspond 
        c = classes.index(label)#récupérer l'indice de label récupéré 
        y_prob = []#liste des probabilités pour un wavfile
    
        for i in range(0,wav.shape[0]-config.step, config.step):#loop sur la longueur de wavfile avec un pas de 1/10s
            sample = wav[i:i+config.step]#à chaque iteration , on récupére 1/10s de wavfile
            x = mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,
                       nfft = config.nfft ) #préparation de l'échantillon en utilisant la formule mfccs
            x =( x - config.min) / (config.max - config.min) #normaliser le X avec les valeurs min et max qui sont déjà calculées a la phase de l'apprentissage
            x = x.reshape(1,x.shape[0],x.shape[1], 1)#remodeler X sans modifier ses données pour l'adapter au modèle convolutionnel
            y_hat = model.predict(x)#la probabilité de X  d'etre chaque classe , sa forme : [0.8954995 , 0,0256264, 0,1684461,0.2556898] ,chaque valeur correspond a une classe
            y_prob.append(y_hat)#rassembler les probabilités du même wavfile (chaque prob correspond au 1/10s de wavfiles encours)
            y_pred.append(np.argmax(y_hat))#y_pred : contient les indices des classes prédites ou la probabilité etait la plus élévé
   
            y_true.append(c)#y_true : contient l'indice de la bonne classe à prévoir pour chaque fichier wav
            
            
        fn_prob[fn] = np.mean(y_prob , axis = 0).flatten()
        #chaque fichier wav de 8s nous l'avons coupé en échantillons de 1 / 10s, chaque échantillon nous avons calculé la probabilité d'être l'une des classes
        # nous avons rassemblé ces probabilités dans une liste que nous avons appelée y_prob
        # donc en conclusion pour chaque fichier wav il y a une liste de probabilités
        # pour obtenir la probabilité moyenne du fichier wav pour chaque classe, on applique np.mean sur chaque colonne de probabilités correspondant à la même classe
        

    return y_true, y_pred , fn_prob 
  
# La fonction qui rend l'interpretation de la prédiction calculée            
def Prediction(clean_namedir,prediction_namecsv,config,df,classes,model):
    # Construction des prédictions
    y_true , y_pred , fn_prob = build_predictions(clean_namedir,config,df,classes,model)
    #y_true contient l'indice de la bonne classe à prévoir pour chaque fichier wav
    #y_pred contient l'indice de la classe prédite pour chaque fichier wav

    # Calcul d' Accuracy score : pour calculer le score de précision, nous avons besoin de données réelles et de données prédites
    acc_score = accuracy_score(y_true= y_true, y_pred = y_pred)

    print("\n")
    print("Accuracy score =", acc_score*100 ,"%")
    print("\n")

    # Enregistrement de résulat dans DF 
    y_probs =[]#la liste des probabilités
    for i, row in df.iterrows():#loop sur les lignes des DF (124 lignes)
        y_prob = fn_prob[row.fname] #On récupére la ligne des probilités pour chaque file name qui le correspond
        #fn_prob : disctionnaire clé/ valeur , filename/probabilités  ex : Chorus_0 : [0.82,0.01,0.16,0.23]
        y_probs.append(y_prob) #ajouer à la liste des probabilités ([val, val , val , val])
        for c , p in zip(classes,y_prob):# c boucle sur classes(Chorus , Nickel-Power , ..) , et p sur y_prob qu'on récupéré
            df.at[i, c] =p #pour le même fichier wav il y a 4 probabilités, chacune correspondant à une classe
   
    
    y_predicted_class = [classes[np.argmax(y)] for y in y_probs] #On récupére le nom de la classe correspond à la probabilité la plus élevé pour un wavfile
    df['Output_prediction'] = y_predicted_class #Ajouter un column dans DF "Output_prediction" où nous mettons la prédiction finale pour chaque fichier wav

    #Transformer DataFrame en un fichier csv pour visualiser les résultats
    df.to_csv(prediction_namecsv,index = False) 
    
    return y_true , y_pred #retourner y_true et y_pred pour calculer la matrice de confusion



# Tracage de la matrice de confusion (fonction prédéfinie)
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Calcule de la matrice de confusion
def confusion_matrix_fct(y_true, y_pred): #Cette fonction est besoin de y_pred et y_true
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix for multi-class')


# Initialiser les variables à l'aide de la fonction Init 
df , classes , fn2class , model,config = Init(csv_namefile)
#Récuperer y_predict et y_true et calculer la précision  
y_true , y_pred = Prediction(clean_namedir,prediction_namecsv,config,df,classes,model)
confusion_matrix_fct(y_true,y_pred)
    





