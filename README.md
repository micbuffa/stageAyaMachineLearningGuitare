# Des informations sur les ressources liés au strage
## Les étapes pour éxcuter le code
    1. Veuillez éxcuter eda4.py pour le nettoyage des pistes , les pistes nettoyées seront enregistrés sur le dossier clean4 (vide)
    2. Veuillez éxcuter model4.py pour l'entrainement de modele ( j'ai utilisé le modele convolutionnel car il est plus rapide et performant). Le modele et la configuration seront enregistrés respectivement dans les dossiers models4 et pickles4
    3. Veuillez éxecuter predict4.py pour avoir le resultat de l'entrainement . un fichier excel "predictions_4pistes.csv" sera ajouté a votre répertoire.
    



        * Veuillez utiliser Spyder pour que vous pouvez visualiser les résultats des fonctions de tracage dans l'IDE
        * Wavfiles4 contient les 125 morceaux de piste (Chorus , Nickel-Power , Phaser_ , Reverb)


## Modification de 06/05/2020 :
    1. Kfold.py est equivalant à model4.py (étape 2 ) , la seule difference c'est que ce fichier utilise la technique Kfold validation. Le model entrainé sera enregistré dans le dossier kfold.
    2.J'ai ajouté à la classe config (cfg4) le chemin du dossier des échantillons préparées et enregistrées (samples4) et du dossier kfold.
    3.J'ai ajouté la matrice de confusion dans predict.py
    



## Modification de 10/05/2020 :
    1. j'ai modifié le programme de Kfold pour qu'il soit capable de faire 5 epochs par split
    la précision est augmenté à 92.16%
    2.La ré-organisation de predict4.py (sous forme des fonctions : Predicition , Init ..) et l'ajout de la fonction confusion_matrix_fct qui calcule et affiche la matrice de confusion
    
    





