# Des informations sur les ressources liés au strage
## Les étapes pour éxcuter le code
    1. Veuillez exécuter eda4.py pour le nettoyage des pistes , les pistes nettoyées seront enregistrés sur le dossier clean4 (vide)
    2. Veuillez exécuter model4.py pour l'entrainement de modele ( j'ai utilisé le modele convolutionnel car il est plus rapide et performant). Le modele et la configuration seront enregistrés respectivement dans les dossiers models4 et pickles4
    3. Veuillez exécuter predict4.py pour avoir le resultat de l'entrainement . un fichier excel "predictions_4pistes.csv" sera ajouté a votre répertoire.
    



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

    3.les changements demandés pour la semaine sont effectués:
    * Concernant l'optimisation du code, j'ai modifié tous les programmes python en ajoutant des fonctions pour organiser le code.
    * les fonctions demandées sont ajoutées au niveau de model4_validation_manuelle.py
     et predict4.py
         model4_validation_manuelle.py : ce programme utilise la validation manuelle pour l'apprentissage (validation_data)
         model4.py : ce programme utilise la validation automatique (validation_split)
    * Pour la partie de test (track_test) :

        Le dossier Test contient le dossier clean_test , wavfiles_test , le fichier excel des pistes et eda_test.py le programme qui nous permet de nettoyer  la piste de test "LaGronge-Guitars"
        ## Pour tester  :
            1.vous exécutez le programme "eda_test.py" pour le nettoyage 
            2.Puis vous exécutez le programme "predict_test.py" qui nous permet d'avoir la prédiction de test sous forme fichier EXCEL et le graphique de variation de ces prédictions de RN.
            PS : Vous devez former le RN avant de tester
    
## Modification de 20/05/2020 :
    1.l'ajout des commentaires dans tous les programmes.
    2.Correction du programme de Predict_test.py (ajout des fonctions de tracage et l'optimisation du code)
    3.Elimination des varibales globales 
    5.Correction du probleme du coupage des pistes en petits morceaux 

## Modification de 27/05/2020 :
    1.L'ajout des commentaires pour les fonctions 
    2.L'ajout de la fonction Plot_EMA pour le tracage de la moyenne mobille exponentielle
    3.L'ajout de Zooming & brushing dans les graphes 

 ## Modification de 03/06/2020 :
    1.L'ajout du dossier templates ou il y a les pages HTML 
    2.L'ajout de programme flask_prog.py pour tester flask : c'est un framework nous permet l'interaction d'un programme python avec les page HTML dans le dossier templates
    **Pour tester le prgramme :
        1.Exécutez le programme flask_prog.py dans un IDE.
        2.Dans le navigateur , tapez http://127.0.0.1:5000/music,quelques secondes et vous pouvez lire le fichier audio dans clean_test.

   


    





