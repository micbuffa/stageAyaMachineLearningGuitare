 ## Explanation of how to run the programs 
 # Presentation of project directories
1. Dataset : contains the learning dataset, prepared by Dr. Michel W. It has 12 tracks, each corresponding to a learning class.
2. models : contains the trained model (SavedModel).
3. pickles : contains a configuration file corresponding to the parameters and data used in the pre-processing phase (MFCC configurations ... ).
4. samples : contains pre-processed samples.
5. templates : contains the HTML page used to view the test track results as well as js and css pages.
6. Test : contains the test tracks and excel files resulting from the Test_Tracks program.



 # Presentation of project programs
 1. Configuration_Class.py : contains a configuration class, it is used to avoid repetition of the configuration for each model.
 2. Data_Loading&Preprocessing.py : includes the two functions of data loading and pre-processing, the result of this program is pre-processed samples.
 3. Get_Model&Training.py : includes the two functions get_model and training, the first function contains the definition of the model used in the training. The result of this function is the loss/accuracy graph and a ready-to-use trained model. 
 4. Evaluation_Model.py : evaluates the trained model on the dataset used in the learning. The result of this program is the Confusion Matrix and Precision Score.
 5. Test_Tracks.py : evaluates the formed model on new tracks, the result of this program is 3 graphs and 3 excel files corresponding to the resulting graphs.


# The order of program execution
Data_Loading&Preprocessing.py --> Get_Model&Training.py --> Evaluation_Model.py OR Test_Tracks.py

1. Data_Loading&Preprocessing.py : 
- requirements : 
    * The *Dataset* name need to be provided in the code.
    * The necessary libraries are listed at the top of the program.

* Running this program on SPYDER IDE is faster than Google Colab.
* Every instructions is explained in the program , and each function has it own docstrings.


2. Get_Model&Training.py : 
 requirements : 
    - *path_samples* is the path of the file where the pre-processed samples were saved, it needs to be provided in the code.
    - *path_models*  is the path to the folder where to save the model after training, it needs to be provided in the code.
    - The necessary libraries are listed at the top of the program.

** Running this program on Google colab to use GPU for faster training.
** Every instructions is explained in the program , and each function has it own docstrings.


3. Evaluation_Model.py : 
requirements : 
    - The *Dataset* name need to be provided in the code. 
    - *path_models* is the path to the folder where the trained model is saved, it needs to be provided in the code.
    - The necessary libraries are listed at the top of the program.

** the resulted excel file *Evaluation_Model_Results.csv* is saved in the main folder *src*.
** the *confusion matrix* can be visualized in google colab and spyder
** Every instructions is explained in the program , and each function has it own docstrings.


4. Test_Tracks.py : 
requirements : 
    - The test track must be in the *Test* folder.
    - The name of the test track must be provided as *Track_name*.
    - *path_models* is the path to the folder where the trained model is saved, it needs to be provided in the code.
    - The necessary libraries are listed at the top of the program.
    
** the resulted excel files are saved in the folder *Test*.
** the graphics can be visualized in google colab and spyder or HTML PAGE
** Every instructions is explained in the program , and each function has it own docstrings.



# HTML page
- the Html page contains the graphs resulting from the Test_Tracks program, a music player of the test track and its wave representation.
- It can be launched from  *Templates->Page HTML de test->Projet.html*

*requirements : 
    1. The resulting excel files and the test track must be available in *https://github.com/micbuffa/stageAyaMachineLearningGuitare/tree/master/src/Test*.
    - To view other test tracks presented in Github, the *music_file* must be replaced by the name of the desired test track in *Templates->Page HTML de test->player.js*. 
    