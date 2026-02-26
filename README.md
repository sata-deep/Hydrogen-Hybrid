
<img width="846" height="361" alt="Workflow" src="https://github.com/user-attachments/assets/b7ac1273-3a95-4ea6-9929-91abf07e757a" />








With this code, one can perform the hybrid machine learning, unsupervised learnig with Autoencoder (AE) and predict storage capacity via a MLP for a hydrogen storage data given in the file train.dat. 
Here are the **descriptions of the programs**:
1. B1.py: Uses AE+MLP approach
2. MLP-B1.py: property predicts directly with the MLP without any feature transformation.
3. Corr.py: Studies the Pearson correlation between the features in the latent space and the real features.
4. train.dat: Training data for 1483 materials [First column target, next 36 columns features]
   
**The directory Unknown-materials contains:**
1. Program U.py to predict to the hydrogen storage capacity for the materials from only features:
2. set1.dat:[ TiAlN2, V2H2, Zr2TiAl, MgC, NLi]
3. set2.dat: [NMn2Ti,MgCHF, CAlB, MgCHF, MgMnVTi ]
->copy these files to unknown.dat and run U.py

**The directory LLM contains:**
1. Script to train the GPT-2 model and to save in a directory called TrainedModel. (python GPT-2.py)
2. Generate chemical formulas based on the loaded model (python Generators.py)
3. It should be noted that the generated materials depend on the parameters used.
4. To generate more materials at a time, change the parameters.
