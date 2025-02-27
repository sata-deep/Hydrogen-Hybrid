With this code, one can perform the hybrid machine learning, unsupervised learnig with Autoencoder (AE) and predict storage capacity via a MLP for a hydrogen storage data given in the file train.dat. 
Here are the **descriptions of the programs**:
1. B1.py: Uses AE+MLP approach
2. MLP-B1.py: property predicts directly with the MLP without any feature transformation.
3. Corr.py: Studies Pearson correlation between the latent space and the real features.
4. train.dat: Training data for 1483 materials [First column target, next 36 columns features]
   
**The directory Unknown-materials contains:**
1. program U.py to predict to the hydrogen storage capacity for the materials from only features:
2. set1.dat:[ TiAlN2, V2H2, Zr2TiAl, MgC, NLi]
3. set2.dat: [NMn2Ti,MgCHF, CAlB, MgCHF, MgMnVTi ]
copy these files to unknown.dat and run U.py
