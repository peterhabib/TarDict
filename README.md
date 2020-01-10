#TarDict: RandomForestClassifier-based software predict Drug-Tartget interaction in human Based on drug simplified molecular-input line-entry system (SMILES)
TarDict, a RandomForestClassifier based-software predict the target pathway based on SMILES of chemical. TarDict receives SMILES and returns a list of the possible similar drug, then export list to the user the target pathways that drug contribute in. Training data set of 20442 entry and testing reveal %95 accuracy.

#How To Use
1. Build the model by running: python3 Pathdict_Learn.py

2. Start prediction: python3 Pathdict_predict.py Test_Data.csv Tardict.csv

3. check TarDict.csv ti see the results