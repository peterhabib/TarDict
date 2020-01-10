import os
import sys
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, ShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, homogeneity_score, adjusted_rand_score, \
    roc_auc_score, roc_curve, f1_score, auc, explained_variance_score, balanced_accuracy_score, cohen_kappa_score, \
    hinge_loss, matthews_corrcoef, fbeta_score, hamming_loss, jaccard_score, log_loss, multilabel_confusion_matrix, \
    precision_recall_fscore_support, precision_score, recall_score, zero_one_loss

from sklearn.ensemble.forest import RandomForestClassifier
import time



from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



in_file = sys.argv[1]
out_file = sys.argv[2]


df_seq = pd.read_csv('Drug_Path_Struct.csv', sep=',')[:]
in_SMILES = pd.read_csv(in_file, sep=',')
print("Welcome To Tardict")
print('Datasets have been loaded...')
Out_File = open(out_file,'w')
Out_File.write("Predicted_Drug\tSMILES\n")






# Join two datasets on structureId
model_f = df_seq[['target','ChemicalStructure']]
# print(model_f.head())
model_f = model_f.dropna()
in_SMILES = in_SMILES.dropna()


# Look at classification type counts
counts = model_f.target.value_counts()

# Get classification types where counts are over 1000
types = np.asarray(counts[(counts > 0)].index)

# Filter dataset's records for classification types > 1000
data = model_f[model_f.target.isin(types)]

# print(types)
#Data Preparation for Vectoraization
X_train, X_test,y_train,y_test = train_test_split(data['ChemicalStructure'], data['target'], test_size = 0.2, random_state = 1)


# Create a Count Vectorizer to gather the unique elements in sequence
vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))

# Fit and Transform CountVectorizer
vect.fit(X_train)


filename = 'finalized_model.sav'
# load the model from disk
model = joblib.load(filename)

# print(classification_report(y_true, y_pred))
for smile in in_SMILES['ChemicalStructure']:
    # print(smile)
    X_train_df = vect.transform([smile])
    y_pred = model.predict(X_train_df)
    print(y_pred, smile)
    Out_File.write("%s\t%s\n" % (y_pred[0],smile))




