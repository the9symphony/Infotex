import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras import ops
import tldextract
from sklearn.metrics import precision_recall_fscore_support as score
from feature import maxlen, X, X_eval, y_eval

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np


# Загружаем модель и делаем прогнозы
model = keras.saving.load_model("model.keras")
predictions = model.predict(X_eval)

# Считаем показатели
tn, fp, fn, tp = confusion_matrix(y_eval, predictions > 0.5).ravel().astype(int)
precision, recall, fscore, support = score(y_eval, predictions > 0.5)

# Сохраняем показатели в файл validation.txt
f = open('validation.txt','w')
print('True positive: ' + str(tp), file=f)
print('False positive: ' + str(fp), file=f) 
print('False negative: ' + str(fn), file=f) 
print('True negative: ' + str(tn), file=f) 
print('Accuracy: ' + str((tp + tn) / (tp + tn + fp + fn)), file=f) 
print('Precision: ' + str(precision), file=f) 
print('Recall: ' + str(recall), file=f) 
print('F1: ' + str(fscore), file=f) 