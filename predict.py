import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras import ops
import tldextract
from sklearn.metrics import precision_recall_fscore_support as score
from feature import X, X_eval, X_test, maxlen, y, y_eval, maxFeatures, test

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np

# Загружаем модель и делаем прогнозы
model = keras.saving.load_model("model.keras")
predictions = model.predict(X_test)

# Сохраняем прогнозы в файл predict.csv
predict = pd.DataFrame({'domain': test['domain'], 'is_dga': predictions.ravel() > 0.5})
predict['is_dga'] = predict['is_dga'].astype(int)
predict.to_csv('predict.csv', encoding='utf-8', index=False, header=True)