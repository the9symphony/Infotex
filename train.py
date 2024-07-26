import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras import ops
import tldextract
from sklearn.metrics import precision_recall_fscore_support as score
from feature import X, X_eval, X_test, maxlen, y, y_eval, maxFeatures

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np

# Архитектура модели
# Простота обоснована техническими ограничениями моего компьютера
# Можно улучшить, используя несколько слоев BiLSTM c Attention
model = Sequential()
model.add(Embedding(maxFeatures, 128, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# Обучаем модель и сохраняем в файл model.keras. На моем компьютере потребовало 4 часа.
model.fit(X, y, batch_size=16, epochs=15)
model.save('model.keras')