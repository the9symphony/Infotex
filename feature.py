
import tldextract
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import pickle
# Читаем файлы с датасетами
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
eval = pd.read_csv('val.csv')

# Убираем лишнее
train = train.drop('gozi', axis=1)
train['target'] = 1 - pd.factorize(train['dga'])[0]
train = train.drop('dga', axis=1)

# Оставляем только домены верхнего уровня
train['tld'] = [tldextract.extract(d).domain for d in train['mortiscontrastatim.com']]
test['tld'] = [tldextract.extract(d).domain for d in test['domain']]
eval['tld'] = [tldextract.extract(d).domain for d in eval['domain']]

# Формируем датасеты
X, y = train['tld'], train['target']
X_eval, y_eval = eval['tld'], eval['is_dga']
X_test = test['tld']

# Назначаем цифру каждой букве в домене, считаем параметры maxlen и maxFeatures
# Загружаем dictionary букв в доменах, который использовался при обучении с помощью Pickle
with open('enumeration_dictionary.pkl', 'rb') as f:
    validChars = pickle.load(f)
maxFeatures = len(validChars) + 1
maxlen = np.max([len(x) for x in X])

# Применяем энумерацию ко всем доменам во всех датасетах
# Нижние строки можно представить в виде функции, но так, мне кажется, проще читать
X = [[validChars[y] for y in x] for x in X]
X = pad_sequences(X, maxlen=maxlen)
X_eval = [[validChars[y] for y in x] for x in X_eval]
X_eval = pad_sequences(X_eval, maxlen=maxlen)
X_test = [[validChars[y] for y in x] for x in X_test]
X_test = pad_sequences(X_test, maxlen=maxlen)