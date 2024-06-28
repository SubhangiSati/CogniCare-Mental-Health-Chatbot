import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

with open(r'intents.json') as f: #wrapping
    data = json.load(f) #loading

df = pd.DataFrame(data['intents']) 
#Each tag contain multiple questions & answers so i want to sprate them
dic = {"tag":[], "patterns":[], "responses":[]} 
for i in range(len(df)): 
    ptrns = df[df.index == i]['patterns'].values[0] # dataframe[ith row] value from pattern column[single]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)): #iterate 1 by 1 
        dic['tag'].append(tag) 
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
df = pd.DataFrame.from_dict(dic) #create df from dictionary
#PRE-PROCESSING
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns']) #convert to integer index wise
tokenizer.get_config() #conf in dic (details)

vacab_size = len(tokenizer.word_index) #unique words 
print('number of unique words = ', vacab_size)

from tensorflow.keras.preprocessing.sequence import pad_sequences #i/p same length 
from sklearn.preprocessing import LabelEncoder #categorical to numerical

ptrn2seq = tokenizer.texts_to_sequences(df['patterns']) #seq of int
X = pad_sequences(ptrn2seq, padding='post') #padding for eqi length
print('X shape = ', X.shape) #row, pad

lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag']) #DETERMINE PARAMETER AND TRANSFORM INTO INTEGER
print('y shape = ', y.shape)
print('num of classes = ', len(np.unique(y)))


#BUILDING AND TRAINING MODEL
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout
from tensorflow.keras.utils import plot_model

model = Sequential()
model.add(Input(shape=(X.shape[1]))) #num of seq, "max seq length" 
model.add(Embedding(input_dim=vacab_size+1, output_dim=100, mask_zero=True)) #vec,inc complexity, 0 as padding
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32))
model.add(LayerNormalization())
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(y)), activation="softmax"))
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy']) #integer

model.summary()
model_history = model.fit(x=X,
                          y=y,
                          batch_size=10,
                          callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)],
                          epochs=40)
import re
import random

def model_responce(query): 
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', query)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)
        
    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0] #num to org tag , return list of only element in list
    responses = df[df['tag'] == tag]['responses'].values[0]

    print("you: {}".format(query))
    print("model: {}".format(random.choice(responses)))
# Save the trained model
model.save('your_model.h5')
