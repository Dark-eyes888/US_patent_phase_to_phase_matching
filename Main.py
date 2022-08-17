import pandas as pd
import numpy as np
import random
import os
import re
import tensorflow as tf
from tensorflow.keras import Model, Input
from transformers import RobertaTokenizer, TFRobertaModel


print(' gpu ', len(tf.config.list_physical_devices('GPU')))


def set_seed(seed=123):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed = set_seed(456)

train_ds = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv')
cpc_codes = pd.read_csv('titles.csv')

cpc_codes.rename({'code': 'context'}, axis=1, inplace=True)
train_ds = train_ds.merge(cpc_codes[['context', 'title']], on='context', how='left')
test_ds = test_ds.merge(cpc_codes[['context', 'title']], on='context', how='left')

train_ds.title = train_ds.title.apply(lambda x: re.sub('[;,]', " ", x))
test_ds.title = test_ds.title.apply(lambda x: re.sub('[;,]', " ", x))

train_ds['anc'] = train_ds['anchor'].astype(str) + " " + train_ds.title.astype(str)
test_ds['anc'] = test_ds['anchor'].astype(str) + " " + test_ds.title.astype(str)

max_len = 128
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def create_data(id_, anchors, target, scores, train=True):
    input_ids = []
    attention_mask = []
    labels = []
    ids = []
    token = tokenizer.batch_encode_plus([(x[0], x[1]) for x in zip(anchors, target)],
                                        max_length=max_len,
                                        padding='max_length', truncation=True)
    for i in range(len(anchors)):
        input_ids.append(token['input_ids'][i])
        attention_mask.append(token['attention_mask'][i])
        ids.append(id_[i])
        if train:
            labels.append(scores[i])
    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ids': ids,
            }, scores


train_data, train_label = create_data(train_ds['id'], train_ds['anc'], train_ds['target'], train_ds['score'],
                                      train=True)
test_data, test_labels = create_data(test_ds['id'], test_ds['anc'], test_ds['target'], None, train=False)


def create_model():
    model_ids = Input(shape=(max_len,), dtype=tf.int32)
    model_mask = Input(shape=(max_len), dtype=tf.int32)
    model = TFRobertaModel.from_pretrained('roberta-base')

    x = model(input_ids=model_ids, attention_mask=model_mask)
    x = tf.keras.layers.GlobalMaxPooling1D()(x.last_hidden_state)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1)(x)
    roberta_model = Model([model_ids, model_mask], output)

    roberta_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])
    roberta_model.summary()
    return roberta_model


model = create_model()

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_mse',
                                             patience=5,
                                             mode='max',
                                             verbose=1,
                                             restore_best_weights=True)

model.fit((np.array(train_data['input_ids']), np.array(train_data['attention_mask'])), np.array(train_label).ravel(),
          epochs=15, shuffle=True, validation_split=0.2, callbacks=[callbacks])

predictions=model.predict((np.array(test_data['input_ids']),
                            np.array(test_data['attention_mask'])))
print('Predictions:', predictions)
