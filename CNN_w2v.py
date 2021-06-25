import os
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.regularizers import l2
from gensim.models import FastText
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.initializers import Constant
from nltk.tokenize import word_tokenize
from keras.layers import Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from gensim.test.utils import common_texts, get_tmpfile
from sklearn.metrics import classification_metrics, f1_score
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, SGD


train = pd.read_csv("data/train.csv")
dev = pd.read_csv("data/dev.csv")
test = pd.read_csv("data/test.csv")

le = LabelEncoder()
le = le.fit(train.label.unique().tolist())

#save encodings to a file
with open("label_encoding.txt","w") as f:
    f.write(str({class_:value for value,class_ in enumerate(le.classes_)}))

train_x = train.text.tolist()
train_y =  le.transform(train.label.tolist())

valid_x =  dev.text.tolist()
valid_y = le.transform(dev.label.tolist())

test_x = test.label.tolist()

tokenizer_obj =  Tokenizer()

texts = []
lines = []
lines.extend(train_x)
lines.extend(valid_x)
lines.extend(test_x)

for line in lines:
    texts.append(word_tokenize(line))

trainX_token = []
for x in train_x:
    trainX_token.append(x)

validX_token = []
for x in valid_x:
    validX_token.append(x)

testX_token = []
for x in test_x:
    testX_token.append(x)


tokenizer_obj.fit_on_texts(texts)
word_index = tokenizer_obj.word_index

train_seq = tokenizer_obj.texts_to_sequences(trainX_token)
valid_seq = tokenizer_obj.texts_to_sequences(validX_token)
test_seq = tokenizer_obj.texts_to_sequences(testX_token)
w2v_file = set()
target_name = le.classes_.tolist()


print('Found {} unique tokens.'.format(len(word_index)))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def plot_confusion_matrix(cm,
                          trial_no,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):



    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Set2')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.5f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.5f}; misclass={:0.5f}'.format(accuracy, misclass))
    plt.savefig(f"results/Trial{trial_no}/confusion_matrix.png")


def objective(trial):
    try:
        os.mkdir(f"results/fastText/Trial{trial.number}"))
    except FileExistsError :
        pass
    embed_size = trial.suggest_categorical("vec_size",[100,200,300])
    max_seq_len = trial.suggest_categorical("max_seq_len",[32,64])
    w2v_epochs = trial.suggest_categorical("w2v_epochs",[5,10,15,20,25])
    win_size = trial.suggest_categorical("win_size",[2,3,4,5,6,7,8,9,10])
    sg = trial.suggest_categorical("model_type",[0,1])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    epochs = trial.suggest_categorical("epochs", [30, 40, 50, 60])

    fname = f"w2v_{embed_size}_{w2v_epochs}_{win_size}_{sg}.txt"
    if fname not in w2v_file:
        model = Word2Vec(texts, size=embed_size, window=win_size, min_count=1, workers=6,epochs=w2v_epochs)
        model.wv.save_word2vec_format(fname,binary=False)
        w2v_file.add(fname)
        del model

    embeddings_index = {}
    with open(fname) as f:
        for line in f:
          values = line.split().strip("\n")
          word = values[0]
          coefs = np.asarray(values[1:],dtype="float32")
          embeddings_index[word] = coefs

    X_train = pad_sequences(train_seq,maxlen=max_seq_len,padding="post",truncating="post")
    X_valid = pad_sequences(valid_seq,maxlen=max_seq_len,padding="post",truncating="post")
    X_test = pad_sequences(test_seq,maxlen=max_seq_len,padding="post",truncating="post")

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    start = time()
    EMBEDDING_DIM = embed_size
    num_classes = 3
    filters = 256
    kernel_size = 1
    dp1 = trial.suggest_uniform('dropout_rate1', 0.0, 1.0)
    l2_11 = l2(trial.suggest_uniform('l2_11', 0.0, 1.0))
    l2_12 = l2(trial.suggest_uniform('l2_12', 0.0, 1.0))
    act1 = trial.suggest_categorical('activation1', ['relu', 'elu','selu'])
    dp2 = trial.suggest_uniform('dropout_rate2', 0.0, 1.0)
    l2_21 = l2(trial.suggest_uniform('l2_21', 0.0, 1.0))
    l2_22 = l2(trial.suggest_uniform('l2_22', 0.0, 1.0))
    dp3 = trial.suggest_uniform('dropout_rate3', 0.0, 1.0)
    act2 = trial.suggest_categorical('activation2', ['relu', 'elu','selu']))
    l2_31 = l2(trial.suggest_uniform('l2_31', 0.0, 1.0))
    l2_32 = l2(trial.suggest_uniform('l2_32', 0.0, 1.0))


    model = Sequential()
    model.add(Embedding(num_words,
                      EMBEDDING_DIM,
                      embeddings_initializer=Constant(embedding_matrix),
                      input_length=max_length,
                      trainable=False))
    model.add(SpatialDropout1D(dp1))
    model.add(Conv1D(filters,
                  kernel_size,
                  padding='valid',
                  kernel_regularizer=l2_11,
                  bias_regularizer=l2_12,
                  activation=act1,
                  strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dp2))
    model.add(Dense(units = 256,
                  kernel_regularizer=l2_21,
                  bias_regularizer=l2_22))
    model.add(Dropout(dp3))
    model.add(Activation(act2))
    model.add(Dense(num_classes,
                  kernel_regularizer=l2_31,
                  bias_regularizer=l2_32))
    model.add(Activation('softmax'))
    lr = trial.suggest_loguniform("lr", 1e-5, 10e-3)),
    optimizer_dict = {
      'Adagrad': Adagrad(learning_rate = lr),
      'Adadelta': Adadelta(learning_rate = lr),
      'Adam': Adam(learning_rate = lr),
      'RMSprop': RMSprop(learning_rate = lr),
      'SGD': SGD(learning_rate = lr, nesterov=True)
    }
    optim = trial.suggest_categorical('Optimizer', list(optimizer_dict.keys()))
    model.compile(loss='categorical_crossentropy',
              optimizer = optimizer_dict[optim],
              metrics=['accuracy', f1_m])
    history = model.fit(X_train, Y_train,batch_size = batch_size,
                    epochs = epochs,
                    validation_data = (X_valid, Y_valid),
                    verbose=2)
    end = time()
    history = history.history



  valid_pred = np.argmax(model.predict(X_valid),axis=1)
  conf_mat = confusion_matrix(valid_y,valid_pred)
  f1  = f1_score(valid_y,valid_pred,average='weighted')
  with open(f"results/Trial{trial.number}/results.txt","w") as f:
        report = classification_report(valid_y,valid_pred,target_names=target_name,digits=6)
        f.write(report)
        f.write("\nClasswise accuracy:\n")
        class_acc = [] #calculate class wise accuracy
        for i in range(len(conf_mat)):
            class_acc.append(conf_mat[i][i]/conf_mat[i].sum()*100)
        f.write(str(class_acc)+"\n")
        f.write(f"weighted f1 score: {f_score}\n")

        # save hyper -parameters in same file
        f.write("------------------HyperParameters------------------\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Optimizer: {optim}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Max seq len: {max_seq_len}\n")
        f.write(f"Embedding size: {embed_size}\n")
        f.write(f"w2v epochs:{w2v_epochs}\n")
        f.write(f"window size; {win_size}\n")
        f.write(f"Different activations: {act1}, {act2}\n")
        f.write(f"Layerwise l2 : {l2_11,l2_12}, {l2_21,l2_21}, {l2_31},{l2_32}\n")
        f.write(f"Dropout : {dp1},{dp2},{dp3}\n")
        f.write(f"Time Taken: {(end_time-start_time)//60} min {(end_time-start_time)%60} secs\n\n")

    #plot accuracy and loss
    fig = plt.figure()
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.xlabel("epoch ===>")
    plt.ylabel("Loss")
    plt.legend(['train data','validation data'],loc="upper right")
    plt.savefig(f"results/Trial{trial.number}/loss.png")
    fig.clear()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("Model accuracy")
    plt.xlabel("epoch ===>")
    plt.ylabel("Accuracy")
    plt.legend(['train data','validation data'],loc="upper right")
    plt.savefig(f"results/Trial{trial.number}/accuracy.png")
    fig.clear()
    plt.close(fig)



    plot_confusion_matrix(cm = np.array(conf_mat),
                      trial_no = trial.number,
                      target_names = target_name,
                      normalize    = True,
                      title        = "Confusion Matrix")

    test_pred = model.predict(X_test)
    test_pred = np.argmax(test_pred,axis=1).tolist()
    test_pred_label = []
    for i in test_pred:
        test_pred_label.append(target_name[i])
    result = pd.DataFrame({"text":test_text,"label":test_pred_label})
    result.to_csv(f"results/Trial{trial.number}/result.csv")

    del model
    return f1
    
try:
    os.mkdir(f"results")
except FileExistsError :
     pass
     
start_time = time() #ending time for all tials
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, timeout=200000)
end_time =  time() # ending time for all trials

with open(f"results/best_trial_det.txt","w") as f:
  f.write(f"Best Trial: {study.best_trial.number}\n")
  for k, v in study.best_trial.params.items():
    f.write(f"{k}: {v}\n")
  f.write(f"Time taken for all trials: {(end_time-start_time)//60} min {(end_time-start_time)%60} secs\n\n")
