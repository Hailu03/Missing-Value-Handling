import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = pd.read_csv(f'IrisNan20.csv',delimiter=',',header=0) # read csv file
df_train_nan_mask = iris.isna()

iris = np.array(iris) # convert to numpy

# print shape of dataset
print("Shape of Iris dataset: ", iris.shape)

# divide data into features and labels
X = iris[:,0:4]
# X[np.isnan(X).any(axis=1)] = 0
iris_mean = np.nanmean(X, axis=0)
X = np.where(np.isnan(X), iris_mean, X)
y = iris[:,4]

df_train_nan_mask = df_train_nan_mask.to_numpy()
df_train_nan_mask = df_train_nan_mask[:,:4]
train_nan_mask = df_train_nan_mask*1.0

# reshape x_train_nan
x_train_nan = X.reshape((X.shape[0], X.shape[1], 1))
print(x_train_nan.shape)

# reshape train nan mask
df_train_nan_mask = train_nan_mask.reshape((train_nan_mask.shape[0], train_nan_mask.shape[1], 1))

# concat both
x_train_input = np.concatenate([x_train_nan, df_train_nan_mask],axis=2)

target_iris = pd.read_csv("Iris_Filling_Target.csv",delimiter=',',header=0)
target_iris = np.array(target_iris)

X_train = target_iris[:,:4]
x_train_target = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

train_ds = tf.data.Dataset.from_tensor_slices((x_train_input, x_train_target)).batch(28)
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

input_layer = tf.keras.layers.Input((4, 2))

origin = input_layer[:,:,0]
condition = tf.cast(input_layer[:,:,1], dtype=tf.bool)

x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

pre = x
x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.PReLU()(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.PReLU()(x)

x = x + pre

x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding="same")(x)
x = tf.keras.layers.Flatten()(x)

x = tf.where(condition, x, origin)

model_fill = tf.keras.models.Model(inputs=input_layer, outputs=x)
model_fill.summary()

epochs = 200
batch_size = 28

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, min_lr=0.0001),
]
model_fill.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mse"],
)
history = model_fill.fit(train_ds,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)

# metric = "mse"
# plt.plot(history.history[metric])
# plt.show()

x_train = np.concatenate([x_train_nan, df_train_nan_mask], axis=2)
x_train_fill = model_fill(x_train, training=False).numpy()
X_train = x_train_fill.reshape((X_train.shape[0], X_train.shape[1], 1))
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y)).batch(32)
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# model
# gnb = GaussianNB().fit(X_train,y)
#
# # predict
iris_test = pd.read_csv('Iris_test.csv',delimiter=',',header=0)
iris_test = np.array(iris_test)
X_test, y_test = iris_test[:,0:4],iris_test[:,4]
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# y_pred = gnb.predict(X_test)

# Build a classification model
input_layer = tf.keras.layers.Input((4, 1))

x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

pre = x

x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = x + pre

x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(3, activation="softmax")(x)
clf_model = tf.keras.models.Model(inputs=input_layer, outputs=x)
clf_model.summary()

epochs = 200
batch_size = 28

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, min_lr=0.0001),
]
clf_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = clf_model.fit(train_ds,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)

train_loss_output = history.history['loss']
# train_loss_output2 = history.history['pred_loss']
plt.title("Prediction Loss")
plt.plot(train_loss_output,c='b')
plt.show()

# predict

# Get the classification prediction output
y_pred = clf_model.predict(X_test)
# Convert the predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# accuracy
print("Accuracy: {}%".format(accuracy_score(y_pred_labels,y_test)*100))