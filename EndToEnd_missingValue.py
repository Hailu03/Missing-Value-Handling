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
x_train_input = np.concatenate([x_train_nan, df_train_nan_mask], axis=2)

target_iris = pd.read_csv("Iris_Filling_Target.csv",delimiter=',',header=0)
target_iris = np.array(target_iris)
print(target_iris.shape)
X_train = target_iris[:,:4]
y_train = target_iris[:,4]
x_train_target = X_train.reshape((X_train.shape[0], X_train.shape[1]))

train_ds = tf.data.Dataset.from_tensor_slices((x_train_input, (y_train, x_train_target))).batch(32)
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

iris_test = pd.read_csv('Iris_test.csv',delimiter=',',header=0)
df_test_nan_mask = iris_test.isna()
df_test_nan_mask = df_test_nan_mask.to_numpy()
df_test_nan_mask = df_test_nan_mask[:, :4]
df_test_nan_mask = df_test_nan_mask*1.0
df_test_nan_mask = df_test_nan_mask.reshape((df_test_nan_mask.shape[0], df_test_nan_mask.shape[1], 1))

iris_test = np.array(iris_test)
x_test_nan = iris_test[:, :4]
x_test_nan = x_test_nan.reshape((x_test_nan.shape[0], x_test_nan.shape[1], 1))

y_test = iris_test[:, 4]
x_test_input = np.concatenate([x_test_nan, df_test_nan_mask], axis=2)

# Build a filling model

input_layer = keras.layers.Input((4, 2))

origin = input_layer[:,:,0]
condition = tf.cast(input_layer[:,:,1], dtype=tf.bool)

x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)

pre = x
x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.PReLU()(x)
x = keras.layers.Dropout(0.2)(x)

x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.PReLU()(x)

x = x + pre

x = keras.layers.Dropout(0.2)(x)

x = keras.layers.Conv1D(filters=1, kernel_size=3, padding="same")(x)
x = keras.layers.Flatten()(x)
data_filled = tf.where(condition, x, origin)

# Build a classification model
x = keras.layers.Reshape((4, 1))(data_filled)

x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)

pre = x

x = keras.layers.Dropout(0.2)(x)

x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
x = keras.layers.Dropout(0.2)(x)

x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)

x = x + pre

x = keras.layers.Dropout(0.3)(x)

x = keras.layers.GlobalAveragePooling1D()(x)
pred = keras.layers.Dense(3, activation="softmax", name='pred')(x)

model = keras.models.Model(inputs=input_layer, outputs=[pred, data_filled])
model.summary()

epochs = 200
batch_size = 32

callbacks = [keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                               factor=0.5,
                                               patience=20,
                                               min_lr=0.0001)]

output1 = 'pred'
output2 = 'tf.where'
losses = {
	output1:"sparse_categorical_crossentropy",
	output2:"mse",
}
metrics = {
    output1:"sparse_categorical_accuracy",
}
lossWeights = {output1: 1.0, output2: 50.0}

model.compile(optimizer="adam",
              loss=losses, loss_weights=lossWeights,
              metrics=metrics)

history = model.fit(x=x_train_input,
                    y={output1: y_train, output2: x_train_target},
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(x_test_input, {output1: y_test}),
                    verbose=1)

train_loss_output1 = history.history['loss']
train_loss_output2 = history.history['pred_loss']
plt.title("MSE Loss")
plt.plot(train_loss_output1,c='g')
plt.show()
plt.title("Prediction Loss")
plt.plot(train_loss_output2,c='b')
plt.show()

# predict

# Get the classification prediction output
y_pred = model.predict(x_test_input)[0]
# Convert the predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
# accuracy
print("Accuracy: {}%".format(accuracy_score(y_pred_labels,y_test)*100))