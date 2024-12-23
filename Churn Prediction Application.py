import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv(r"D:\FSDS Material\Dataset\Classification\Churn_Modelling.csv")

# Data Preprocessing
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Build ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train ANN
ann.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

# Save the trained model using pickle
with open('churn_model.pkl', 'wb') as model_file:
    pickle.dump(ann, model_file)

# Optionally, save the preprocessor as well (StandardScaler and LabelEncoder)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

with open('column_transformer.pkl', 'wb') as ct_file:
    pickle.dump(ct, ct_file)

print("Model and preprocessing objects saved as pickle files.")
