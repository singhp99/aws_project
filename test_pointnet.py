"""
Script to benchmark PointNet implementation.

Author: pranjal
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pointnet import create_pointnet_model
from plotting import plot_learning_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import itertools
import tqdm
import os


def train_best_model():
    batch_size_num, learning_rate_num = 256, 6e-6
    print(f"\n>>> Training with optimised batch size = {batch_size_num} and learning rate = {learning_rate_num}")

    train_features = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data_pointwise/16O_size800_train_features.npy')
    train_labels = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data_pointwise/16O_size800_train_labels.npy')
    print("Training data shape:", train_features.shape)
    print("Training labels shape:", train_labels.shape)
    train_ds = tf.data.Dataset.from_tensor_slices((train_features[:,:,:3], train_labels))
    train_ds = train_ds.batch(batch_size=batch_size_num, drop_remainder=False)

    val_features = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data_pointwise/16O_size800_val_features.npy')
    val_labels = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data_pointwise/16O_size800_val_labels.npy')
    val_ds = tf.data.Dataset.from_tensor_slices((val_features[:,:,:3], val_labels))
    val_ds = val_ds.batch(batch_size=batch_size_num, drop_remainder=True)
    
    train_labels = train_labels.astype(np.int32)
    val_labels   = val_labels.astype(np.int32)
    
    #early stopping
    early_stopping = EarlyStopping(
        monitor = "val_sparse_categorical_accuracy",
        mode = "max",
        patience = 10,
        restore_best_weights = True,
    )

    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/mnt/home/singhp19/alpha/PointNet_ATTPC/training/16O_w0_noise/best_model.keras",
    monitor="val_sparse_categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

    # build and train event-wise classification model and plot learning curve
    model = create_pointnet_model(num_points=800, 
                                  num_classes=5, 
                                  num_dimensions=3, #for changing number of features
                                  is_regression=False,
                                  is_pointwise_prediction=True)

    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate_num),
                  metrics=["sparse_categorical_accuracy"])
    

    #checkpointing
    history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[early_stopping,best_model_callback],verbose=2) #early stopping
    plot_learning_curve(history, '/mnt/home/singhp19/alpha/PointNet_ATTPC/learning-curve.png',batch_size_num,learning_rate_num)

    best_epoch = np.argmax(history.history["val_sparse_categorical_accuracy"]) + 1
    best_val_acc = np.max(history.history["val_sparse_categorical_accuracy"])
    print(f"Best model was at epoch {best_epoch} with val_accuracy={best_val_acc:.4f}") 


    
if __name__ == '__main__':
    tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train_best_model()