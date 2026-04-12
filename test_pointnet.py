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


def hyperparameter_tuning(batch_size_num,learning_rate_num):
    print(f"\n>>> Training with batch size = {batch_size_num} and learning rate = {learning_rate_num}")

    train_features = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data_pointwise/16O_size800_train_features.npy')
    train_labels = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data_pointwise/16O_size800_train_labels.npy')
    print("Training data shape:", train_features.shape)
    print("Training label shape:", train_labels.shape)
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

#     best_model_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath="/mnt/home/singhp19/alpha/PointNet_10Be/training/best_model.keras",
#     monitor="val_sparse_categorical_accuracy",
#     mode="max",
#     save_best_only=True,
#     verbose=1,
# )

    # build and train event-wise classification model and plot learning curve
    model = create_pointnet_model(num_points=800, 
                          num_classes=5, 
                          num_dimensions=3, #for changing number of features
                          is_regression=False,
                          is_pointwise_prediction=True)
    #model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate_num),
                  metrics=["sparse_categorical_accuracy"])
    

    #checkpointing
    history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[early_stopping],verbose=2) #early stopping

    # best_epoch = np.argmax(history.history["val_sparse_categorical_accuracy"]) + 1
    best_val_acc = np.max(history.history["val_sparse_categorical_accuracy"])
    print(f"Best val_accuracy={best_val_acc:.4f}")   
    return best_val_acc


def run_experiment():
    batch_options = [128,256]
    lr_options = [3e-6,5e-6,6e-6,7.5e-6]
    # batch_options = [256]
    # lr_options = [6e-6]
    results = {}

    for bs, lr in itertools.product(batch_options,lr_options):
        val_acc = hyperparameter_tuning(bs,lr)
        results[(bs,lr)] = val_acc

    opt_params = max(results, key=results.get)
    print(f"\n Best parameter pair is {opt_params} with the accuracy of {results[opt_params]}")

    return opt_params


def train_best_model():
    batch_size_num, learning_rate_num = run_experiment()
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




def load_classfication_model():
    tf.keras.backend.clear_session() 
    test_features = np.load("/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data_pointwise/16O_size800_test_features.npy")
    test_features = test_features[:,:,:3]
    test_labels = np.load("/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data_pointwise/16O_size800_test_labels.npy")

    print("Shape of test_features:", test_features.shape)

    model = create_pointnet_model(num_points=800, 
                          num_classes=5, 
                          num_dimensions=3, #for changing number of features
                          is_regression=False,
                          is_pointwise_prediction=True)
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=5e-06),
                  metrics=["sparse_categorical_accuracy"])
    
    model.summary()
    loss, accuracy_d = model.evaluate(test_features,test_labels,verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * accuracy_d))
    print("Unique label values:", np.unique(test_labels))

    #model = tf.keras.models.load_model('/mnt/home/singhp19/alpha/PointNet_10Be/training/cp-0050.ckpt') #MODEL CHANGE
    model = tf.keras.models.load_model("/mnt/home/singhp19/alpha/PointNet_ATTPC/training/16O_w0_noise/best_model.keras")
    loss, accuracy_d = model.evaluate(test_features,test_labels,verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * accuracy_d))
    
    y_pred = model.predict(test_features)
    predicted_classes = np.argmax(y_pred, axis=-1)
    
    print("y_pred shape:", y_pred.shape)
    print("predicted_classes shape:", predicted_classes.shape)
    np.save("/mnt/home/singhp19/alpha/PointNet_ATTPC/predicted_classes.npy",predicted_classes)

    #––––––––––––––––––––––––––––––––––––––– uncomment if NOT pointwise ––––––––––––––––––––––––––––––––––––––––––––––––
    
    # f1 = f1_score(test_labels, predicted_classes, average='weighted')
    # f1_cl2 = f1_score(test_labels, predicted_classes, labels=[2], average='weighted')
    # cf_matrix = confusion_matrix(test_labels, predicted_classes)
    # #print(predicted_classes)
    
    # print(f"F1 Score: {f1}")
    # print(f"F1 Score Class 2: {f1_cl2}")
    # print("Confusion Matrix:")
    # #print(cm)

    # group_counts = ["{0:0.0f}".format(value) for value in
    #             cf_matrix.flatten()]
    # group_percentages = ["{0:.2%}".format(value) for value in
    #                  cf_matrix.flatten()/np.sum(cf_matrix)]
    # labels = [f"{v1}\n{v2}" for v1, v2 in
    #       zip(group_counts,group_percentages)]
    # labels = np.asarray(labels).reshape(6,6)
    # class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]  # Change to your labels

    # report = classification_report(test_labels, predicted_classes, target_names=class_names)
    # print(report)

    # svm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds')
    # svm.set_xlabel("Predicted Labels")
    # svm.set_ylabel("True Labels")
    # svm.set_xticklabels(class_names)
    # svm.set_yticklabels(class_names, rotation=0)
    # #svm = sns.heatmap(cm, annot=True,cmap='Reds',fmt='g') 
    # plt.savefig("/mnt/home/singhp19/alpha/PointNet_ATTPC/confusion_matrix.png")


    # #model.save(filepath='/mnt/home/singhp19/alpha/PointNet_10Be/model/') 
    # #reloaded_model = keras.models.load_model('model/')
    # #print('Successfully saved and loaded back model!')
    
    # np.save("/mnt/home/singhp19/alpha/PointNet_ATTPC/generate_cm/confusion_matrix.npy",cf_matrix) 
    # np.save("/mnt/home/singhp19/alpha/PointNet_ATTPC/generate_cm/exp_labels_pred.npy",predicted_classes)

    #–––––––––––––––––––––––––––––––––––––– uncomment out if pointwise –––––––––––––––––––––––––––––––––––––––––––––––––––
    y_true_flat = test_labels.reshape(-1)
    y_pred_flat = predicted_classes.reshape(-1)
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    svm = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    svm.set_xlabel("Predicted")
    svm.set_ylabel("True")
    svm.set_title("Point-wise Confusion Matrix")
    plt.savefig("/mnt/home/singhp19/alpha/PointNet_ATTPC/confusion_matrix_pointwise.png")

def run_model_experimnent(run_num,run_path):
    
    tf.keras.backend.clear_session() 
    noise_percnt = 5
    test_features = np.load(run_path)
    test_features = test_features[:,:,:3]
    
    print("Shape of test_features:", test_features.shape)
    
    model = create_pointnet_model(num_points=800, 
                          num_classes=6, 
                          num_dimensions=3, #for changing number of features
                          is_regression=False,
                          is_pointwise_prediction=False)
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=5e-06),
                  metrics=["sparse_categorical_accuracy"])
    
    model = tf.keras.models.load_model(f"/mnt/home/singhp19/alpha/PointNet_ATTPC/training/24Mg_w{noise_percnt}_noise/best_model.keras")
    
    y_pred = model.predict(test_features)
    predicted_classes = np.argmax(y_pred, axis=1)
    
    np.save(f"/mnt/scratch/singhp19/experiment_predicted/exp{run_num}_pred_w{noise_percnt}.npy",predicted_classes)
    

def run_loop():
    for run_num in tqdm.tqdm(range(30,31), desc="Running model on experimental runs:"):
        run_path = f"/mnt/scratch/singhp19/experiment_ml_test/run{run_num}_24Mg_size800_test_features.npy"
        # f"/mnt/scratch/singhp19/experiment_ml_test/run{run_num}_16O_size800_test_features.npy"
        if not os.path.exists(run_path):
            continue
        else:
            run_model_experimnent(run_num,run_path)
            
    
    
def generate_cm():
    cm = np.load("/mnt/home/singhp19/alpha/PointNet_ATTPC/generate_cm/confusion_matrix.npy")
    tr_labels = np.load("/mnt/research/attpc/E546/Pointnet_MLclassification/engine_training_data_longer/24Mg_size800_test_labels.npy") #needs to be on test labels
    pred_labels = np.load("/mnt/home/singhp19/alpha/PointNet_ATTPC/generate_cm/exp_labels_pred.npy")
    features = np.load("/mnt/research/attpc/E546/Pointnet_MLclassification/engine_training_data_longer/24Mg_size800_test_features.npy")
    counter=0

    true_elemnt = 4
    pred_elemnt = 5
    print(f"Element of intrest in confusion matrix: {cm[true_elemnt][pred_elemnt]}")

    with PdfPages(f"/mnt/home/singhp19/alpha/PointNet_ATTPC/inspect_offdg/plots_spy_me{true_elemnt}{pred_elemnt}_longer.pdf") as pdf:
        plt.figure(figsize=(7,7))
        plt.text(0.5, 0.5, f'Matrix element T = {true_elemnt} | P = {pred_elemnt}', fontsize=24, ha='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()

        for i in range(len(tr_labels)):
            if tr_labels[i]==true_elemnt and pred_labels[i]==pred_elemnt:
                fig = plt.figure(figsize=(7,7))
                ax = fig.add_subplot(projection='3d')
                img = ax.scatter(features[i,:,0], features[i,:,1], features[i,:,2])
                plt.xlim(-1,1)
                plt.ylim(-1,1)
                ax.set_zlim(-1,1)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # plt.savefig("/mnt/home/singhp19/alpha/PointNet_10Be/generate_cm/class0_0.png")  
                pdf.savefig()
                plt.close()

                counter+=1

            if counter==20:
                break   

if __name__ == '__main__':
    tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    #train_best_model()
    load_classfication_model()
    #run_loop()
    #generate_cm()