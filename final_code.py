#code link:https://colab.research.google.com/drive/1Ku1hu1B_Z4wf1unhpTE_pHTzEyeobUP9?usp=sharing

import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
from zipfile import ZipFile

# Seed setting for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Model creation using MobileNetV2 as the base
def create_model():
    base_model = applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Pooling layer to reduce dimensionality
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Dense layer for learning
    x = BatchNormalization()(x)  # Normalize activations
    x = Dropout(0.5)(x)  # Dropout for regularization
    outputs = Dense(2, activation='softmax')(x)  # Output layer with softmax activation
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to load and preprocess dataset
def load_dataset(data_path):
    X = []
    y = []
    for filename in os.listdir(data_path):
        label = 0 if filename.startswith('A') else 1
        image_path = os.path.join(data_path, filename)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array /= 255.0  # Normalize images
        X.append(img_array)
        y.append(label)
    return np.array(X), np.array(y)

def main():
    set_seed()  # Apply seed

    # Extract dataset
    zip_file_path = '/content/project_2_dataset.zip'
    extraction_directory = '/content/dataset'
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_directory)

    # Load and split dataset
    data_path = '/content/dataset/train'
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = create_model()
    # Setup callbacks for training efficiency
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[checkpoint, early_stopping, reduce_lr])

    # Fine-tuning last layers of the model
    model = create_model()
    for layer in model.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, callbacks=[checkpoint, reduce_lr, early_stopping])

    # Load best performing model
    model = tf.keras.models.load_model('best_model.h5')

    # Evaluate model performance
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    cm = confusion_matrix(y_test, y_pred_classes)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred_classes))
    print(f'Confusion Matrix:\n{cm}')
    print(f'True Positives: {cm[1, 1]}')
    print(f'True Negatives: {cm[0, 0]}')
    print(f'False Positives: {cm[0, 1]}')
    print(f'False Negatives: {cm[1, 0]}')

if _name_ == "_main_":
    main()

