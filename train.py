import csv
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = "MNIST/train"
test_path = "MNIST/test"

train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        color_mode="grayscale",
        target_size=(28, 28),
        batch_size=128,
        class_mode='categorical',
        shuffle=False)

test_generator = test_datagen.flow_from_directory(
        test_path,
        color_mode="grayscale",
        target_size=(28, 28),
        batch_size=128,
        class_mode='categorical',
        shuffle=False)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')],
)

history = model.fit(train_generator, epochs=3, validation_data=test_generator)

metrics = {key: history.history[key][-1] for key in
           ['val_accuracy', 'val_precision', 'val_recall']}

with open("metrics.json", 'w') as outfile:
    json.dump(metrics, outfile)

# =========== Plot confusion matrix ===========
y_pred = model.predict(test_generator)

# Convert predictions from one-hot encoding to integer labels
y_pred = np.argmax(y_pred, axis=1)

# Get true labels for the test set
y_true = test_generator.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

fig = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=np.unique(y_true))

fig.plot(cmap='Blues')
fig.figure_.suptitle('Confusion matrix on the test set')
fig.figure_.savefig('confusion_matrix.png')


# ========== Write loss values to csv ===========
loss_csv_filename = 'loss.csv'
loss_values = history.history['loss']

with open(loss_csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row to the CSV file
    writer.writerow(['loss'])

    # Write each float value as a row in the CSV file
    for loss in loss_values:
        writer.writerow([loss])
