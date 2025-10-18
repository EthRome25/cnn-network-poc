# 1. Imports
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")

def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])
    return pd.DataFrame({'Class Path': class_paths, 'Class': classes})


def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])
    return pd.DataFrame({'Class Path': class_paths, 'Class': classes})


tr_df = train_df('input_data/Training')
ts_df = test_df('input_data/Testing')

# Split test into valid/test
valid_df, ts_df = train_test_split(ts_df, train_size=0.5, random_state=20, stratify=ts_df['Class'])

# Keep at most N samples per class for a quick CPU demo
N = 80  # adjust between 40–200 depending on speed vs. accuracy


def balance_limit(df, n):
    return (df.groupby('Class', group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), n), random_state=42))
              .reset_index(drop=True))


tr_df_small = balance_limit(tr_df, N)
valid_df_small = balance_limit(valid_df, max(20, N//4))
ts_df_small = balance_limit(ts_df, max(20, N//4))

print('Subset sizes:', len(tr_df_small), len(valid_df_small), len(ts_df_small))

batch_size = 8
img_size = (128, 128)

_gen = ImageDataGenerator(rescale=1/255.0)  # light preprocessing only
ts_only = ImageDataGenerator(rescale=1/255.0)

tr_gen = _gen.flow_from_dataframe(tr_df_small, x_col='Class Path', y_col='Class',
                                  batch_size=batch_size, target_size=img_size, shuffle=True)

valid_gen = _gen.flow_from_dataframe(valid_df_small, x_col='Class Path', y_col='Class',
                                     batch_size=batch_size, target_size=img_size)

ts_gen = ts_only.flow_from_dataframe(ts_df_small, x_col='Class Path', y_col='Class',
                                     batch_size=batch_size, target_size=img_size, shuffle=False)

class_dict = tr_gen.class_indices
classes = list(class_dict.keys())

img_shape = (128, 128, 3)
base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                               input_shape=img_shape, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False  # freeze backbone for fast CPU training

model = Sequential([
    base_model,
    Dropout(0.2),
    Dense(4, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

model.summary()

hist = model.fit(
    tr_gen,
    epochs=6,               # quick CPU demo
    validation_data=valid_gen,
    shuffle=True,
)
history = hist.history
_ = history.keys()

train_score = model.evaluate(tr_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(ts_gen, verbose=1)

print(f"Train Loss: {train_score[0]:.4f}")
print(f"Train Accuracy: {train_score[1]*100:.2f}%")
print('-' * 20)
print(f"Validation Loss: {valid_score[0]:.4f}")
print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
print('-' * 20)
print(f"Test Loss: {test_score[0]:.4f}")
print(f"Test Accuracy: {test_score[1]*100:.2f}%")

# Safely extract metrics (in case some are missing)
tr_acc = history.get('accuracy', [])
tr_loss = history.get('loss', [])
tr_per = history.get('precision', [])
tr_recall = history.get('recall', [])
val_acc = history.get('val_accuracy', [])
val_loss = history.get('val_loss', [])
val_per = history.get('val_precision', [])
val_recall = history.get('val_recall', [])

# Determine number of epochs from available series
num_epochs = max(len(tr_acc), len(tr_loss), len(val_acc), len(val_loss))
Epochs = [i + 1 for i in range(num_epochs)]

plt.figure(figsize=(18, 10))
plt.style.use('fivethirtyeight')

# Loss
plt.subplot(2, 2, 1)
if len(tr_loss):
    plt.plot(Epochs[:len(tr_loss)], tr_loss, 'r', label='Training loss')
if len(val_loss):
    plt.plot(Epochs[:len(val_loss)], val_loss, 'g', label='Validation loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(2, 2, 2)
if len(tr_acc):
    plt.plot(Epochs[:len(tr_acc)], tr_acc, 'r', label='Training accuracy')
if len(val_acc):
    plt.plot(Epochs[:len(val_acc)], val_acc, 'g', label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Precision
plt.subplot(2, 2, 3)
if len(tr_per):
    plt.plot(Epochs[:len(tr_per)], tr_per, 'r', label='Precision')
if len(val_per):
    plt.plot(Epochs[:len(val_per)], val_per, 'g', label='Val Precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Recall
plt.subplot(2, 2, 4)
if len(tr_recall):
    plt.plot(Epochs[:len(tr_recall)], tr_recall, 'r', label='Recall')
if len(val_recall):
    plt.plot(Epochs[:len(val_recall)], val_recall, 'g', label='Val Recall')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.suptitle('Training Metrics Over Epochs', fontsize=16)
plt.tight_layout()
plt.show()

preds = model.predict(ts_gen, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = ts_gen.classes
labels = classes  # class names in index order

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test)')
plt.tight_layout()
plt.show()

# Text report
print(classification_report(y_true, y_pred, target_names=labels))

# Per-class accuracy bar chart
per_class_acc = (cm.diagonal() / cm.sum(axis=1).clip(min=1))
plt.figure(figsize=(8,4))
sns.barplot(x=labels, y=per_class_acc)
plt.ylim(0,1)
plt.title('Per-class Accuracy (Test)')
plt.ylabel('Accuracy')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()




