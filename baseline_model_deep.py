import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import TFBertModel
from tensorflow.keras.layers import Input, Concatenate, Dense, BatchNormalization, GlobalAveragePooling2D, Flatten, Conv1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, BertTokenizer
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.regularizers import l2
from sklearn.decomposition import PCA
import seaborn as sns
from PIL import Image
from scipy.stats import spearmanr

np.random.seed(42)
tf.random.set_seed(42)

#Loading extracted visual features
visual_features_earth = np.load("visual_features_earth.npy")
visual_features_pics = np.load("visual_features_pics.npy")

#Loading the datasets
df_pics = pd.read_csv("pics_final_model.csv")
df_earth = pd.read_csv("earth_final_model.csv")

#Target values
score_pics = df_pics['log_score'].values
score_earth = df_earth['log_score'].values

#copy of target values 
score_pics_copy = df_pics['log_score'].values
score_earth_copy = df_earth['log_score'].values

#captions
title_earth = df_earth['title'].values
title_pics = df_pics['title'].values

#pca
pca_visual = PCA(n_components=20)

#reducing dimensionality using pca
visual_features_pca = pca_visual.fit_transform(visual_features_earth)


#Initialising everything
images_array = visual_features_pca
additional_data_array= df_earth[['caption_length', 'author_frequency', 'flair', 'is_original_content','is_over_18','is_locked','cos_posted_at','sin_posted_at']]
social_features = additional_data_array
scores_array = score_earth

# Split into train and (temporary) test set
images_train, images_temp, additional_data_train, additional_data_temp, scores_train, scores_temp = train_test_split(
    visual_features_pca, additional_data_array, scores_array, test_size=0.2, random_state=42, shuffle = True)  # 60% for training

# Split test set into validation and final test set
images_val, images_test, additional_data_val, additional_data_test, scores_val, scores_test = train_test_split(
    images_temp, additional_data_temp, scores_temp, test_size=0.5, random_state=42, shuffle = True)  # 20% for validation, 20% for testing

images_train = np.reshape(images_train, (images_train.shape[0], images_train.shape[1], 1))
images_val = np.reshape(images_val, (images_val.shape[0], images_val.shape[1], 1))
images_test = np.reshape(images_test, (images_test.shape[0], images_test.shape[1], 1))

additional_data_train = np.reshape(additional_data_train, (additional_data_train.shape[0], additional_data_train.shape[1], 1))
additional_data_val = np.reshape(additional_data_val, (additional_data_val.shape[0], additional_data_val.shape[1], 1))
additional_data_test = np.reshape(additional_data_test, (additional_data_test.shape[0], additional_data_test.shape[1], 1))


#Scaling
scaler = MinMaxScaler()
scores_train = np.array(scores_train).reshape(-1, 1)
scores_val= np.array(scores_val).reshape(-1, 1)
scores_test= np.array(scores_test).reshape(-1, 1)
scores_train= scaler.fit_transform(scores_train)
scores_val = scaler.transform(scores_val)
scores_test = scaler.transform(scores_test)
scores_train = scores_train.flatten()
scores_val = scores_val.flatten()
scores_test = scores_test.flatten()

scaler1 = MinMaxScaler()
score_pics_copy = np.array(score_pics_copy).reshape(-1, 1)
score_earth_copy = np.array(score_earth_copy).reshape(-1, 1)
scores_pics_c = scaler1.fit_transform(score_pics_copy)
scores_earth_c = scaler1.fit_transform(score_earth_copy)

# Visual Network
visual_input = Input(shape=(20,1))
v = Conv1D(32, 3, activation='relu', padding='same')(visual_input)
v = Dropout(0.1)(v)
v = BatchNormalization()(v)
v = Conv1D(64, 3, activation='relu', padding='same')(v)
v = Dropout(0.1)(v)
v = BatchNormalization()(v)
v = Conv1D(128, 3, activation='relu', padding='same')(v)
v = Dropout(0.1)(v)
v = BatchNormalization()(v)
v = Flatten()(v)

# Social Network
social_input = Input(shape=(8,1))
s = Conv1D(32, 2, activation='relu', padding='same')(social_input)
s = Dropout(0.1)(s)
s = BatchNormalization()(s)
s = Conv1D(64, 2, activation='relu', padding='same')(s)
s = Dropout(0.1)(s)
s = BatchNormalization()(s)
s = Conv1D(128, 2, activation='relu', padding='same')(s)
s = Dropout(0.1)(s)
s = BatchNormalization()(s)
s = Flatten()(s)

# Merge Visual and Social Networks
merged = Concatenate()([v, s])

# Fusion Network
f = Dense(1024, activation='relu')(merged)
f = Dropout(0.1)(f)
f = Dense(500, activation='relu')(f)
f = Dropout(0.2)(f)

# Output
output = Dense(1)(f)  

model = Model(inputs=[visual_input, social_input], outputs=output)

#scheduler
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        lr = lr * 0.1
    return lr

initial_learning_rate = 0.001
batch_size = 20

# Creating the Adam optimizer
adam_optimizer = Adam(learning_rate=initial_learning_rate)

lr_callback = LearningRateScheduler(lr_scheduler)

# Create a callback for model checkpoint
mc_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# List of callbacks
callbacks_list = [lr_callback, mc_callback]

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
# Early Stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)



history = model.fit([images_train, additional_data_train], 
                    scores_train, 
                    validation_data=([images_val, additional_data_val], scores_val),
                    epochs=50, 
                    batch_size=20,
                    callbacks=callbacks_list)


# Predicting on test data
predictions = model.predict([images_test, additional_data_test])

#unscaling the predictions to analyse on original scale
predictions = predictions.reshape(-1, 1)
predictions_rescaled = scaler.inverse_transform(predictions)
predictions_unlogged = np.exp(predictions_rescaled.flatten())


scores_test = scores_test.reshape(-1, 1)
scores_test_rescaled = scaler.inverse_transform(scores_test)
scores_test_unlogged = np.exp(scores_test_rescaled.flatten())


scores_pics_c = scores_pics_c.reshape(-1, 1)
scores_earth_c = scores_earth_c.reshape(-1, 1)
scores_pics_c_rescaled = scaler.inverse_transform(scores_pics_c)
scores_earth_c_rescaled = scaler.inverse_transform(scores_earth_c)
scores_unlogged_pics = np.exp(scores_pics_c_rescaled.flatten())
scores_unlogged_earth = np.exp(scores_earth_c_rescaled.flatten())

# predictions_unlogged = predictions.astype(int).flatten()
# scores_test_unlogged = scores_test.flatten()

#calculating metrics
mse = mean_squared_error(scores_test_unlogged, predictions_unlogged)
mae = mean_absolute_error(scores_test_unlogged, predictions_unlogged)
spearmans_rho,_ = spearmanr(predictions_unlogged, scores_test_unlogged)

#printing metrics
print("MSE: ", mse)
print("MAE: ", mae)
print("Spearmans rho: ", spearmans_rho)


# Set a seaborn style for prettier plots
sns.set_style('whitegrid')

# Scatter Plot of Predicted vs Actual Scores
plt.figure(figsize=(8, 4))
plt.scatter(predictions_unlogged, scores_test_unlogged, alpha=0.7, edgecolor='k')
plt.plot([min(scores_test_unlogged), max(scores_test_unlogged)], [min(scores_test_unlogged), max(scores_test_unlogged)], color='black', linewidth=2, alpha=0.5)  
plt.ylabel('Test Scores')
plt.xlabel('Predicted Scores')
plt.title('Scatter Plot of Predicted vs Actual Scores')
plt.show()

# Calculate the prediction errors
errors = scores_test_unlogged - predictions_unlogged

# Calculate the average prediction error
avg_error = np.mean(errors)

# Histogram of Prediction Errors
plt.figure(figsize=(10, 6))
plt.hist(errors, bins='auto', color='skyblue', edgecolor='black')

# Add vertical line at average error
plt.axvline(avg_error, color='k', linestyle='dashed', linewidth=1)
plt.text(avg_error+0.5,20,f'Avg error: {avg_error:.2f}',rotation=90)

plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors')

if avg_error > 0:
    plt.text(0.6, 0.9, 'Model tends to underestimate', transform=plt.gca().transAxes, color='red')
else:
    plt.text(0.6, 0.9, 'Model tends to overestimate', transform=plt.gca().transAxes, color='blue')

plt.show()

# Print overall accuracy within tolerance
print(f'Overall average prediction error: {avg_error:.2f}')



actual_scores = scores_test_unlogged
pred_scores = predictions_unlogged


#images of most accurate and least accurate

errors = np.abs(actual_scores - pred_scores)
sorted_indices = np.argsort(errors)
most_accurate = sorted_indices[:3]
least_accurate = sorted_indices[-3:]



original_scores = scores_unlogged_earth
image_dir = "C:\\Users\\seric\\Desktop\\thesis\\earth\\"  

# Create a numpy array for more efficient operation
original_scores_np = np.array(original_scores)

import textwrap

def show_images_for_predictions(indices, title):
    for i in indices:
        # Find the corresponding original index
        orig_index = np.where(original_scores_np == actual_scores[i])[0][0]
        caption = f"Caption: {title[orig_index]}"
        # Wrap text to 60 characters per line for legibility
        caption_wrapped = textwrap.fill(caption, 60)

        img_path = f"{image_dir}/{orig_index}.jpg" 
        img = Image.open(img_path)

        plt.imshow(img)
        # Use wrapped caption and include line breaks in title
        plt.title(f"{caption_wrapped}\nActual score: {int(actual_scores[i])}, "
                  f"Predicted score: {int(pred_scores[i])},\n Error: {int(errors[i])}")
        plt.show()
        print("\n")

# Show images for the most and least accurate predictions
# show_images_for_predictions(most_accurate, title_pics)
show_images_for_predictions(least_accurate, title_earth)



scores = scores_test_unlogged # actual scores
n = len(scores)  # since both arrays have the same length

def get_accuracy_and_count_for_range(scores, pred_scores, min_diff, max_diff):
    pairs = [(i, j) for i in range(len(scores)) for j in range(i+1, len(scores)) 
             if min_diff < abs(scores[i] - scores[j]) < max_diff]
    
    correct_predictions = 0
    for i, j in pairs:
        if (pred_scores[i] > pred_scores[j]) == (scores[i] > scores[j]):
            correct_predictions += 1
    
    if len(pairs) == 0:
        return None, 0

    return correct_predictions / len(pairs), len(pairs)

# Define score difference bins
bins = np.linspace(0, scores.max() - scores.min(), num=20)

# Calculate accuracies and counts for each bin
accuracies_and_counts = [get_accuracy_and_count_for_range(scores, pred_scores, bins[i], bins[i+1]) for i in range(len(bins)-1)]
accuracies, counts = zip(*accuracies_and_counts)

# Filter out bins without pairs
bins, accuracies, counts = zip(*[(bin_center, acc, count) for bin_center, acc, count in zip(bins[:-1], accuracies, counts) if acc is not None])

# Define color gradient
colors = plt.cm.viridis(np.linspace(0, 1, len(bins)))

# Calculate accuracy ratio (comparing each bin to the first one)
accuracy_ratios = [acc/accuracies[0] for acc in accuracies]

# Plot histogram with colored bars
plt.figure(figsize=(10, 6))
barContainer = plt.bar(bins, accuracies, color=colors, width=np.diff(bins)[0])
plt.xlabel('Difference in Actual Scores')
plt.ylabel('Accuracy')
plt.title('Accuracy per Score Difference Bin')
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Score Difference')

# Adding ratio and count as annotation inside the bars, rotated vertically
for idx, rect in enumerate(barContainer):
    height = rect.get_height()
    text_color = 'white' if idx < len(bins)/2 else 'black' # change text color based on the color of the bars
    plt.text(rect.get_x() + rect.get_width() / 2., height/2, 
             f'ratio: {accuracy_ratios[idx]:.2f} count: {counts[idx]}',
             ha='center', va='center', color=text_color, rotation='vertical')

plt.show()







plt.show()