import pandas as pd

# Use this code to create a .csv file with the necessary format needed for 
# categorical emotion recognition model

# Load Original label_consensus.csv file provided with dataset
df = pd.read_csv('path/to/original/label_consensus.csv')

# Define the emotions
emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]

# Create a dictionary for one-hot encoding
one_hot_dict = {e: [1.0 if e == ec else 0.0 for ec in emotion_codes] for e in emotion_codes}

# Filter out rows with undefined EmoClass
df = df[df['EmoClass'].isin(emotion_codes)]

# Apply one-hot encoding
for i, e in enumerate(emotion_codes):
    df[emotions[i]] = df['EmoClass'].apply(lambda x: one_hot_dict[x][i])

# Select relevant columns for the new CSV
df_final = df[['FileName', *emotions, 'Split_Set']]

# Save the processed data to a new CSV file
df_final.to_csv('processed_labels.csv', index=False)

print("Processing complete. New file saved as 'processed_labels.csv'")
