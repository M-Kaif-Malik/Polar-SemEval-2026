import os
import pandas as pd

# Define paths
data_dir = "data"
dev_dir = os.path.join(data_dir, "dev")
train_dir = os.path.join(data_dir, "train")

# Output folders
prep_dev_dir = "prep1_dev"
prep_train_dir = "prep2_train"
os.makedirs(prep_dev_dir, exist_ok=True)
os.makedirs(prep_train_dir, exist_ok=True)

# Preprocessing function
def preprocess_df(df):
    # Lowercase
    df['text'] = df['text'].str.lower()
    # Strip extra whitespace
    df['text'] = df['text'].str.strip().replace('\s+', ' ', regex=True)
    # Remove empty or very short texts (less than 3 characters)
    df = df[df['text'].str.len() > 2]
    return df

# Function to load and combine files
def load_and_combine(folder_path):
    combined = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            lang = file.split('.')[0]  # filename without extension as language
            df = pd.read_csv(os.path.join(folder_path, file))
            df['language'] = lang
            df = preprocess_df(df)
            combined.append(df)
    return pd.concat(combined, ignore_index=True)

# Process dev
dev_df = load_and_combine(dev_dir)
dev_df.to_csv(os.path.join(prep_dev_dir, "dev_combined.csv"), index=False)
print(f"Dev data processed and saved to {prep_dev_dir}/dev_combined.csv")

# Process train
train_df = load_and_combine(train_dir)
train_df.to_csv(os.path.join(prep_train_dir, "train_combined.csv"), index=False)
print(f"Train data processed and saved to {prep_train_dir}/train_combined.csv")