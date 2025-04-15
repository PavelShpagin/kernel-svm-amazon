import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    print("tqdm not installed. Install with 'pip install tqdm' for progress bars.")
    use_tqdm = False
    class FakeTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
    tqdm = FakeTqdm

def read_csv_file(file_path):
    print(f"Reading CSV file: {file_path}")
    try:
        use_cols = ['review_text', 'class_index']
        df = pd.read_csv(file_path, usecols=use_cols)

        df.rename(columns={'review_text': 'reviewText', 'class_index': 'label'}, inplace=True)

        if 'reviewText' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"CSV file {file_path} must contain 'review_text' and 'class_index' columns.")

        df['label'] = df['label'].apply(lambda x: -1 if x == 1 else 1)

        return df[['reviewText', 'label']]

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error reading or processing CSV file {file_path}: {e}")
        raise

def save_in_chunks(sparse_matrix, columns, labels, output_file, chunk_size=10000):
    total_rows = sparse_matrix.shape[0]
    
    for i in tqdm(range(0, total_rows, chunk_size), desc=f"Saving to {output_file}"):
        end = min(i + chunk_size, total_rows)
        chunk = sparse_matrix[i:end].toarray()
        
        chunk_df = pd.DataFrame(chunk, columns=columns)
        chunk_df['label'] = labels[i:end]
        
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        chunk_df.to_csv(output_file, index=False, mode=mode, header=header)

def preprocess_amazon_reviews(train_file, test_file, output_train_file, output_test_file):
    print("Loading datasets...")
    train_df = read_csv_file(train_file)
    test_df = read_csv_file(test_file)
    
    X_train = train_df['reviewText']
    y_train = train_df['label']
    
    X_test = test_df['reviewText']
    y_test = test_df['label']
    
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=1000, dtype=np.float32)
    
    print("Transforming training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print("Transforming test data...")
    X_test_tfidf = vectorizer.transform(X_test)
    
    feature_names = vectorizer.get_feature_names_out()
    
    print("Saving training data...")
    save_in_chunks(X_train_tfidf, feature_names, y_train.values, output_train_file)
    
    print("Saving test data...")
    save_in_chunks(X_test_tfidf, feature_names, y_test.values, output_test_file)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_amazon_reviews('dataset/train.csv', 'dataset/test.csv', 'train_data.csv', 'test_data.csv') 