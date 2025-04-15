import pandas as pd

# Read the first 40000 rows of the training data
df_train = pd.read_csv('train_data.csv', nrows=50000)

# Save the truncated training data back to the file
df_train.to_csv('train_data.csv', index=False)
print(f"Saved {len(df_train)} rows to train_data.csv")

# Read the first 40000 rows of the test data
df_test = pd.read_csv('test_data.csv', nrows=20000)

# Save the truncated test data back to the file
df_test.to_csv('test_data.csv', index=False)
print(f"Saved {len(df_test)} rows to test_data.csv")
