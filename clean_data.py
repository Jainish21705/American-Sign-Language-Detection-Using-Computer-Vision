import pickle

# Load the original data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

clean_data = []
clean_labels = []

for sample, label in zip(data, labels):
    if len(sample) >= 42:
        clean_data.append(sample[:42])  # Keep only first 42 features
        clean_labels.append(label)

print(f"Original samples: {len(data)}")
print(f"Cleaned samples (42 features): {len(clean_data)}")

# Save cleaned data
with open('data_42.pickle', 'wb') as f:
    pickle.dump({'data': clean_data, 'labels': clean_labels}, f)

print("Saved cleaned data to 'data_42.pickle'")
