import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
with open('data_42.pickle', 'rb') as f:
    data_dict = pickle.load(f)

raw_data = data_dict['data']
raw_labels = data_dict['labels']

# Filter out samples with incorrect length
filtered_data = []
filtered_labels = []

for sample, label in zip(raw_data, raw_labels):
    if len(sample) == 42:  # 21 landmarks × 2 (x, y)
        filtered_data.append(sample)
        filtered_labels.append(label)

if not filtered_data:
    raise ValueError("No valid samples found with exactly 42 elements.")

# Convert to NumPy arrays
data = np.asarray(filtered_data, dtype=np.float32)
labels = np.asarray(filtered_labels)

# Split and train
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy * 100:.2f}% of samples were classified correctly!")

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'model.p'")
