import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers

# ----------------------------
# Load Dataset
# ----------------------------
fake = pd.read_csv("Fake.csv", encoding="latin1")
true = pd.read_csv("True.csv", encoding="latin1")
# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

# Select columns
data = data[["text", "label"]]

# Remove empty rows
data = data.dropna()

# Convert to numpy
X = data["text"].astype(str).to_numpy()
y = data["label"].astype("int32").to_numpy()

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Text Vectorization
# ----------------------------
vectorizer = TextVectorization(
    max_tokens=50000,
    output_sequence_length=200
)

vectorizer.adapt(X_train)

# ----------------------------
# Build Deep Neural Network
# ----------------------------
model = tf.keras.Sequential([
    vectorizer,
    layers.Embedding(50000, 64),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# Train Model (Every Run)
# ----------------------------
print("Training model...\n")

model.fit(
    X_train,
    y_train,
    epochs=15,
    validation_data=(X_test, y_test)
)

# ----------------------------
# Evaluate Model
# ----------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nModel Accuracy:", round(accuracy*100,2), "%")

print("\nFake News Detection System Ready!")

# ----------------------------
# User Input Loop
# ----------------------------
while True:

    news = input("\nEnter news text to analyze (type 'exit' to quit): ").strip()

    if news.lower() == "exit":
        print("Program Closed.")
        break

    if len(news) < 10:
        print("⚠ Please enter a longer news sentence.")
        continue

    input_data = tf.data.Dataset.from_tensor_slices([news]).batch(1)

    prediction = model.predict(input_data)[0][0]

    confidence = prediction * 100

    if prediction > 0.5:
        print("\nResult: Real News ✅")
        print("Confidence:", round(confidence,2), "%")
    else:
        print("\nResult: Fake News ❌")
        print("Confidence:", round(100-confidence,2), "%")
        prediction = model.predict(input_data)[0][0]

real_prob = prediction
fake_prob = 1 - prediction

labels = ["Real", "Fake"]
values = [real_prob, fake_prob]

plt.bar(labels, values)
plt.title("Prediction Confidence")
plt.ylabel("Probability")
plt.show()