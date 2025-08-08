import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
import random
import json
import pickle
import os

# Create a stemmer for word normalization
stemmer = LancasterStemmer()

# Download NLTK's tokenizer model (only needed once)
#nltk.download('punkt')

# ---------------------------
# LOAD INTENTS FROM JSON FILE
# ---------------------------
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# ---------------------------
# LOAD OR CREATE TRAINING DATA
# ---------------------------
DATA_PICKLE = "data.pickle"
try:
    # Try loading preprocessed data to save time
    with open(DATA_PICKLE, "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    # Variables to store vocabulary, labels, and training samples
    words = []
    labels = []
    docs_x = []  # tokenized patterns
    docs_y = []  # corresponding tags

    # Loop through each intent
    for intent in data["intents"]:
        for pattern in intent.get("patterns", []):
            # Tokenize the pattern into words
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        # Add label if not already in the list
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Stem words (reduce to base form) and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # Create bag-of-words and one-hot label arrays
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]  # base vector for labels

    for x, doc in enumerate(docs_x):
        bag = []

        # Stem each word in the pattern
        wrds = [stemmer.stem(w.lower()) for w in doc]

        # Create bag-of-words: 1 if word exists in pattern, else 0
        for w in words:
            bag.append(1 if w in wrds else 0)

        # One-hot encode the label
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Convert to NumPy arrays for TensorFlow
    training = np.array(training, dtype=np.float32)
    output = np.array(output, dtype=np.float32)

    # Save processed data for future runs
    with open(DATA_PICKLE, "wb") as f:
        pickle.dump((words, labels, training, output), f)

# ---------------------------
# DEFINE MODEL FUNCTION
# ---------------------------
def build_model(input_size, output_size):
    """
    Builds a simple feedforward neural network using Keras.
    Equivalent to the original tflearn structure.
    """
    model = Sequential([
        Input(shape=(input_size,)),              # Input layer
        Dense(8, activation="relu"),             # First hidden layer
        Dense(8, activation="relu"),             # Second hidden layer
        Dense(output_size, activation="softmax") # Output layer
    ])
    # Compile model with Adam optimizer & categorical crossentropy loss
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------
# LOAD OR TRAIN MODEL
# ---------------------------
MODEL_FILE = "model.keras"
should_retrain = False
model = None

if os.path.exists(MODEL_FILE):
    # Ask the user whether to retrain
    choice = input("Model already exists. Do you want to retrain the model? (yes/no): ").strip().lower()
    if choice == "no":
        try:
            model = load_model(MODEL_FILE)
            print("✅ Model loaded successfully.")
            should_retrain = False
        except Exception as e:
            print("⚠️ Error loading model:", e)
            print("ℹ️ Proceeding to retrain the model due to loading error.")
            should_retrain = True
    else:
        print("ℹ️ Retraining the model as per user choice...")
        should_retrain = True
else:
    print("ℹ️ Model file not found. Training a new model...")
    should_retrain = True

if should_retrain:
    print("--- Starting Model Training ---")
    model = build_model(len(training[0]), len(output[0]))
    # Train the model (500 epochs = same as n_epoch=500 in tflearn)
    model.fit(training, output, epochs=500, batch_size=8, verbose=1)
    model.save(MODEL_FILE)
    print("Model trained and saved to", MODEL_FILE)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def bag_of_words(s, words):
    """
    Convert a user sentence into a bag-of-words array based on known vocabulary.
    """
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)

# ---------------------------
# CHAT LOOP
# ---------------------------
def chat():
    print("Start chatting with the Baun: a paramedic assistant (type 'quit' to exit):")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Convert input to model-readable format
        bow = bag_of_words(inp, words)

        # Get model predictions
        results = model.predict(np.array([bow]), verbose=0)[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        # Check confidence threshold
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg.get('responses', [])
                    break
            # Debugging info
            #print(f"Confidence scores: {results}")
            #print(f"Predicted intent: {tag} with confidence {results[results_index]:.2f}")
            if responses:
                print("Baun:", random.choice(responses))
            else:
                print("Baun: (no response found for this intent)")
        else:
            print("Baun: My apologies, could you maybe rephrase that?")

# ---------------------------
# RUN CHATBOT
# ---------------------------
if __name__ == "__main__":
    chat()
