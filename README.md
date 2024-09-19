import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import string
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import emoji
nltk.download('punkt')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.layers import LSTM, Dense, SimpleRNN, Embedding, Flatten, Dropout
from keras.activations import softmax
from sklearn.model_selection import train_test_split
# ignore warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/content/sentiment_tweets3.csv")

df.rename(columns={'message to examine': 'Text', 'label (depression result)': 'Label'}, inplace=True)

df['Text'] = df['Text'].str.lower()

def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

# Remove HTML tags from 'Text' column
df['Text'] = df['Text'].apply(remove_html_tags)

# Define a function to remove URLs using regular expressions
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

# Apply the function to the 'Text' column
df['Text'] = df['Text'].apply(remove_urls)

string.punctuation

# Define the punctuation characters to remove
punctuation = string.punctuation

# Function to remove punctuation from text
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))

# Apply remove_punctuation function to 'Text' column
df['Text'] = df['Text'].apply(remove_punctuation)

# Define a dictionary of chat word mappings
chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}
# Function to replace chat words with their full forms
def replace_chat_words(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in chat_words:
            words[i] = chat_words[word.lower()]
    return ' '.join(words)

# Apply replace_chat_words function to 'Text' column
df['Text'] = df['Text'].apply(replace_chat_words)

# Download NLTK stopwords corpus
nltk.download('stopwords')

# Get English stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Function to remove stop words from text
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Apply remove_stopwords function to 'Text' column
df['Text'] = df['Text'].apply(remove_stopwords)

# Function to remove emojis from text
def remove_emojis(text):
    return emoji.demojize(text)

# Apply remove_emojis function to 'Text' column
df['Text'] = df['Text'].apply(remove_emojis)

# Intilize Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Apply
df['Text_lemmatized'] = df['Text'].apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word , pos='v') for word in x.split()]))

# Head
df.head()

Index           	Text	                                      Label	   Text_lemmatized
0	  106	   real good moment missssssssss much               	0	      real good moment miss much
1	  217	   reading manga	                                    0      	read manga
2	  220	   comeagainjen                                      	0      	comeagainjen
3	  288	   lapcat need send em accountant tomorrow oddly ...	0	      lapcat need send em accountant tomorrow oddly ...
4	  540	    add myspace myspacecomlookthunder	                0      	add myspace myspacecomlookthunder

X = df['Text']
y = df['Label']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(oov_token = 'nothing')
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)
tokenizer.document_count
10314
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
# Max Len in X_train_sequences
maxlen = max(len(tokens) for tokens in X_train_sequences)
print("Maximum sequence length (maxlen):", maxlen)

# Perform padding on X_train and X_test sequences
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, padding='post')
# Print the padded sequences for X_train and X_test
print("X_train_padded:")
print(X_train_padded)
print("\nX_test_padded:")
print(X_test_padded)
Maximum sequence length (maxlen): 75
X_train_padded:
[[   22   652   908 ...     0     0     0]
 [   15  4059  1215 ...     0     0     0]
 [ 6280  6281     3 ...     0     0     0]
 ...
 [18977   936     0 ...     0     0     0]
 [  386  1834  5474 ...     0     0     0]
 [  409    14  1098 ...     0     0     0]]

X_test_padded:
[[18978   341   125 ...     0     0     0]
 [18979   696   182 ...     0     0     0]
 [18980    82    96 ...     0     0     0]
 ...
 [22120 22121 22122 ...     0     0     0]
 [22123   619   140 ...     0     0     0]
 [22124   626   812 ...     0     0     0]]

# Embedding Input Size / Vocabulary Size
input_Size = np.max(X_train_padded) + 1
input_Size
18978

# Define the model
model = Sequential()

# Use LSTM instead of SimpleRNN for better capturing long-term dependencies
model.add(LSTM(128, input_shape=(75,1), return_sequences=True))

# Add dropout regularization
model.add(Dropout(0.5))

# Add another LSTM layer
model.add(LSTM(128))

# Add dropout regularization
model.add(Dropout(0.5))

# Add a dense layer with ReLU activation
model.add(Dense(64, activation='relu'))

# Output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm_2 (LSTM)                        │ (None, 75, 128)             │          66,560 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 75, 128)             │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_3 (LSTM)                        │ (None, 128)                 │         131,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 1)                   │              65 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 206,465 (806.50 KB)
 Trainable params: 206,465 (806.50 KB)
 Non-trainable params: 0 (0.00 B)

# Model Train
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_data=(X_test_padded, y_test))

Epoch 1/5
258/258 ━━━━━━━━━━━━━━━━━━━━ 66s 235ms/step - accuracy: 0.8076 - loss: 0.5056 - val_accuracy: 0.8357 - val_loss: 0.4371
Epoch 2/5
258/258 ━━━━━━━━━━━━━━━━━━━━ 81s 231ms/step - accuracy: 0.8226 - loss: 0.4518 - val_accuracy: 0.8352 - val_loss: 0.4345
Epoch 3/5
258/258 ━━━━━━━━━━━━━━━━━━━━ 81s 226ms/step - accuracy: 0.8488 - loss: 0.3925 - val_accuracy: 0.9394 - val_loss: 0.1614
Epoch 4/5
258/258 ━━━━━━━━━━━━━━━━━━━━ 85s 240ms/step - accuracy: 0.9157 - loss: 0.2101 - val_accuracy: 0.9675 - val_loss: 0.0951
Epoch 5/5
258/258 ━━━━━━━━━━━━━━━━━━━━ 80s 231ms/step - accuracy: 0.9535 - loss: 0.1401 - val_accuracy: 0.9782 - val_loss: 0.0811

# Step 2: Save the model
model.save('sentiment_model.h5')

# Step 3: Save the history as a CSV
import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Step 4: Download the model and history files
from google.colab import files

# Download model
files.download('sentiment_model.h5')

# Download training history
files.download('training_history.csv')

from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model('sentiment_model.h5')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = BeautifulSoup(text, 'html.parser').get_text()  # Remove HTML
    text = re.sub(r'http\S+|www\S+', '', text)            # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)                   # Remove punctuation
    return text

# Example input
input_text = "This movie was nice"

# Preprocess the input
preprocessed_text = preprocess_text(input_text)

# Tokenize and pad the input (you must load or recreate the tokenizer you used during training)
# tokenizer = ... (load your tokenizer)
# sequence = tokenizer.texts_to_sequences([preprocessed_text])
# padded_sequence = pad_sequences(sequence, maxlen=100)  # Adjust maxlen as per training

# Simulated prediction for this example (Replace with actual model input)
# prediction = model.predict(padded_sequence)
prediction = np.array([[0.85]])  # Simulating a positive sentiment prediction

# Output the prediction
if prediction > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_padded, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

65/65 ━━━━━━━━━━━━━━━━━━━━ 6s 83ms/step - accuracy: 0.9694 - loss: 0.0951
Test Accuracy: 97.82%

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plotting the training and testing loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Step 1: Install Gradio in Colab
!pip install gradio

import gradio as gr
import re
from bs4 import BeautifulSoup
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

nltk.download('stopwords')

# Assume the model is saved as a .h5 file and load it
# model = load_model('path_to_your_model.h5')

# Preprocessing functions (same as in your code)
def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Preprocess the input text
def preprocess_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    return text

# Step 2: Define the prediction function
def predict_sentiment(text):
    processed_text = preprocess_text(text)

    # Tokenization and padding logic (adjust as per your model)
    # tokenizer = Tokenizer()  # Load the tokenizer used during training
    # sequence = tokenizer.texts_to_sequences([processed_text])
    # padded_sequence = pad_sequences(sequence, maxlen=100)  # Adjust maxlen based on your model

    # Make the prediction using the model
    # prediction = model.predict(padded_sequence)

    # Simulate a prediction for demonstration
    prediction = "Positive" if len(text) % 2 == 0 else "Negative"  # Replace with actual prediction

    return prediction

# Step 3: Create the Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),  # Updated syntax
    outputs="text",
    title="Sentiment Analysis",
    description="Enter text to predict its sentiment"
)

# Step 4: Launch the interface
interface.launch()

sample output:
