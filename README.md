# LSTM Sentiment Analysis
Authored by saeed asle

# Description
This project uses LSTM (Long Short-Term Memory) neural networks to perform sentiment analysis on movie reviews.
The dataset contains phrases from movie reviews labeled on a scale of 0 to 4:

    0: Negative
    1: Somewhat negative
    2: Neutral
    3: Somewhat positive
    4: Positive
The project preprocesses the text data, tokenizes it, and pads sequences to a fixed length.
It then builds an LSTM model to classify the sentiment of the reviews.

#Features
  * Data preprocessing: Cleans text data, tokenizes, and pads sequences.
  * LSTM model building: Constructs an LSTM model using Keras with an Embedding layer, LSTM layer, and Dense layers.
  * Training and evaluation: Trains the LSTM model on the preprocessed data and evaluates its performance on the validation set.
    
# How to Use
  * Ensure you have the necessary libraries installed, such as numpy, pandas, matplotlib, keras, and scikit-learn.
  * Download the dataset containing movie reviews and their corresponding sentiments. You can find the dataset here.
  * Run the provided code to preprocess the data, build the LSTM model, train the model, and visualize the training and validation accuracy.
  * 

# Dependencies
  * numpy: For numerical operations.
  * pandas: For data manipulation and analysis.
  * matplotlib: For plotting graphs.
  * keras: For building and training neural networks.
  * scikit-learn: For machine learning utilities.
    
# Output

The code outputs a plot showing the training and validation accuracy of the LSTM model over epochs and the type.
