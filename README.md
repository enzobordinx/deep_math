# deep_math
Estudo de deep learning com python e treino em matematica

 _______  _______  _______  _______  _______  _______  _______  _______ 
|       ||       ||   _   ||       ||       ||       ||       ||       |
|  _____||    ___||  |_|  ||_     _||  _____||_     _||    ___||_     _|
| |_____ |   |___ |       |  |   |  | |_____   |   |  |   |___   |   |  
|_____  ||    ___||       |  |   |  |_____  |  |   |  |    ___|  |   |  
 _____| ||   |___ |   _   |  |   |   _____| |  |   |  |   |___   |   |  
|_______||_______||__| |__|  |___|  |_______|  |___|  |_______|  |___|  


Math Question Answer
This Python script uses deep learning techniques to classify whether a given input question is related to mathematics or not. The script first loads a text file containing a list of mathematical functions, and preprocesses the data to remove special characters and convert everything to lowercase.

The text is then split into individual lines and fed into a tokenizer, which generates an index for each unique word in the text. The tokenizer then converts the lines of text into sequences of these word indices.

To ensure that all sequences are of the same length, the script pads the sequences with zeros to the length of the longest sequence.

The model architecture consists of an embedding layer, followed by two LSTM layers with a dropout layer in between. Finally, a dense layer with a sigmoid activation function is added as the output layer.

The model is compiled with the Adam optimizer and binary cross-entropy loss function, and trained on the sequences of mathematical functions.

Once trained, the model is saved to a file for later use. The script also includes a function that takes a string input, converts it into a sequence, and passes it through the model to predict whether the input is related to mathematics or not.

Usage
To use the script, simply run the Python file in a Python environment with the necessary dependencies installed.

The script will train the model and save it to a file named math_model.h5. After that, you can use the math_question_answer function to classify whether a given input is related to mathematics or not.

python
Copy code
result = math_question_answer("What is the Pythagorean theorem?")
print(result)
This will output "Essa pergunta é sobre matemática" if the input is related to mathematics, or "Essa pergunta não é sobre matematica" if it is not related to mathematics.

Dependencies
The script requires the following dependencies:

numpy
keras
tensorflow
opencv-python-headless
regex
