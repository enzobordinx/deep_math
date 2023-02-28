import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Carregando o arquivo de texto com as funções matemáticas
with open('math.txt', 'r') as f:
    text = f.read()

# Removendo caracteres especiais e convertendo tudo para letras minúsculas
text = re.sub('[^a-zA-Z\n]', ' ', text)
text = text.lower()

# Dividindo o texto em linhas
lines = text.split('\n')

# Criando um tokenizador e gerando os índices das palavras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# Preenchendo as sequências para que todas tenham o mesmo comprimento
max_length = max([len(s) for s in sequences])
sequences = pad_sequences(sequences, maxlen=max_length)

# Criando o modelo da rede neural
model = Sequential()

# Adicionando a camada de embedding
model.add(Embedding(len(tokenizer.word_index)+1, 100, input_length=max_length))

# Adicionando camadas LSTM e Dropout
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))

# Adicionando a camada densa de saída
model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando a rede
model.fit(sequences, np.zeros(len(sequences)), epochs=100)

# Salvando o modelo treinado
model.save('math_model.h5')

# Função para responder perguntas de matemática
def math_question_answer(question):
    # Convertendo a pergunta para uma sequência
    question = tokenizer.texts_to_sequences([question])
    question = pad_sequences(question, maxlen=max_length)
    # Fazendo a previsão
    prediction = model.predict(question)[0]
    # Retornando a resposta
    if prediction > 0.5:
        return "Essa pergunta é sobre matemática"
    else:
        return "Essa pergunta não é sobre matematica"
