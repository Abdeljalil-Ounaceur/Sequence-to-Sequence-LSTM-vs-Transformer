import tensorflow  as tf
import keras 
from keras import layers 
import numpy as np

input_data = "Hi I am a student and I like food"
target_data = "Bonjour je suis un etudiant et j' aime la nourriture <EOS>"

input_list = input_data.split(" ")
target_list = target_data.split(" ")

num_input_words = len(input_list)
num_target_words = len(target_list)

unique_input_words = set(input_list)
unique_target_words = set(target_list)


input_lookup = {word: i for i,word in enumerate(unique_input_words)}
target_lookup = {word:i for i, word in enumerate(unique_target_words)}

input_vocab_size = len(unique_input_words)
target_vocab_size = len(unique_target_words)

input_one_hot = np.zeros((num_input_words,input_vocab_size))
target_one_hot = np.zeros((num_target_words, target_vocab_size))

for i in range(num_input_words):
  input_one_hot[i,input_lookup[input_list[i]]] = 1


for i in range(num_target_words):
  target_one_hot[i,target_lookup[target_list[i]]] = 1

target_one_hot_y = target_one_hot[1:]
target_one_hot = target_one_hot[:-1]
encoder_input_data = input_one_hot.reshape(1,input_one_hot.shape[0],-1)
decoder_input_data = target_one_hot.reshape(1,target_one_hot.shape[0],-1)
desired_output = target_one_hot_y.reshape(1,target_one_hot_y.shape[0],-1)



encoder_inputs = keras.Input(shape=(None, input_vocab_size))
decoder_inputs = keras.Input(shape=(None, target_vocab_size))

# encoder_embeddings = layers.Embedding(input_dim=input_vocab_size, output_dim=8)(encoder_inputs)
# decoder_embeddings = layers.Embedding(input_dim=target_vocab_size, output_dim=8)(decoder_inputs)

_,h, c = layers.LSTM(8, return_state=True)(encoder_inputs)
decoder_lstm_output, _, _ = layers.LSTM(
  8, 
  return_sequences=True, 
  return_state=True
  )(decoder_inputs, initial_state=[h,c])

outputs = layers.Dense(target_vocab_size, activation="softmax")(decoder_lstm_output)

model = keras.Model([encoder_inputs, decoder_inputs], outputs)

model.compile(optimizer="rmsprop", 
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy()]
)


history = model.fit([encoder_input_data, decoder_input_data], desired_output, epochs=300)


def decode_sequence(input_seq):
    # Encode the input sequence to get the initial state vectors
    _,h,c = model.layers[2](input_seq)
    # Generate an empty target sequence with a start token
    target_seq = np.zeros((1, 1, target_vocab_size))
    target_seq[0, 0, target_lookup["Bonjour"]] = 1.0
    # Create the list to store the decoded sentence
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        lstm_output, h, c = model.layers[3](target_seq, initial_state=[h,c])
        output_tokens=model.layers[4](lstm_output)
        # Sample the most likely word index and convert it to the corresponding word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = {v: k for k, v in target_lookup.items()}[sampled_token_index]
        print(sampled_word)
        decoded_sentence += " " + sampled_word
        # Exit condition: either hit max length or find stop token.
        if (sampled_word == "<EOS>" or len(decoded_sentence.split()) > 20):
            stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, target_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.0
    return decoded_sentence


decode_sequence(encoder_input_data[:,::-1,:])