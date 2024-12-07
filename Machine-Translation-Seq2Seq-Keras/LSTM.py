import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense # type:ignore
from keras.models import Model # type:ignore
from keras.optimizers import Adam # type:ignore
from keras.losses import sparse_categorical_crossentropy # type:ignore
from tools.data_class import Data_class


##########################1&2. *LOADING AND PREPARING DATA* ###################  
data = Data_class()

######################### 3. BUILD *TRAINING* MODEL ###########################

encoder_input = Input(shape = (None, ),
                      name = "Encoder_Input")

embedding_dim = 200
embedded_input = Embedding(input_dim = data.eng_vocab_size,
                           output_dim = embedding_dim,
                           name = "Embedding_Layer")(encoder_input)

encoder_lstm = LSTM(units = 256,
                    activation = "relu",
                    return_sequences = False,
                    return_state = True,
                    name = "Encoder_LSTM")

_, last_h_encoder, last_c_encoder = encoder_lstm(embedded_input)

decoder_input = Input(shape = (None, 1),
                      name = "Deocder_Input")

decoder_lstm = LSTM(units = 256,
                    activation = "relu",
                    return_sequences = True,
                    return_state = True,
                    name = "Decoder_LSTM")

all_h_decoder, _, _ = decoder_lstm(decoder_input,
                                   initial_state = [last_h_encoder, last_c_encoder])

final_dense = Dense(data.fre_vocab_size,
                    activation = 'softmax',
                    name = "Final_Dense_Layer")

logits = final_dense(all_h_decoder)

seq2seq_model = Model([encoder_input, decoder_input],logits)

seq2seq_model.compile(loss = sparse_categorical_crossentropy,
                      optimizer = Adam(0.002),
                      metrics = ['accuracy'])


############################ 4. TRAIN THE MODEL ###############################
# Decoder: input - all but last word, target - all but "starofsentence" token
decoder_fre_input = data.fre_encoded.reshape((-1, data.fre_seq_len, 1))[:, :-1, :]
decoder_fre_target = data.fre_encoded.reshape((-1, data.fre_seq_len, 1))[:, 1:, :]

seq2seq_model.fit([data.eng_encoded, decoder_fre_input],
                  decoder_fre_target,
                  epochs = 16,
                  batch_size = 1024)
        
        
######################### 5. BUILD *INFERENCE* MODEL ##########################
inf_encoder_model = Model(encoder_input, [last_h_encoder, last_c_encoder])

decoder_initial_states = [Input(shape = (256,)), 
                          Input(shape = (256,))]

all_h_decoder, last_h_decoder, last_c_decoder = decoder_lstm(decoder_input,
                                                             initial_state = decoder_initial_states)

logits = final_dense(all_h_decoder)

inf_decoder_model = Model([decoder_input,*decoder_initial_states],
                          [logits,last_h_decoder, last_c_decoder])


############################### 6. TRANSLATE!! ################################
# word id -> word dict for frenish:
fre_id2word = {idx:word for word, idx in data.fre_tokenizer.word_index.items()}

def translate(eng_sentence):
    eng_sentence = eng_sentence.reshape((1, data.eng_seq_len))  # give batch size of 1
    initial_states = inf_encoder_model.predict(eng_sentence)
    # Initialize decoder input as a length 1 sentence containing "SOS",
    # --> feeding the start token as the first predicted word
    prev_word = np.zeros((1,1,1))
    prev_word[0, 0, 0] = data.fre_tokenizer.word_index["sos"]
    stop_condition = False
    translation = []
    while not stop_condition:
        # 1. predict the next word using decoder model
        logits, last_h, last_c = inf_decoder_model.predict([prev_word,*initial_states])
        # 2. Update prev_word with the predicted word
        predicted_id = np.argmax(logits[0, 0, :])
        predicted_word = fre_id2word[predicted_id]
        translation.append(predicted_word)
        # 3. Enable End Condition: (1) if predicted word is "eos" OR
        #                          (2) if translated sentence reached maximum sentence length
        if (predicted_word == 'eos' or len(translation) > data.fre_seq_len):
            stop_condition = True
        # 4. Update prev_word with the predicted word
        prev_word[0, 0, 0] = predicted_id
        # 5. Update initial_states with the previously predicted word's encoder output
        initial_states = [last_h, last_c]
    return " ".join(translation).replace('eos', '')