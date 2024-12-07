# type: ignore
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)


class Data_class:

  def __init__(self):
    
    self.eng_sentences, self.fre_sentences = self.get_data()

    self.eng_tokenized, self.eng_tokenizer = self.tokenize(self.eng_sentences)
    self.fre_tokenized, self.fre_tokenizer = self.tokenize(self.fre_sentences,
                                            encode_start_end = True)
    self.eng_encoded = self.pad(self.eng_tokenized)
    self.fre_encoded = self.pad(self.fre_tokenized)

    self.eng_vocab_size = len(self.eng_tokenizer.word_index)
    self.fre_vocab_size = len(self.fre_tokenizer.word_index)

    self.eng_seq_len = len(self.eng_encoded[0])
    self.fre_seq_len = len(self.fre_encoded[0])
    
  def get_data(self):
    with open(dir_path+'/data/small_vocab_en', 'r') as f:
        eng_sentences = f.read().split('\n')
        
    with open(dir_path+'/data/small_vocab_fr', 'r') as f:
        fre_sentences = f.read().split('\n')
    
    return eng_sentences,fre_sentences

  def tokenize(self,sentences, encode_start_end = False):
      
      if encode_start_end:
          sentences = ["SOS " + s + "EOS" for s in sentences]
      
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(sentences)
      tokenizer.word_index["pad"] = 0
      tokenized_sentences = tokenizer.texts_to_sequences(sentences)
      
      return tokenized_sentences, tokenizer

  def pad(self,sentences, length = None):

      if length is None:
          length = max([len(s) for s in sentences])
          
      padded_sentences = pad_sequences(sentences, 
                                      maxlen = length,
                                      padding = 'post',
                                      truncating = 'post')

      return padded_sentences

