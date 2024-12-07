from tools.data_class import Data_class
data = Data_class()


import numpy as np
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
def positional_encoding2(length, depth):
  depth = depth/2
  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)
  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)
  return tf.cast(pos_encoding, dtype=tf.float32)

@keras.saving.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

@keras.saving.register_keras_serializable()
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

@keras.saving.register_keras_serializable()
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

@keras.saving.register_keras_serializable()
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

@keras.saving.register_keras_serializable()
class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

@keras.saving.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x

@keras.saving.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

@keras.saving.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.

@keras.saving.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

@keras.saving.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x


@keras.saving.register_keras_serializable()
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.num_layers = num_layers
    self.d_model = d_model  
    self.num_heads = num_heads    
    self.dff = dff
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size
    self.dropout_rate = dropout_rate

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def get_config(self):
    config = super().get_config()
    # config.pop('trainable')
    # config.pop('dtype')
    config.update({
      'num_layers':  self.num_layers,
      'd_model'  :  self.d_model,
      'num_heads'    :  self.num_heads,
      'dff':  self.dff,
      'input_vocab_size':  self.input_vocab_size,
      'target_vocab_size':  self.target_vocab_size,
      'dropout_rate':  self.dropout_rate
    })
    return config

  # @classmethod
  # def from_config(cls, config):
  #     # Extract necessary parameters
  #     build_config = config.get('build_config', {})
  #     compile_config = config.get('compile_config', {})
  #     optimizer_config = compile_config.get('optimizer', {})
  #     loss_config = config.get('loss', {})
  #     metrics_config = config.get('metrics', [])
  #     weighted_metrics_config = config.get('weighted_metrics', [])

  #     # Create a new instance of the model
  #     model = cls(**config)

  #     # Set optimizer, loss, and metrics
  #     optimizer_cls = tf.keras.utils.deserialize_keras_object(optimizer_config)
  #     optimizer = optimizer_cls(**optimizer_config.get('config', {}))
  #     model.compile(optimizer=optimizer, loss=loss_config, metrics=metrics_config, weighted_metrics=weighted_metrics_config)

  #     return model

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits

num_layers = 2
d_model = 128
dff = 256
num_heads = 4

input_vocab_size = data.eng_vocab_size
target_vocab_size = data.fre_vocab_size
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=data.eng_vocab_size,
    target_vocab_size=data.fre_vocab_size,
    dropout_rate=dropout_rate)

output = transformer((data.eng_encoded[:2], data.fre_encoded[:2, :-1]))

attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)


transformer.summary()

@keras.saving.register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

  def get_config(self):
    return {'d_model': self.d_model.numpy(), 'warmup_steps': self.warmup_steps}
  
  @classmethod
  def from_config(cls, config):
    d_model = config.get('d_model')
    warmup_steps = config.get('warmup_steps')
    print(d_model)
    return cls(d_model=d_model, warmup_steps=warmup_steps)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

@keras.saving.register_keras_serializable()
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

@keras.saving.register_keras_serializable()
def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

# zeroes = np.zeros((data.eng_encoded.shape[0], 7))
# eng = np.concatenate((data.eng_encoded, zeroes), axis=1)
fre = data.fre_encoded[:, :-1]
fre_labels = data.fre_encoded[:, 1:]

transformer.fit((data.eng_encoded, fre),fre_labels, epochs=2, batch_size=(256))

fre_id2word = {idx:word for word, idx in data.fre_tokenizer.word_index.items()}

class Translator():
  def __init__(self, transformer):
    self.transformer = transformer

  def __call__(self, sentence, max_length=23):

    sentence = data.eng_tokenizer.texts_to_sequences([sentence])
    sentence = data.pad(sentence, 15)
    encoder_input = sentence.reshape(1,-1)

    start_end = data.fre_tokenizer.texts_to_sequences(['sos eos'])
    start = start_end[0][0]
    end = start_end[0][1]
    res = np.array([[start]])
    translation = []

    for _ in range(max_length):
      predictions = self.transformer([encoder_input, res], training=False)
      predictions = predictions[0, -1, :]
      predicted_id = np.argmax(predictions)
      predicted_word = fre_id2word[predicted_id]
      translation.append(predicted_word)
      res = np.append(res,predicted_id).reshape(1,-1)

      if predicted_id == end:
        break

    return " ".join(translation).replace('eos', '')

# translator = Translator(transformer)
# print(data.eng_sentences[0])
# print(data.fre_sentences[0])
# translator(data.eng_sentences[0])
  
transformer.save('transformer.keras')
loaded = keras.saving.load_model('transformer.keras')