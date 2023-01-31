import tensorflow as tf
from translate.storage.tmx import tmxfile

# Load the parallel corpus data
source_sentences = []
target_sentences = []

with open("en-fr.tmx", 'rb') as fin:
    tmx_file = tmxfile(fin, 'en', 'fr')
for node in tmx_file.unit_iter():
    source_sentences.append(node.source)
    target_sentences.append(node.target)

# Tokenize the source and target sentences
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(source_sentences + target_sentences)

source_sequences = tokenizer.texts_to_sequences(source_sentences)
target_sequences = tokenizer.texts_to_sequences(target_sentences)

# Pad the sequences to the same length
max_sequence_length = max(len(s) for s in source_sequences + target_sequences)
source_padded = tf.keras.preprocessing.sequence.pad_sequences(source_sequences, maxlen=max_sequence_length, padding="post", truncating="post")
target_padded = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_sequence_length, padding="post", truncating="post")

# Create a tf.data.Dataset from the padded sequences
dataset = tf.data.Dataset.from_tensor_slices((source_padded, target_padded))
dataset = dataset.batch(32)

# Define the encoder and decoder models
encoder_inputs = tf.keras.layers.Input(shape=(max_sequence_length,))
encoder_embedding = tf.keras.layers.Embedding(tokenizer.num_words, 128, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(128, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(max_sequence_length,))
decoder_embedding = tf.keras.layers.Embedding(tokenizer.num_words, 128, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(tokenizer.num_words, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

#Compile the model
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#Train the model
batch_size = 32
epochs = 10

history = model.fit([source_padded, target_padded], target_padded, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#Save the model
model.save("s2s.h5")