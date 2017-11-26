from keras.layers import Bidirectional
from keras import sqeuential

def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    
    model = Sequential()
    
    model.add(Bidirectional(GRU(units=french_vocab_size,return_sequences=True),\
                            input_shape = (input_shape[1:])))
    
    
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    
    model.summary()
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy']) 
    
    return model
 
    
tests.test_bd_model(bd_model)




# TODO: Train and Print prediction(s)


# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))



# Train the neural network
bd_rnn_model = bd_model(tmp_x.shape,\
                        preproc_french_sentences.shape[1],\
                        len(english_tokenizer.word_index),\
                        len(french_tokenizer.word_index))


bd_rnn_model.fit(tmp_x,\
                 preproc_french_sentences,\
                 batch_size=1024,\
                 epochs=10,\
                 validation_split=0.2)


# Print prediction(s)
print(logits_to_text(bd_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
