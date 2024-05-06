import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from attention import AttentionLayer
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pyarabic.araby as araby
from bs4 import BeautifulSoup
import re

#=====================================================================================================
#=====================================================================================================
# Seq2SeqModel
#=====================================================================================================
#=====================================================================================================
class Seq2SeqModel:
    def __init__(self, input_vocab_size, output_vocab_size, max_input_length, max_output_length, latent_dim=256, embedding_dim=100):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.m = None
    
    def build_model(self):
        K.clear_session()
        latent_dim = self.latent_dim
        embedding_dim=self.embedding_dim
        
        # Encoder
        encoder_inputs = Input(shape=(self.max_input_length,))

        #embedding layer
        enc_emb =  Embedding(self.input_vocab_size, embedding_dim,trainable=True)(encoder_inputs)

        #encoder lstm 1
        encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        #encoder lstm 2
        encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        #encoder lstm 3
        encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))

        #embedding layer
        dec_emb_layer = Embedding(self.output_vocab_size, embedding_dim,trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
        decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

        # Attention layer
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention input and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        #dense layer
        decoder_dense =  TimeDistributed(Dense(self.output_vocab_size, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Define the model 
        m = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.m = m
        return m

    def compile(self, optimizer='rmsprop', loss='sparse_categorical_crossentropy'):
        self.m.compile(optimizer=optimizer, loss=loss)
    
    def fit(self, X_train, y_train, checkpoints_saving_path=None, epochs=100, batch_size=10, validation_data=(), callbacks=[]):
        encoder_inputs= [X_train, y_train[:,:-1]]
        decoder_inputs = y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:]
        if len(validation_data)==2:
            X_validate= validation_data[0]
            y_validate = validation_data[1]
            encoder_validate_inputs= [X_validate, y_validate[:,:-1]]
            decoder_validate_inputs = y_validate.reshape(y_validate.shape[0],y_validate.shape[1], 1)[:,1:]

        if checkpoints_saving_path is not None and os.path.isdir(checkpoints_saving_path):
            # Save checkpoints every epoch
            checkpoint_filepath = checkpoints_saving_path+'/checkpoint-{epoch:02d}.model.keras'
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)

        params = {'callbacks':callbacks}
        if 'model_checkpoint_callback' in locals():
            params['callbacks'].append(model_checkpoint_callback)
        if 'encoder_validate_inputs' in locals():
            params['validation_data']=(encoder_validate_inputs, decoder_validate_inputs)
        
        return self.m.fit(encoder_inputs, decoder_inputs, epochs=epochs, batch_size=batch_size, **params)
    
    def fit_using_data_generator(self, train_generator, checkpoints_saving_path=None, epochs=100, validation_generator=None, callbacks=[]):
        if checkpoints_saving_path is not None and os.path.isdir(checkpoints_saving_path):
            # Save checkpoints every epoch
            checkpoint_filepath = checkpoints_saving_path+'/checkpoint-{epoch:02d}.model.keras'
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)

        params = {'callbacks':callbacks}
        if 'model_checkpoint_callback' in locals():
            params['callbacks'].append(model_checkpoint_callback)
        if validation_generator is not None:
            params['validation_data']=validation_generator
        
        return self.m.fit(train_generator, epochs=epochs, **params)
    
    def save(self, path):
        return self.m.save(path)

#=====================================================================================================
#=====================================================================================================
# Data Cleaner
#=====================================================================================================
#=====================================================================================================
class DataCleaner:
    def __init__(self):
        pass
    @staticmethod
    def __clean_text(text, substitutions_regex=[], lang='multi',remove_html_tags=True , remove_stop_words=True, expand_contractions=True, lower=True,replace_hindi_numbers_with_arabic=True, strip_tashkeel=True, strip_tatweel=True, uniform_arabic_characters=False):
        contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                            "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                            "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                            "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                            "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                            "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                            "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                            "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                            "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                            "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                            "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                            "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                            "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                            "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                            "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                            "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                            "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                            "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                            "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                            "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                            "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                            "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                            "you're": "you are", "you've": "you have"}
        
        stop_words = set(stopwords.words('english') + stopwords.words('arabic')) if lang in ['multi', 'multilingual'] else set(stopwords.words('arabic')) if lang in ['ar', 'arabic'] else set(stopwords.words('english'))

        if remove_html_tags:
            text=BeautifulSoup(text, "lxml").text
        if lower:
            text = text.lower()
        if expand_contractions:
            text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
        

        if remove_stop_words:
            tokens = [token for token in word_tokenize(text) if not token in stop_words]
            text = TreebankWordDetokenizer().detokenize(tokens)

        if replace_hindi_numbers_with_arabic:
            translation_table = str.maketrans('١٢٣٤٥٦٧٨٩٠', '1234567890')
            text = text.translate(translation_table)

        if strip_tashkeel:
            text = araby.strip_tashkeel(text)
        if strip_tatweel:
            text = araby.strip_tatweel(text)
        if uniform_arabic_characters:
            text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي').replace('ة','ه')
        if substitutions_regex:
            for sub in substitutions_regex:
                if isinstance(sub, (list, tuple)):
                    text = re.sub(sub[0],sub[1], text)
                else:
                    text = re.sub(sub,' ', text)
        return text

    @staticmethod
    def clean(data, substitutions_regex=[], lang='multi', remove_html_tags=True, remove_stop_words=True, expand_contractions=True, lower=True,replace_hindi_numbers_with_arabic=True , strip_tashkeel=True, strip_tatweel=True, uniform_arabic_characters=False):
        if isinstance(data, str):
            cleaned_data=DataCleaner.__clean_text(data, substitutions_regex, lang, remove_html_tags, remove_stop_words, expand_contractions, lower,replace_hindi_numbers_with_arabic, strip_tashkeel, strip_tatweel, uniform_arabic_characters)
        else:
            data = np.array(data)
            cleaned_data = []
            for sample in data:
                cleaned_data.append(DataCleaner.__clean_text(sample, substitutions_regex, lang, remove_html_tags, remove_stop_words, expand_contractions, lower,replace_hindi_numbers_with_arabic, strip_tashkeel, strip_tatweel, uniform_arabic_characters))
        
        return cleaned_data
    

#=====================================================================================================
#=====================================================================================================
# Data Generator
#=====================================================================================================
#=====================================================================================================
class TextSummaryWordLevelDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data_frame, tokenizer, text_column='text', summary_column='summary', max_text_len=512, max_summary_len=256, batch_size=10):
        self.df = data_frame
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.summary_column = summary_column
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__prepare_data(batches)
        return X, y

    def __prepare_data(self, df):
        input_texts = list(df[self.text_column])
        target_texts = list(df[self.summary_column])

        #convert text sequences into integer sequences & padding zero upto maximum length
        input_sequences    =   pad_sequences(self.tokenizer.texts_to_sequences(input_texts),  maxlen=self.max_text_len, padding='post')
        target_sequences   =   pad_sequences(self.tokenizer.texts_to_sequences(target_texts), maxlen=self.max_summary_len, padding='post')

        # encoder_inputs= [input_sequences, target_sequences[:,:-1]]
        decoder_inputs = target_sequences.reshape(target_sequences.shape[0],target_sequences.shape[1], 1)[:,1:]

        return (input_sequences, target_sequences[:,:-1]), decoder_inputs
    





