def echo_log(action, text=""):
    print("\033[4m\033[1m"+action+"\033[0m\033[0m "+text)

#=======================================================================
#   Import Modules
#=======================================================================
import os
from libs import DataCleaner, Seq2SeqModel, TextSummaryWordLevelDataGenerator
import numpy as np
import pandas as pd 
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

#=======================================================================
max_text_len=512 #word
max_summary_len=256 #word
#=======================================================================

#=======================================================================
#   Read Dataset and preform some preprocessing
#=======================================================================
prepared_dataset_path = 'text_summary_prepared.csv'
echo_log("Searching...", f"for prepared data set on path: {prepared_dataset_path}")
if os.path.exists(prepared_dataset_path):
    echo_log("Exist.", f"prepared dataset exist on: {prepared_dataset_path}")
    echo_log("reading dateset")
    df=pd.read_csv(prepared_dataset_path) #,nrows=1000
else:
    echo_log("Not Found.", f"prepared dataset not exist on: {prepared_dataset_path}")
    echo_log("reading default dateset", "on text_summary.csv")
    df=pd.read_csv("text_summary.csv") #,nrows=1000
    echo_log("Begin Cleaning")
    df.drop_duplicates(subset=['text'],inplace=True) #dropping duplicates
    df.dropna(axis=0,inplace=True)#dropping na

    echo_log("cleaning ......")
    #Preprocessing
    df['text'] = DataCleaner.clean(df['text'], uniform_arabic_characters=True)
    df['summary'] = DataCleaner.clean(df['summary'], remove_stop_words=False, uniform_arabic_characters=True)

    df.replace('', np.nan, inplace=True)
    df.dropna(axis=0,inplace=True)

    df['summary'] = df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')
    short_text=[]
    short_summary=[]
    cleaned_text = np.array(df['text'])
    cleaned_summary=np.array(df['summary'])

    echo_log("filtering .......")
    for i in range(len(cleaned_text)):
        if(len(cleaned_text[i].split())<max_text_len and len(cleaned_summary[i].split())<max_summary_len):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])
    del cleaned_text, cleaned_summary
    df=pd.DataFrame({'text':short_text,'summary':short_summary})
    df.to_csv(prepared_dataset_path, index=False)
    echo_log("Successfully Cleaned")

#=======================================================================
#   Initialize tokenizer, split dataset
#=======================================================================
tokenizer_path = "./outputs/tokenizer.json"
echo_log("Searching...", f"for saved tokenizer on path: {tokenizer_path}")
if os.path.exists(tokenizer_path):
    echo_log("Loading tokenizer ...", f"from: {tokenizer_path}")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        loaded_tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(loaded_tokenizer_json)
else:
    echo_log("Init tokenizer and fit on texts ...")
    #prepare a tokenizer for reviews on training data
    tokenizer = Tokenizer(oov_token='OOV') 
    tokenizer.fit_on_texts(list(df['summary']) + list(df['text']))
    echo_log("Number of Vocab:",str(len(tokenizer.word_index)))
    echo_log("Reducing number of vocab...")
    #reduce words in tokenizer
    unique_words = ' '.join([word for word in tokenizer.word_index.keys() if word in tokenizer.word_counts.keys() and tokenizer.word_counts[word] >= 100])
    del tokenizer
    tokenizer = Tokenizer(oov_token='OOV')
    tokenizer.fit_on_texts(['sostok', 'eostok']+[unique_words])
    echo_log("Number of Vocab:",str(len(tokenizer.word_index)))

    # Save tokenizer to a JSON file
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    echo_log("Saving tokenizer ...", f"json_file on => {tokenizer_path}")

echo_log("Number of Vocab:",str(len(tokenizer.word_index)))

x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']),np.array(df['summary']),test_size=0.1,random_state=0,shuffle=False)

# Using data generator
train_gen = TextSummaryWordLevelDataGenerator(data_frame=pd.DataFrame({"text": x_tr, "summary": y_tr}), tokenizer=tokenizer, batch_size=1)
valid_gen = TextSummaryWordLevelDataGenerator(data_frame=pd.DataFrame({"text": x_val, "summary": y_val}), tokenizer=tokenizer, batch_size=1)



#=======================================================================
#   Build, Compile, Train Model
#=======================================================================
vocab_size = len(tokenizer.word_index) + 1

model = Seq2SeqModel(input_vocab_size=vocab_size, output_vocab_size=vocab_size,
                    max_input_length=max_text_len, max_output_length=max_summary_len, latent_dim=300, embedding_dim=100);
model.build_model()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# model.fit(X_train= x_tr, y_train=y_tr, checkpoints_saving_path='outputs/checkpoints', batch_size=10, validation_data=(x_val, y_val))
model.fit_using_data_generator(train_generator=train_gen, checkpoints_saving_path='outputs/checkpoints',
                                validation_generator=valid_gen)
model.save('outputs/s2s_model.keras')