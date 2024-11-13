import pandas as pd
from deep_translator import GoogleTranslator
from khmernltk import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

df=  pd.read_csv("/home/ponleur.veng/image-caption/flickr8k/captions.txt")
train_df , test_df = train_test_split(df , test_size = 0.3,random_state=42, shuffle=True)

# print(f'Test_df: {len(test_df)}')

#train, val_df = train_test_split(train_df,test_size=0.2,random_state=42)

print(f'Total_df: {len(df)}, Train_df: {len(train_df)}, Val_df: {len(test_df)}')

def translate_en_km(raw_en_text):
    #     print(raw_en_text)
    translated = GoogleTranslator(source='en', target='km').translate(raw_en_text)
    word_tokenized = word_tokenize(translated)
    sentence = " ".join(word_tokenized)
    return sentence
tqdm.pandas(desc='Translating Caption')

train_df['khmer'] = train_df['caption'].progress_apply(lambda x: translate_en_km(x))
#val_df['khmer'] = val_df['caption'].progress_apply(lambda x: translate_en_km(x))
test_df['khmer'] = test_df['caption'].progress_apply(lambda x: translate_en_km(x))

train.to_csv('/home/ponleur.veng/image-caption/flickr8k/train70.csv')
#val_df.to_csv('/home/ponleur.veng/image-caption/flickr8k/val.csv')
test_df.to_csv('/home/ponleur.veng/image-caption/flickr8k/val30.csv')
