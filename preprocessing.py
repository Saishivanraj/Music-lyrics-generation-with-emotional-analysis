
#import subprocess
#import sys


#def install_packages():
#    required_packages = [
 #'numpy',      # Add any packages your notebook uses
 #       'pandas',
   #     'scikit-learn',
   #     'nltk'
   # ]
   # for package in required_packages:
   #     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Call the install function to ensure all packages are installed
#install_packages()



import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline


df=pd.read_csv('/kaggle/input/genius-song-lyrics-with-language-information/song_lyrics.csv')
df.drop(['title','tag','artist','year','views','features','id','language_cld3','language_ft'],axis=1,inplace=True)
df['language'].value_counts()
df=df[df['language']=='en']
df = df.sample(n=50000, random_state=42)
def clean_lyrics(lyrics):
   
    lyrics = lyrics.lower()
    
    
    lyrics = re.sub(r'[^a-z\s]', '', lyrics) 
    return lyrics

df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)
def tokenize_lyrics(lyrics):
  
    return word_tokenize(lyrics)

df['tokenized_lyrics'] = df['cleaned_lyrics'].apply(tokenize_lyrics)
 
def simple_lemmatizer(tokens):
    
    return [word[:-1] if word.endswith('s') else word for word in tokens]

df['lemmatized_lyrics'] = df['tokenized_lyrics'].apply(simple_lemmatizer)
df['processed_lyrics'] = df['lemmatized_lyrics'].apply(lambda tokens: ' '.join(tokens))

print(df[['lyrics', 'processed_lyrics']].head())


processed_lyrics_df = df[['processed_lyrics']]


processed_lyrics_df.to_csv('processed_lyrics.csv', index=False)


print("Processed lyrics have been saved to 'processed_lyrics.csv'.")


pdf= pd.read_csv('processed_lyrics.csv')


print(pdf.head())


pdf['processed_lyrics'] = pdf['processed_lyrics'].str.replace(r'\b(intro|verse)\b', '', case=False, regex=True)


print(pdf[['processed_lyrics']].head())

pdf['processed_lyrics'] = pdf['processed_lyrics'].str.replace(r'\b(intro|verse)\b', '', case=False, regex=True)

pdf['processed_lyrics'] = pdf['processed_lyrics'].fillna('')

tokens = []
for sentence in pdf['processed_lyrics']:
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    tokens.extend(sentence.split())

vocab = set(tokens)
vocab_size = len(vocab)
print("Vocabulary Size:", vocab_size)

processed_lyrics_df.to_csv('/kaggle/working/processed_lyrics.csv', index=False)

print("Processed lyrics have been saved to 'processed_lyrics.csv'.")
