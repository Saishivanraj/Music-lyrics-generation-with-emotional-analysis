
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
    # Convert to lowercase
    lyrics = lyrics.lower()
    
    # Remove special characters, numbers, and punctuation
    lyrics = re.sub(r'[^a-z\s]', '', lyrics)  # Keep only lowercase alphabets and spaces
    
    return lyrics

df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)
def tokenize_lyrics(lyrics):
    # Tokenize the lyrics into words
    return word_tokenize(lyrics)

df['tokenized_lyrics'] = df['cleaned_lyrics'].apply(tokenize_lyrics)
 
def simple_lemmatizer(tokens):
    # A very basic rule to remove 's' for plural words (you can expand this list as needed)
    return [word[:-1] if word.endswith('s') else word for word in tokens]

df['lemmatized_lyrics'] = df['tokenized_lyrics'].apply(simple_lemmatizer)
df['processed_lyrics'] = df['lemmatized_lyrics'].apply(lambda tokens: ' '.join(tokens))

# Display the first few rows of the processed dataframe
print(df[['lyrics', 'processed_lyrics']].head())

# Step 1: Select only the 'processed_lyrics' column
processed_lyrics_df = df[['processed_lyrics']]

# Step 2: Save this selected column to a CSV file
processed_lyrics_df.to_csv('processed_lyrics.csv', index=False)

# Confirm that the file is saved
print("Processed lyrics have been saved to 'processed_lyrics.csv'.")

# Step 1: Load the processed lyrics CSV file into a DataFrame
pdf= pd.read_csv('processed_lyrics.csv')

# Step 2: Check the first few rows to ensure it's loaded correctly
print(pdf.head())

# Step 2: Remove "intro" and "verse" from the lyrics (case insensitive)
pdf['processed_lyrics'] = pdf['processed_lyrics'].str.replace(r'\b(intro|verse)\b', '', case=False, regex=True)

# Step 3: Check the DataFrame after cleaning
print(pdf[['processed_lyrics']].head())
