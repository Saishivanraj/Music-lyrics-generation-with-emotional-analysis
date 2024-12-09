from transformers import pipeline
import pandas as pd
import torch

device = 0 if torch.cuda.is_available() else -1

emotion_classifier = pipeline("text-classification",model="bhadresh-savani/bert-base-uncased-emotion",max_length=512,truncation=True,device=device)

pdf = pd.read_csv('/kaggle/working/processed_lyrics.csv')
sample_pdf = pdf.sample(n=20000, random_state=42).reset_index(drop=True)

def get_emotion(lyric):
    prediction = emotion_classifier(lyric[:512])
    return prediction[0]['label']

sample_pdf['emotion'] = sample_pdf['processed_lyrics'].apply(get_emotion)
sample_pdf.to_csv('/kaggle/working/sample_lyrics_with_emotions.csv', index=False)

print("Emotion labeling completed and saved to 'sample_lyrics_with_emotions.csv'")
