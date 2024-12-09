import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

dataset_path = '/kaggle/working/sample_lyrics_with_emotions.csv'
df = pd.read_csv(dataset_path)

def generate_lyrics(emotion: str, min_words: int = 75):
    emotion_lyrics = df[df['emotion'] == emotion]['processed_lyrics'].sample(1).values[0]
    
    emotion_prompt = f"The emotion is {emotion}. The lyrics start like this: "
    prompt = f"{emotion_prompt} {emotion_lyrics} [MASK]"
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(input_ids,max_new_tokens=150,num_return_sequences=1,num_beams=5,no_repeat_ngram_size=2)
    
    generated_lyrics = tokenizer.decode(output[0], skip_special_tokens=True)
    
    word_count = len(generated_lyrics.split())
    if word_count < min_words:
        while word_count < min_words:
            input_ids = tokenizer.encode(generated_lyrics + " [MASK]", return_tensors='pt')
            output = model.generate(input_ids,max_new_tokens=150,num_return_sequences=1,num_beams=5,no_repeat_ngram_size=2)
            generated_lyrics += " " + tokenizer.decode(output[0], skip_special_tokens=True)
            word_count = len(generated_lyrics.split())

    return generated_lyrics

if __name__ == "__main__":
    emotion = input("Enter the emotion (e.g., sadness, joy, love): ").strip().lower()
    
    if emotion not in df['emotion'].unique():
        print(f"Emotion '{emotion}' not found in the dataset. Please try again with a valid emotion.")
    else:
        generated_lyrics = generate_lyrics(emotion)
        print(f"Generated Lyrics for emotion '{emotion}':\n{generated_lyrics}")
