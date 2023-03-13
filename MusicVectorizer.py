import torch
from transformers import Wav2Vec2Processor, HubertModel
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.metrics import DistanceMetric
from tqdm import tqdm

"""
This class is used to vectorize audio files. It first uses a pre-trained word2vec
processor then uses the MAP-MERT model to vectorize the audio. The 'trajectory' of
a song is the vectorized audio at different time intervals.
"""

class MusicVectorizer:
    def __init__(self):
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Running on device: ", torch_device)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertModel.from_pretrained("m-a-p/MERT-v0").to(torch_device)
    
    def vectorize_audio(self, audio, sr, layer=12):  # model gives 13 possible feature layers
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")

        # breakpoint()
        with torch.no_grad():
            outputs = self.model(inputs['input_values'].cuda(), inputs['attention_mask'].cuda(), output_hidden_states=True)
        
        # hidden_states = torch.stack(outputs.hidden_states).squeeze()
        # return hidden_states.mean(-2)
        # breakpoint()
        return torch.stack([i.cpu() for i in outputs.hidden_states]).squeeze().mean(-2).numpy()  # returns a (13, 768) feature vector
        # # hidden_states = 
        # # return hidden_states.mean(-2)

    def trajectorize_song(self, song, sr, sample_time=5, layer=12, verbose=False):
        song_length = song.shape[0]
        song_time = song_length / sr
        num_samples = int(np.ceil(song_time / sample_time))
        song_vectors = np.zeros((num_samples, 13, 768))

        iter = tqdm(range(num_samples)) if verbose else range(num_samples)
        for i in iter:
            start = i * sample_time * sr
            end = min(start + sample_time * sr, song_length)
            song_vectors[i] = self.vectorize_audio(song[start:end], sr, layer)
        return song_vectors
    
    def plot_waveform(self, song, sr):
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(song, sr=sr)
        plt.show()
    
    def plot_spectrogram(self, song, sr):
        plt.figure(figsize=(14, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(song)), ref=np.max)
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')
        plt.show()