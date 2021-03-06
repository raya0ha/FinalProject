import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split

# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# # we allow these emotions
# AVAILABLE_EMOTIONS = {
#     "neutral",
#     "calm",
#     "happy",
#     "sad",
#     "angry",
#     "fearful",
#     "disgust",
#     "surprised",
#     "boredom"
# }

# we allow these emotions
AVAILABLE_EMOTIONS = {
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
}

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g.:
        `features = extract_feature(path, mel=True, mfcc=True)
        """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result

def load_data(test_size=0.25):
    X, y = [], []
    for file in glob.glob("data\\Actor_*\\*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
        #---------------------------------------------------------------------------------------------------------------
    for file in glob.glob("data\\arabic\\*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = (((basename.split("_")[1:])[0]).split(".")[:1])[0]
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    #-------------------------------------------------------------------------------------------------------------------
    for file in glob.glob("data\\emodb\\*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        label = (basename[5:6])
        if label == "A":
            emotion = 'angry'
        if label == "D":
            emotion = 'disgust'
        if label == "F":
            emotion = 'fearful'
        if label == "H":
            emotion = 'happy'
        if label == "S":
            emotion = 'sad'
        if label == "N":
            emotion = 'neutral'
        if label == "B":
            emotion = 'boredom'
        else:
            continue
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    #-------------------------------------------------------------------------------------------------------------------
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
# load RAVDESS dataset, 75% training 25% testing
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

