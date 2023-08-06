from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import math
import tensorflow as tf
from keras.models import load_model
import sys
import librosa
#from mutagen.mp3 import MP3
import json

# Create your views here.
def index(request):
    return render(request, 'index.html')

def classify(request):
    return render(request, 'classify.html')

def post(request):
    return render(request, 'post.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def process_input(audio_file, track_duration):

  SAMPLE_RATE = 22050
  NUM_MFCC = 13
  N_FTT=2048
  HOP_LENGTH=512
  TRACK_DURATION = track_duration # measured in seconds
  SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
  NUM_SEGMENTS = 10

  samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
  num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

  signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
  
  for d in range(10):

    # calculate start and finish sample for current segment
    start = samples_per_segment * d
    finish = start + samples_per_segment

    # extract mfcc
    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T

    return mfcc
    

def pred(request):
    if request.method=="POST":
        #sys.path.append(r"C:{Users\Dhirendra Sir\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.11}\Lib\site-packages")
        
        # get the audio file
        audio = request.FILES["audio"]
        
        #length = audio.info.length
        model = load_model("Music_Classification/model/Music_Genre_10_CNN.h5")
        mapping = ['Reggae','Disco','Jazz','Classical','Rock','Classical','Hiphop','Blues','Pop','Country','Metal']


        input_mfcc = process_input(audio, 30)
        X_to_predict = input_mfcc[np.newaxis, ..., np.newaxis]

        pred = model.predict(X_to_predict)
        prediction = np.argmax(pred, axis=1)
        return render(request, "predict.html", {"predictions":mapping[int(prediction)]})
    return render(request, "predict.html")