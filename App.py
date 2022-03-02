# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:02:10 2022

@author: rafikelnaggar
"""

import sounddevice as sd
import soundfile as sf
import wavio as wv
import os
from tkinter import *
  
import numpy as np

import librosa
from pydub import AudioSegment, effects
import noisereduce as nr

import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd  
   
def Voice_rec():
    fs = 48000
      
    # seconds
    duration = 2
    myrecording = sd.rec(int(duration * fs), 
                         samplerate=fs, channels=2)
    
     
    sd.wait()
    path = r"C:\Users\rafikelnaggar\Desktop\asd\Simples"
    sound_name = "ode.wav"
    wv.write(os.path.join(path, sound_name), myrecording, fs, sampwidth=2) 
    asd = ""
    ASD = test()
    Label(master, text=ASD
     ).grid(row=50, sticky=W, rowspan=4)
  
master = Tk()
  
Label(master, text=" Voice Recoder : "
     ).grid(row=0, sticky=W, rowspan=4)
  
  
b = Button(master, text="Start", command=Voice_rec)
b.grid(row=0, column=2, columnspan=2, rowspan=2,
       padx=0, pady=20)

mainloop()
 
 
def test ():


    # emotions = []
    
    # Initialize variables
    total_length = 228864 #228864  #305152  #5005152    # desired frame length for all of the audio samples.
    frame_length = 2048
    hop_length = 512
    folder_path = r"C:\Users\rafikelnaggar\Desktop\asd\Simples"
    for subdir, dirs, files in os.walk(folder_path):
      for file in files: 
        print (file)
        rms = []
        zcr = []
        mfcc = []
        chroma = []
        # Fetch the sample rate.
        _, sr = librosa.load(path = os.path.join(subdir,file), sr = None) # sr (the sample rate) is used for librosa's MFCCs. '_' is irrelevant.
        # Load the audio file.
        rawsound = AudioSegment.from_file(os.path.join(subdir,file)) 
        # Normalize the audio to +5.0 dBFS.
        normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
        # Transform the normalized audio to np.array of samples.
        normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')
        # Trim silence from the beginning and the end.
        xt,  index = librosa.effects.trim(normal_x, top_db=30)
        # Pad for duration equalization.
        # print(xt.shape)
        padded_x = np.pad(xt, (0, total_length-len(xt)), 'constant')
        # Noise reduction.
        final_x = nr.reduce_noise(y=padded_x,y_noise=padded_x, sr=sr)
        
        # Features extraction 
        f1 = librosa.feature.rms(final_x, frame_length=frame_length, hop_length=hop_length) # Energy - Root Mean Square
        f2 = librosa.feature.zero_crossing_rate(final_x , frame_length=frame_length, hop_length=hop_length, center=True) # ZCR
        f3 = librosa.feature.mfcc(final_x, sr=sr, n_mfcc=13, hop_length = hop_length) # MFCC
        f4 = librosa.feature.chroma_stft(final_x,sr=sr) #chroma
         
        rms.append(f1)
        zcr.append(f2)
        mfcc.append(f3)
        chroma.append(f4)
        
        
        f_rms = np.asarray(rms).astype('float32')
        f_rms = np.swapaxes(f_rms,1,2)
        f_zcr = np.asarray(zcr).astype('float32')
        f_zcr = np.swapaxes(f_zcr,1,2)
        f_mfccs = np.asarray(mfcc).astype('float32')
        f_mfccs = np.swapaxes(f_mfccs,1,2)
        f_chroma = np.asarray(chroma).astype('float32')
        f_chroma = np.swapaxes(f_chroma,1,2)
        
        X = np.concatenate(( f_rms,f_zcr,f_mfccs,f_chroma), axis=2)  #
        print(X.shape)
        
        saved_model_path = r'C:\Users\rafikelnaggar\Desktop\asd\model8723.json'
        saved_weights_path = r'C:\Users\rafikelnaggar\Desktop\asd\model8723_weights.h5'
        
        with open(saved_model_path, 'r') as json_file:
            json_savedModel = json_file.read()
            
        model = tf.keras.models.model_from_json(json_savedModel)
        model.load_weights(saved_weights_path)
        
        model.compile(loss='categorical_crossentropy', 
                        optimizer='RMSProp', 
                        metrics=['categorical_accuracy'])
        print(model.summary())
        predictions = model.predict(X, use_multiprocessing=True)
        pred_list = list(predictions)
        pred_np = np.squeeze(np.array(pred_list).tolist(), axis=0)
        
        
        emotions = {
            0 : 'neutral',
            1 : 'happy',
            2 : 'sad',
            3 : 'angry',
            4 : 'fearful',  
            5 : 'disgust',
            6 : 'suprised'   
        }  
        emo_list = list(emotions.values())
        plt.bar(emo_list, pred_np, color = 'darkturquoise')
        plt.ylabel("Probabilty (%)")
        plt.show()
        max_emo = np.argmax(predictions) 
        print('max emotion:', emotions.get(max_emo,-1))
        return emotions.get(max_emo,-1)
     
