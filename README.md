# Speech-Emotion-Recognition-
A Deep Learning (LSTM) model with keras.
In nowadays the speech is the most natural way for human to communicate with so by this project we can know the emotion of the person A classification model is developed in a Deep Learning method, meaning a Deep Neural Network (DNN) while an advanced model for time-series analysis has been chosen, which is the Long Short-Term Memory (LSTM).
by his speech that will help companies to improve themselves 
First Stage is datasets:
1-	Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D) Used Only 1100 audio
2-	RAVDESS Emotional Speech Audio (RAVDESS) Used All emotions except the Calm Emotion 
3-	Toronto Emotional Speech Set (TESS) 
Second Stage the Preprocessing State: (This stage used for Preprocessing the dataset to used it in the model)
1-	Audio Segment Object 
2-	Normalize
3-	Trim Silence from beginning and ending
4-	Padding for length
5-	Noise reduction 
6-	Get an array of Samples
Third Stage Feature from Librosa:
1-	Root Mean Square (RMS)
2-	Zero Crossed Rate (ZCR)
3-	Mel-Frequency Cepstral Coefficients (MFCCs)
4-	Chroma 
Results had shown an accuracy of 78% of emotional recognition from speech. 