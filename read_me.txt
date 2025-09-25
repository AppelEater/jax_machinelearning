# Overview of the files in the repo


## Data visualisaion
- data_visualiser.py, evaluates the performance of the grid search by searching for the model with the highest accuracy.

## Testing file
- Testing_file.jpynb used for short test and extracting data, not a fixed test of functionality

# Overview of the grid_search performances

- Grid search
  File: ????
  Comment: Steady frequeny, CN0 15 dbHz

- Grid search 2
  File: ????
  Comment: Steady frequeny, CN0 15 dbHz

- Grid search 3
  File: ????
  Comment: Steady frequeny, CN0 15 dbHz

- Grid search 4 : Checking learning values
  File: ????
  Comment: Steady frequeny, CN0 15 dbHz

- Grid search 5 : Checking learning values
  File: ????
  Comment: Steady frequeny, CN0 15 dbHz

- Grid search 6 : Checking learning values
  File: ????
  Comment: Steady frequeny, CN0 15 dbHz

- Grid search 7 : Checking learning values
  File: ????
  Comment: Steady frequeny, CN0 15 dbHz

- Grid search 8
  File: ????
  Comment: MFSK signal, with DU [0 0.01 ] and DRU [16.66]. CN0 varies between [15, 20] dbHz

- Grid search 9 : with dropout
  File: ????
  Comment: MFSK signal, with DU [0 0.01 ] and DRU [16.66]. CN0 varies between [15, 20] dbHz

- Grid search 10 : 
  File: ????
  Comment: MFSK signal, with 1250 Hz spacing and 20 kHz to show that this shits sucks.

- Grid search 11 : 20 kHz the effect of noise
                   Visualizer datavisualisation_for_more_noise, 20kHzwiththenoise.pdf
  File: ????
  Comment: MFSK signal, with 1250 Hz spacing and 20 kHz to show that this shits sucks.

- Grid search 12 : Final test 1.6 kHz sampling rate, with a half a symbol frequency dispersion.
                   Visualiser data_visualizer_test_case,ipynb : test_best_accuracy.pdf, The_different_rates.pdf 
  File: accu_test_waveforms_CNO_[14.2],[16.666666666666668]_and[0.05]_samprate_1600.pkl
  Comment: MFSK signal, with 100 Hz spacing and exact sampling rate <<-- MAY BE WRONG

- Grid search 13 : Test of 2 kHz with different sizes of LRU memory state, 10-25-50-100-256
                   Visualizer data_visualizer_mem_sizes.ipynb , "Different_mem_sizes.pdf"
  File: accu_test_waveforms_CNO_[15],[0.01]_and[0]_samprate_2000.pkl
  Comment: MFSK signal, with 100 Hz spacing and 2kHz sampling rate and different memsizes

- Grid Search 14 : Test of lessend phase upon intilisation [14.2],[16.666666666666668]_and[0.05]_samprate_1600
  File: accu_test_waveforms_CNO_[14.2],[16.666666666666668]_and[0.05]_samprate_1600.pkl
  Comment: MFSK signal, with 100 Hz spacing and 1.6kHz sampling rate with different learning rates and init phases pi/10

- Grid search 15 : Made absolute doppler frequency in the model, and tested the dropout model implemented dropout as well, and dropout. Only one DLRU
                   Tried a larger batch_size as well.
                   Still not convergence
                   Still Adam in some of them, why does AdamW not work, fixed AdamW
                   Dataset size 1000 waveforms.
                   Best result is 18 % accuracy, which is still really bad.
                   No convergence.
  File: absolute_doppler_waveforms_CNO_[14.2],[16.666666666666668]_and10_samprate_2000.pkl
  Comment: MFSK signal with 10 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty

- Grid search 16 : Tested the effects of no Doppler uncertainty but a Doppler Rate instead only. Does not really converge either.
  File: absolute_doppler_waveforms_CNO_[14.2],[16.666666666666668]_and0_samprate_2000.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty

- Grid search 17 : Zero Doppler with 14.2 dB/Hz noise.
                   Still only 61 % ?
                   526 size LRU perfroms at 0.6 percent. That's really nice
  File: absolute_doppler_waveforms_CNO_[14.2],[0]_and0_samprate_2000.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 0 Hz/s doppler rate uncertainty
                   
- Grid search 18 : Add another layer to the deep learninging so another LRU block to see if there are any improvements.
                   Right now it seems to be good, its much better.
                   Of course it takes way longer to process now.
                   Its much better 90% accuracy and the noise detection is bad.
  File: absolute_doppler_waveforms_CNO_[14.2],[0]_and0_samprate_2000.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 0 Hz/s doppler rate uncertainty, 14 db/Hz, increase size to two layers.

- Grid search 19 : Add a slight doppler rate 5 Hz/s
                   50.7 % acuracy pretty good. Converges hard to the middle frequencies.
  File: absolute_doppler_waveforms_CNO_[14.2],[5]_and0_samprate_2000_1736425291.1487653.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 0 Hz/s doppler rate uncertainty, 14 db/Hz, increase size to two layers.

- Grid search 20 : Add a higher Doppler rate 10 Hz/s, start a small grid search.
                   Best accuracy achieved is 29 %
  File: absolute_doppler_waveforms_CNO_[14.2],[10]_and0_samprate_2000_1736437896.2224333.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 0 Hz/s doppler rate uncertainty, 14 db/Hz, increase size to two layers.

- Grid search 21 : Add an even higher Doppler rate 16.67 Hz/s, just to do it really. Proves the lack of performance
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and0_samprate_2000_1736520994.6523163.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 0 Hz/s doppler rate uncertainty, 14 db/Hz, increase size to two layers. <<-- MAY BE WRONG

- Grid search 22 : EMPTY

- Grid search 23 : MFSK signal with 0 Hz doppler uncertainty and 0 Hz/s doppler rate uncertainty, 14 db/Hz, increase size to two layers.
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0002 
  File: absolute_doppler_waveforms_CNO_[14.2],[0]_and0_samprate_2000.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 0 Hz/s doppler rate uncertainty, 14 db/Hz, increase size to two layers.

- Grid search 24 : MFSK signal with 0 Hz doppler uncertainty and 5 Hz/s doppler rate uncertainty, 14 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0002
  File: absolute_doppler_waveforms_CNO_[14.2],[5]_and0_samprate_2000_1736425291.1487653.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 5 Hz/s doppler rate uncertainty, 14 db/Hz. STFT

- Grid search 25 : MFSK signal with 0 Hz doppler uncertainty and 5 Hz/s doppler rate uncertainty, 14 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0002
  File: absolute_doppler_waveforms_CNO_[14.2],[10]_and0_samprate_2000_1736437896.2224333.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 5 Hz/s doppler rate uncertainty, 14 db/Hz. STFT
 
- Grid search 26 : MFSK signal with 0 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0002
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and0_samprate_2000_1736520994.6523163.pkl
  Comment: MFSK signal with 0 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
 
- Grid search 27 : MFSK signal with 10 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0002
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and10_samprate_2000_1736866451.4345906.pkl
  Comment: MFSK signal with 10 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT

- Grid search 28 : MFSK signal with 20 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0002
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and20_samprate_2000_1736868989.3079593.pkl
  Comment: MFSK signal with 20 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT

- Grid search 29 : MFSK signal with 30 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0002
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and30_samprate_2000_1736933738.418169.pkl
  Comment: MFSK signal with 30 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT

- Grid search 30 : MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0002
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and90_samprate_2000_1736935556.6539564.pkl
  Comment: MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT

- Grid search 31 : MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 256, 'Hop': 64
		    - 'Schedule Type': 'Constant', 'Value': 0.0001
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and90_samprate_2000_1736935556.6539564.pkl
  Comment: MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT

- Grid search 32 : MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 512, 'Hop': 128
		    - 'Schedule Type': 'Constant', 'Value': 0.0001
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and90_samprate_2000_1736935556.6539564.pkl
  Comment: MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT

- Grid search 33 : MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
		    - 'Window Filter': 'hann', 'Length': 512, 'Hop': 256
		    - 'Schedule Type': 'Constant', 'Value': 0.0001
  File: absolute_doppler_waveforms_CNO_[14.2],[16.67]_and90_samprate_2000_1736935556.6539564.pkl
  Comment: MFSK signal with 90 Hz doppler uncertainty and 16.67 Hz/s doppler rate uncertainty, 14.2 db/Hz. STFT
