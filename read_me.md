# Overview of the files in the repo


## Data visualisaion
- data_visualiser.py, evaluates the performance of the grid search by searching for the model with the highest accuracy.

## Testing file
- Testing_file.jpynb used for short test and extracting data, not a fixed test of functionality

# Overview of the grid_search performances

- Grid search 4-7 Checking learning values

- Grid search 9 with dropout

- Grid search 10 : 

- Grid search 11 : 20 kHz the effect of noise
- Visualizer datavisualisation_for_more_noise, 20kHzwiththenoise.pdf

- Grid search 12 : Final test 1.6 kHz sampling rate, with a half a symbol frequency dispersion.
- Visualiser data_visualizer_test_case,ipynb : test_best_accuracy.pdf, The_different_rates.pdf 

- Grid search 13 : Test of 2 kHz with different sizes of LRU memory state, 10-25-50-100-256
- Visualizer data_visualizer_mem_sizes.ipynb , "Different_mem_sizes.pdf"

- Grid Search 14 : Test of lessend phase upon intilisation [14.2],[16.666666666666668]_and[0.05]_samprate_1600
- 

- Grid search 15 : Made absolute doppler frequency in the model, and tested the dropout model implemented dropout as well, and dropout. Only one DLRU
                    Tried a larger batch_size as well.
                    Still not convergence
                    Still Adam in some of them, why does AdamW not work, fixed AdamW
                    Dataset size 1000 waveforms.
                    Best result is 18 % accuracy, which is still really bad.
                    No convergence.
                    


- Grid search 16 : Tested the effects of no Doppler uncertainty but a Doppler Rate instead only. Does not really converge either.

- Grid search 17 :  Zero Doppler with 14.2 dB/Hz noise.
                    Still only 61 % ?
                    526 size LRU perfroms at 0.6 percent. That's really nice
                    
- Grid search 18 : Add another layer to the deep learninging so another LRU block to see if there are any improvements.
    Right now it seems to be good, its much better.
    Of course it takes way longer to process now.
    Its much better 90% accuracy and the noise detection is bad.

- Grid search 19 : Add a slight doppler rate 5 Hz/s
    50.7 % acuracy pretty good. Converges hard to the middle frequencies.

- Grid search 20 : Add a higher Doppler rate 10 Hz/s, start a small grid search.
                    Best accuracy achieved is 29 %


- Grid search 21 : Add an even higher Doppler rate 16.67 Hz/s, just to do it really. Proves   
                    the lack of performance
