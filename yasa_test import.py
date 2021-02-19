# -*- coding: utf-8 -*-
"""
Importing data from EDF

"""

# To create an conda environment from scratch using a .yml file with all required packages:
    # conda env create -f yasa_seals_environment.yml

# To write in the anaconda command prompt to update yasa_seals local conda environment with all of the packages required to locally run yasa (and changed staging.py file):
# conda env update --prefix .conda/envs/yasa_seals --file "C:\Users\Jessie\Documents\GitHub\Conda Environments/yasa_seals_environment.yml"  --prune

# If you change staging.py then you can upload it 
# if foo.py has changed:
#     unimport foo  <-- How do I do this?
#     import foo
#     myfoo = foo.Foo()

import yasa
import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import entropy as ent
import seaborn as sns

# Change working directory to sleep data folder
os.chdir("C:/Users/Jessie/Documents/Dissertation Sleep/Data")
print("Current Working Directory ", os.getcwd())

# Load the EDF file, excluding the EOGs and EKG channels
raw = mne.io.read_raw_edf('Land_Scoring_6_excerpt.edf', preload=True, exclude=['MagZ'])
# raw.resample(100)                      # Downsample the data to 100 Hz
# raw.filter(0.1, 40)                    # Apply a bandpass filter from 0.1 to 40 Hz
# raw.pick_channels(['C4-A1', 'C3-A2'])  # Select a subset of EEG channels
raw # Outputs summary data about file

#Inspect Data
print('The channels are:', raw.ch_names)
print('The sampling frequency is:', raw.info['sfreq'])

# Let's now load the human-scored hypnogram, where each value represents a 30-sec epoch.
# hypno = np.loadtxt('sub-02_hypno_30s.txt', dtype=str)
# hypno

# Apply the detection using yasa.spindles_detect
sp = yasa.spindles_detect(raw)

# Display the results using .summary()
sp.summary()


# We first need to specify the channel names and, optionally, the age and sex of the participant
# - "raw" is the name of the variable containing the polysomnography data loaded with MNE.
# - "eeg_name" is the name of the EEG channel, preferentially a central derivation (e.g. C4-M1). This is always required to run the sleep staging algorithm.
# - in my case will set "eeg_name" to "LEEG3_Ch12"
# - "eog_name" is the name of the EOG channel (e.g. LOC-M1). This is optional.
# - in my case will set "eog_name" to "LEOG_Ch2"
# - "emg_name" is the name of the EOG channel (e.g. EMG1-EMG3). This is optional.
# - in my case will set "emg_name" to "LEMG_Ch5"
# - "metadata" is a dictionary containing the age and sex of the participant. This is optional.
sls = yasa.SleepStaging(raw, eeg_name="LEEG3_Ch12", eog_name="LEOG_Ch2", emg_name="LEMG_Ch5", metadata=dict(age=1.8, male=False))

# Getting the predicted sleep stages is now as easy as:
y_pred = sls.predict()
y_pred
