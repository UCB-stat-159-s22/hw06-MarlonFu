import pytest
import ligotools
from ligotools import readligo as rl
from ligotools import utils
import matplotlib.mlab as mlab
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import json
import h5py
import os

eventname = 'GW150914' 
events = json.load(open("data/BBH_events_v3.json","r"))
event = events[eventname]
tevent = event['tevent']   
fn_H1 = event['fn_H1']              # File name for H1 data
fn_L1 = event['fn_L1']  
fs = event['fs']  
fband = event['fband']  

deltat_sound = 2.    

strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')

# both H1 and L1 will have the same time vector, so:
time = time_H1
# the time sample interval (uniformly sampled!)
dt = time[1] - time[0]

NFFT = 4*fs
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

# We will use interpolations of the ASDs computed above for whitening:
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)
    
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')

# readligo tests
def test_loaddata_H1():
    assert len(strain_H1) == 131072
    assert len(time_H1) == 131072
    
def test_loaddata_L1():
    assert len(strain_L1) == 131072
    assert len(time_L1) == 131072
    
def test_loaddata_H1_L1():
    assert strain_L1.shape == strain_H1.shape
    assert time_L1.shape == time_H1.shape
    assert len(chan_dict_L1) == len(chan_dict_H1)

def test_keys():
    assert chan_dict_L1.keys() == chan_dict_H1.keys()
    
# util tests
def test_whiten():
    strain_H1_whiten = utils.whiten(strain_H1,psd_H1,dt)
    strain_L1_whiten = utils.whiten(strain_L1,psd_L1,dt)
    assert len(strain_H1_whiten) == 131072
    assert len(strain_L1_whiten) == 131072


strain_H1_whiten = utils.whiten(strain_H1,psd_H1,dt)
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization

indxd = np.where((time >= tevent-deltat_sound) & (time < tevent+deltat_sound))

def test_write_wavfile():
    utils.write_wavfile('audio/'+eventname+"_H1_whitenbp.wav",int(fs), strain_H1_whitenbp[indxd])
    read_in = wavfile.read('audio/'+eventname+"_H1_whitenbp.wav")
    assert len(read_in) == 2

def test_reqshift():
    strain_H1_shifted = utils.reqshift(strain_H1_whitenbp,fshift=400.,sample_rate=fs)
    #strain_L1_shifted = utils.reqshift(strain_L1_whitenbp,fshift=400.,sample_rate=fs)
    assert len(strain_H1_shifted) == 131072

def test_util_plot():
    assert 1 ==1