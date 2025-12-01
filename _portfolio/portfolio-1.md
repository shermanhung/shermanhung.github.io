---
title: "Audio Projects"
excerpt: "This folder includes audio-related projects with detailed explanations of their objectives and code implementations.<br/>" 
collection: portfolio
---
## Project Goal

This project presents the design and implementation of a simple audio rendering pipeline that reconstructs the first two bars of Johann Sebastian Bach’s Invention No. 1 in C Major using pre-recorded audio samples.

The provided dataset consists of two main components: a `samples/` directory containing short WAV recordings of individual musical notes, and a `music_data.txt` file that encodes the pitch and duration information for the piece. However, the data file contains several notational inconsistencies that must be identified and corrected before audio rendering can take place.

In addition to the code implementation, detailed comments and documentation are provided to explain the processing steps and the nature of the corrections applied to the music data. The resulting output demonstrates a basic yet effective approach to reconstructing a musical passage from symbolic data using sample-based synthesis techniques.

To access the full code, please visit this folder (https://drive.google.com/drive/folders/1-IxWroIc3-FuCdG1PrcINujfWXqDtTJE?usp=sharing) and check `proejct_code.ipynb`.


```python
import numpy as np
import pandas as pd
import operator as op
import math
import wave
import os
import struct
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import json
import pickle
from scipy.io import wavfile
```

## Step 1 — Map Audio Samples to Note Names

Create a dictionary that maps musical note names (e.g., `"C4"`, `"D#5"`, `"R"`) to their corresponding WAV file paths in the `samples/` directory. This provides a direct lookup for each note during rendering.

To identify the musical note from each audio sample, I used a pitch detection algorithm from the Python library librosa, which estimates the fundamental frequency (f₀) of a sound over time. The algorithm then calculates the median of these frequency values (in Hz) to obtain a stable representation of the dominant pitch, reducing the influence of noise and minor fluctuations. Finally, the resulting frequency (e.g., 261.63 Hz) is converted into its corresponding musical note (e.g., “C4”).


```python
def detect_note(filename):
  # Load audio file
  y, sr = librosa.load(filename)

  # Estimate pitch (fundamental frequency)
  f0, voiced_flag, voiced_prob = librosa.pyin(
      y,
      fmin=librosa.note_to_hz('E1'),
      fmax=librosa.note_to_hz('E6')
  )

  # Get the most common (dominant) frequency
  valid_f0 = f0[~np.isnan(f0)]
  if len(valid_f0) == 0:
      print("No pitch detected.")
  else:
      mean_f0 = np.median(valid_f0)
      note_name = librosa.hz_to_note(mean_f0)
      print(f"Detected pitch: {mean_f0:.2f} Hz → {note_name}")

  return note_name

def create_dict(sp_addr, inst_type):

  if 'c' in inst_type:
    sp_files = [sp_addr + f for f in os.listdir(sp_addr) if 'c' in f]
  elif 'b' in inst_type:
    sp_files = [sp_addr + f for f in os.listdir(sp_addr) if 'b' in f]

  dict_map = {}

  for sp_file in sp_files:
    note = detect_note(sp_file)
    sp_file_name = sp_file.split('/')[-1]
    dict_map[note] = sp_file_name

    print(sp_file_name, note)

  return dict_map
```


```python
# Create audio sample to musical note dictionary
SAMPLES_DIR = "samples"
sp_addr = '/content/drive/MyDrive/Colab Notebooks/tune_synthesis/' + SAMPLES_DIR + '/'
c_note_dict = create_dict(sp_addr, 'clairnet')
b_note_dict = create_dict(sp_addr, 'bass')

# Save dictionary to .pkl file
with open('/content/drive/MyDrive/Colab Notebooks/tune_synthesis/clarinet_note_dict.pkl', 'wb') as fp:
    pickle.dump(c_note_dict, fp)
    print('Clarinet dictionary saved successfully to file')

with open('/content/drive/MyDrive/Colab Notebooks/tune_synthesis/bass_note_dict.pkl', 'wb') as fp:
    pickle.dump(b_note_dict, fp)
    print('Bass dictionary saved successfully to file')
```


```python
c_note_dict = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/tune_synthesis/clarinet_note_dict.pkl')
b_note_dict = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/tune_synthesis/bass_note_dict.pkl')
```



## Step 2 — Parse and Validate the Music Data File

Read `music_data.txt` and identify structural or notational inconsistencies.  
The following issues were corrected:

1. **Extra trailing tokens:**  
   Both the treble (clarinet) and bass parts contained stray tokens at the ends of their sequences (for example, a standalone duration symbol such as `Q`, or a trailing `C4`).  
   These caused the total bar durations to exceed the intended measure length.  
   The extra tokens were removed so that each part cleanly sums to four beats per bar.

2. **Misplaced note/duration token:**  
   In the bass part, one entry (`E3`) appeared where a duration token was expected.  
   This was likely a typographical error.  
   It was replaced with an eighth-note duration symbol (`E`) to restore consistent rhythm structure.


```python
# Load music data file
DATA_FILE = "music_data.txt"
md_addr = '/content/drive/MyDrive/Colab Notebooks/tune_synthesis/' + DATA_FILE
md = open(md_addr, 'r').read().splitlines()

# Fix music data file
c_seqs = md[3] + ' ' + md[4] + '' + md[5]
c_seqs = c_seqs.split(' ')
c_seqs = c_seqs[0:-2]

b_seqs = md[8] + '' + md[9] + ' ' + md[10]
b_seqs = b_seqs.split(' ')
b_seqs[11] = 'E'
b_seqs = b_seqs[0:-2]
```



## Step 3 — Build Note Sequences (Duration + Pitch)

Using the corrected `music_data.txt`, construct separate note sequences for the treble and bass parts.  

Each sequence is expressed as pairs of duration and pitch tokens — for example: `(S, C4), (Q, B4), (E, R)`.

Rests are represented as `"R"` or as `(duration, "R")` pairs to indicate silence for a specified duration.


```python
# Create note sequence
def create_note_seq(seqs):
  note_seqs = []
  tempo_mark = None
  for seq in seqs:
    if seq in ['S', 'E', 'Q', 'H']:
      tempo_mark = seq
    else:
      note_seqs.append([tempo_mark, seq])

  return note_seqs
```


```python
c_note_seqs = create_note_seq(c_seqs)
b_note_seqs = create_note_seq(b_seqs)
```

## Step 4 — Render Tracks from Samples

For each `(duration, pitch)` pair, load the corresponding audio sample using its mapped file path.  
Trim or pad each sample to match the exact duration implied by the tempo and note length.  

After rendering both treble and bass tracks:
- Zero-pad the shorter track so both parts align in time.  
- Mix them together using simple summation.  
- Normalize the final waveform to prevent clipping before exporting it as `output.wav`.


```python
def render(note_seqs, note_dict, DURATIONS, sr):
  output = np.zeros(0)

  for note_seq in note_seqs[:]:
    dur = note_seq[0]
    note = note_seq[1]
    note_length = int(DURATIONS[dur] * sr)

    if note == 'R':
      y = np.zeros(note_length)
    else:
      sp_file = note_dict[note]
      y, sr_ = librosa.load('/content/drive/MyDrive/Colab Notebooks/tune_synthesis/samples/' + sp_file)
      y = y[:note_length]

    if len(y) > note_length:
      y = y[:note_length]
    else:
      y = np.pad(y, (0, note_length - len(y)))
    # print(dur, note, note_length)
    output = np.concatenate((output, y))

  return output
```


```python
# === Configuration ===
sr = 44100 # sample rate of audio files
TEMPO = 70  # BPM
BEAT_DURATION = 60 / TEMPO # seconds per beat

# Note durations (S=sixteenth, E=eighth, Q=quarter, H=half)
DURATIONS = {
    'S': BEAT_DURATION / 4,
    'E': BEAT_DURATION / 2,
    'Q': BEAT_DURATION,
    'H': BEAT_DURATION * 2
}
```


```python
# Render tracks
c_audio = render(c_note_seqs, c_note_dict, DURATIONS, sr)
b_audio = render(b_note_seqs, b_note_dict, DURATIONS, sr)

# c_audio = render_c(c_note_seqs, c_note_dict, DURATIONS, sr)
# print('-----------------------')
# b_audio = render_b(b_note_seqs, b_note_dict, DURATIONS, sr)

# Pad shorter track for overlay
max_len = max(len(c_audio), len(b_audio))
c_audio = np.pad(c_audio, (0, max_len - len(c_audio)))
b_audio = np.pad(b_audio, (0, max_len - len(b_audio)))

# Mix both tracks (normalize to avoid clipping)
combined = c_audio + b_audio
combined /= np.max(np.abs(combined))

# Save tracks
sf.write("/content/drive/MyDrive/Colab Notebooks/tune_synthesis/bach_invention_output_clarinet.wav", c_audio, sr)
sf.write("/content/drive/MyDrive/Colab Notebooks/tune_synthesis/bach_invention_output_bass.wav", b_audio, sr)
sf.write("/content/drive/MyDrive/Colab Notebooks/tune_synthesis/bach_invention_output_combined.wav", combined, sr)
```


