# SinoOto UTAU

SinoOto UTAU is an experimental AI-assisted oto.ini generator for Chinese CVVC voicebanks.

The project is designed to learn from existing manually configured Chinese UTAU voicebanks and generate initial oto.ini parameters for new recordings.

## Status

This project is in early development.

Current focus:

- Chinese CVVC voicebanks
- oto.ini parsing
- wav feature extraction
- entry-level training data generation
- CPU-friendly model training
- automatic initial oto.ini generation

## What this project does

SinoOto UTAU reads existing voicebanks containing:

- wav files
- oto.ini files

It converts each oto.ini entry into a training sample.

For example, a single wav file may contain multiple CVVC entries:

```txt
a_ba_pa_ta.wav=a,100,200,-1000,80,40
a_ba_pa_ta.wav=a b,430,180,-800,90,45
a_ba_pa_ta.wav=ba,500,160,-700,100,50