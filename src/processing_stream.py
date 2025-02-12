import os
from helper import get_mel_spectrogram, get_segments

def create_segments(audio_fullname):
	instrument_name = audio_fullname.split('_')[0]
 
	wav_file = os.path.join("Instrument", instrument_name, audio_fullname + ".wav")
	csv_file = os.path.join("Instrument", instrument_name, audio_fullname + ".csv")
	output_directory = os.path.join("Instrument", instrument_name, 'segments')

	get_segments(wav_file, csv_file, output_directory)

def create_mel_spectrogram(instrument_name):
	root_dir=os.path.join("Instrument", instrument_name, 'segments')

	for dirpath, dirnames, filenames in os.walk(root_dir):
		for file in filenames:
			if file.endswith(".wav"):
				# Construct the full file path
				audio_path = os.path.join(dirpath, file)
							
				# Call create_mel_spectrogram for each .wav file
				get_mel_spectrogram(audio_path)

if __name__ == "__main__":
	create_mel_spectrogram('筑')