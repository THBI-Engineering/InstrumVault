import os
import pandas as pd
from pydub import AudioSegment
from requests import get

def get_segments(wav_path, csv_path, output_dir):
	# Load the audio file
	audio = AudioSegment.from_wav(wav_path)

	# Read CSV
	df = pd.read_csv(csv_path)

	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)

	for _, row in df.iterrows():
		start_time = row["TIME"] * 1000  # Convert to milliseconds
		duration = row["DURATION"] * 1000  # Convert to milliseconds
		label = str(row["LABEL"])  # Convert to string to avoid issues
		value = str(row["VALUE"])  # Convert to string to avoid issues

		# Create label directory if it doesn't exist
		label_dir = os.path.join(output_dir, label)
		os.makedirs(label_dir, exist_ok=True)
  
		# Extract audio segment
		end_time = start_time + duration
		segment = audio[start_time:end_time]

		# Define output file name
		output_file = os.path.join(label_dir, f"{label}_{value}.wav")
		segment.export(output_file, format="wav")
		print(f"Saved: {output_file}")


import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def get_mel_spectrogram(audio_path, target_fs=16000, n_mels=256, window_length_ms=100, window_step_ms=25):
	output_path = os.path.splitext(audio_path)[0] + ".png"  # 保存为 PNG 以支持透明背景
	
	# Read WAV file
	fs, data = wavfile.read(audio_path)

	# Convert stereo to mono if needed
	if len(data.shape) > 1:
		data = np.mean(data, axis=1)  # Convert to mono

	# Resample to target_fs
	data = signal.resample(data, len(data) * target_fs // fs)
	fs = target_fs

	# Compute mel spectrogram using librosa
	D = librosa.stft(data, n_fft=int(window_length_ms * fs / 1000), hop_length=int(window_step_ms * fs / 1000))
	mel_spectrogram = librosa.feature.melspectrogram(S=np.abs(D), sr=fs, n_mels=n_mels)
	mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

	# Plot and save mel spectrogram
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_facecolor("none")  # 设置背景透明

	cmap = plt.cm.summer  # 绿色 colormap

	# Plot Mel spectrogram with custom colormap
	hop_length = int(window_step_ms * fs / 1000)
	img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=fs, hop_length=hop_length, cmap=cmap, ax=ax)
	
	# Customize font sizes
	plt.xlabel("Time (s)", fontsize=20, fontweight='bold')
	plt.ylabel("Mel Frequency (Hz)", fontsize=20, fontweight='bold')
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)

	# Adjust x-axis ticks to be multiples of 0.5
	duration = len(data) / fs  # Total duration in seconds
	x_ticks = np.arange(0, duration, 0.5)
	plt.xticks(x_ticks)

	# Save with transparent background
	ax = plt.gca()  # 获取当前轴对象
	for spine in ax.spines.values():
		spine.set_linewidth(3)  # 设置边框线宽

	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
	plt.close()

	print(f"Mel spectrogram saved as {output_path}")



if __name__ == "__main__":
	get_mel_spectrogram("Instrument/筑/segments/筑颤音/筑颤音_1.wav")