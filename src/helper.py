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
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import gaussian_filter  # Import Gaussian filter

def get_spectrogram(audio_path, target_fs=16000, window_length_ms=100, window_step_ms=25, ylimit_max=None, xlimit_max=None, apply_gaussian_filter=True, sigma=1, apply_colorbar=False):
	output_path = os.path.splitext(audio_path)[0] + f"_{target_fs}_{window_length_ms}_{window_step_ms}.png"  # 修改文件名

	# Read WAV file
	fs, data = wavfile.read(audio_path)

	# Convert stereo to mono if needed
	if len(data.shape) > 1:
		data = np.mean(data, axis=1)  # Convert to mono

	# Resample to target_fs
	data = signal.resample(data, len(data) * target_fs // fs)
	fs = target_fs

	# Compute STFT (no mel filter)
	n_fft = int(window_length_ms * fs / 1000)  # Number of FFT points
	hop_length = int(window_step_ms * fs / 1000)  # Step size
	D = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)

	# Convert amplitude to dB scale
	spectrogram_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

	# Apply Gaussian filter if enabled
	if apply_gaussian_filter:
		spectrogram_db = gaussian_filter(spectrogram_db, sigma=sigma)  # Apply Gaussian filter with specified sigma

	# Plot STFT spectrogram
	fig, ax = plt.subplots(figsize=(12, 8))
	if apply_colorbar:
		fig, ax = plt.subplots(figsize=(14, 8))

	# 加粗 x 轴、y 轴刻度线
	ax.tick_params(axis='both', which='major', length=12, width=3, direction='out')  # 主刻度线
	ax.tick_params(axis='x', which='minor', length=8, width=2, direction='out')  # 主刻度线
	
	ax.set_facecolor("none")  # 设置背景透明
	cmap = 'inferno'  # You can change this colormap to 'magma', 'plasma', etc.

	# Plot spectrogram with custom colormap
	img = librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='linear', 
								   sr=fs, hop_length=hop_length, cmap=cmap, ax=ax)


	if apply_colorbar:
		# 添加 colorbar，标注为 Sound Pressure Level (L/dB)
		cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
		# cbar.set_label("Sound Pressure Level (L/dB)", fontsize=30, fontweight="bold")  # 设置标签
		cbar.ax.tick_params(labelsize=30)  # 调整 colorbar 字体大小

	# 设置频率范围 (ylim)
	if ylimit_max is not None:
		ax.set_ylim(0, ylimit_max)

	# 设置时间轴最大值 (xlim)
	if xlimit_max is not None:
		ax.set_xlim(0, xlimit_max)

	# Remove x and y labels
	plt.xlabel("")
	plt.ylabel("")

	# Customize font sizes for ticks
	plt.xticks(fontsize=30)
	plt.yticks(fontsize=30)

	# Adjust x-axis ticks to be multiples of 0.5 and add unit 's' for seconds
	duration = len(data) / fs  # Total duration in seconds
	x_ticks = np.arange(0, duration, 0.5)
	plt.xticks(x_ticks, [f'0' if x == 0 else (f'{int(x)}s' if x % 1 == 0 else f'{x:.1f}s') for x in x_ticks])

	# Add 'Hz' unit to y-axis ticks
	y_ticks = ax.get_yticks()
	# plt.yticks(y_ticks, [f"0" if y == 0 else ( f"{y/1000:g}kHz") for y in y_ticks])		# kHz
	plt.yticks(y_ticks, [f'0' if y == 0 else f'{int(y)} Hz' for y in y_ticks])	# Hz

	# Save with transparent background
	ax = plt.gca()  # 获取当前轴对象
	for spine in ax.spines.values():
		spine.set_linewidth(3)  # 设置边框线宽

	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
	plt.close()

	print(f"Spectrogram saved as {output_path}")

def get_log_spectrogram(audio_path, target_fs=16000, window_length_ms=100, window_step_ms=25, ylimit_max=None, xlimit_max=None, apply_gaussian_filter=True, sigma=1):
	output_path = os.path.splitext(audio_path)[0] + f"_{target_fs}_{window_length_ms}_{window_step_ms}.png"  # 修改文件名

	# Read WAV file
	fs, data = wavfile.read(audio_path)

	# Convert stereo to mono if needed
	if len(data.shape) > 1:
		data = np.mean(data, axis=1)  # Convert to mono

	# Resample to target_fs
	data = signal.resample(data, len(data) * target_fs // fs)
	fs = target_fs

	# Compute STFT (no mel filter)
	n_fft = int(window_length_ms * fs / 1000)  # Number of FFT points
	hop_length = int(window_step_ms * fs / 1000)  # Step size
	D = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)

	# Convert amplitude to dB scale
	spectrogram_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
	# spectrogram_spl = spectrogram_db + 94  # 参考声压转换
	# spectrogram_db=spectrogram_spl

	# Apply Gaussian filter if enabled
	if apply_gaussian_filter:
		spectrogram_db = gaussian_filter(spectrogram_db, sigma=sigma)  # Apply Gaussian filter with specified sigma

	# Plot STFT spectrogram
	fig, ax = plt.subplots(figsize=(24, 16))
	# 加粗 x 轴、y 轴刻度线
	ax.tick_params(axis='both', which='major', length=12, width=3, direction='out')  # 主刻度线
	ax.set_facecolor("none")  # 设置背景透明
	cmap = 'inferno'  # You can change this colormap to 'magma', 'plasma', etc.

	# Plot spectrogram with custom colormap
	img = librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='linear', 
								   sr=fs, hop_length=hop_length, cmap=cmap, ax=ax)

	ax.set_yscale('log', base=10)

	# 添加 colorbar，标注为 Sound Pressure Level (L/dB)
	cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
	# cbar.set_label("Sound Pressure Level (L/dB)", fontsize=30, fontweight="bold")  # 设置标签
	cbar.ax.tick_params(labelsize=30)  # 调整 colorbar 字体大小


	# 设置频率范围 (ylim)
	if ylimit_max is not None:
		ax.set_ylim(10, ylimit_max)

	# 设置时间轴最大值 (xlim)
	if xlimit_max is not None:
		ax.set_xlim(0, xlimit_max)

	# Remove x and y labels
	plt.xlabel("")
	plt.ylabel("")

	# Customize font sizes for ticks
	plt.xticks(fontsize=30)
	plt.yticks(fontsize=30)
	
	# Adjust x-axis ticks to be multiples of 10 and add unit 's' for seconds
	duration = len(data) / fs  # Total duration in seconds
	x_ticks = np.arange(0, duration, 10)
	plt.xticks(x_ticks, [f'0' if x == 0 else (f'{int(x)}s' if x % 1 == 0 else f'{x:.1f}s') for x in x_ticks])
	
	# Add 'Hz' unit to y-axis ticks
	yticks = [10, 20, 50, 100, 200, 500, 1000]  # 指定刻度位置
	ax.set_yticks(yticks)  # 设置刻度
	ax.set_yticklabels([f"{y}Hz" for y in yticks])  # 设置刻度标签


	# Save with transparent background
	ax = plt.gca()  # 获取当前轴对象
	for spine in ax.spines.values():
		spine.set_linewidth(3)  # 设置边框线宽

	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
	plt.close()

	print(f"Spectrogram saved as {output_path}")

if __name__ == "__main__":
	# get_spectrogram("Instrument/筑/segments/筑颤音/筑颤音_2.wav", target_fs=16000, xlimit_max=4,  ylimit_max=1200, window_length_ms=100, window_step_ms=1)
	get_log_spectrogram("Instrument/阮钟/segments/阮钟_合奏-1.wav", target_fs=16000,  xlimit_max=50, ylimit_max=1000, window_length_ms=500, window_step_ms=10)
	 