from helper import get_spectrogram

if __name__ == "__main__":
	get_spectrogram("Instrument/筑/segments/筑颤音/筑颤音_2.wav", target_fs=16000, xlimit_max=3.6,  ylimit_max=1200, window_length_ms=100, window_step_ms=1)
	get_spectrogram("Instrument/筑/segments/筑单音/筑单音_10.wav", target_fs=16000, xlimit_max=4, ylimit_max=1200, window_length_ms=100, window_step_ms=1)
	get_spectrogram("Instrument/筑/segments/筑上滑音/筑上滑音_10.wav", target_fs=16000, xlimit_max=3.6, ylimit_max=1200, window_length_ms=100, window_step_ms=1)
	get_spectrogram("Instrument/亚筝/segments/亚筝_单音/亚筝_单音_6.wav", target_fs=16000, xlimit_max=3.6, ylimit_max=1200, window_length_ms=167, window_step_ms=1)
	get_spectrogram("Instrument/亚筝/segments/亚筝_颤音/亚筝_颤音_2.wav", target_fs=16000, xlimit_max=3.6, ylimit_max=1200, window_length_ms=167, window_step_ms=1)
	get_spectrogram("Instrument/亚筝/segments/亚筝_上滑音/亚筝_上滑音_5.wav", target_fs=16000, xlimit_max=3.3, ylimit_max=1200, window_length_ms=167, window_step_ms=1)
	get_spectrogram("Instrument/阮/segments/阮-1.wav", target_fs=16000, ylimit_max=6000, window_length_ms=50, window_step_ms=1)
	get_spectrogram("Instrument/打击乐/segments/打击乐-1.wav", target_fs=16000, ylimit_max=3000, window_length_ms=100, window_step_ms=1)
	get_spectrogram("Instrument/颤音合奏/segments/筑亚筝合奏颤音/筑亚筝合奏颤音_2.wav", target_fs=16000, ylimit_max=1200, window_length_ms=167, window_step_ms=1, apply_colorbar=True)