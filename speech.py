import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import IPython.display as ipd

def analyze_audio_identically(file_path):
    try:

        sampling_rate, samples = wavfile.read(file_path)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)


        frame_length_ms = 20
        hop_length_ms = 10
        frame_length = int(frame_length_ms / 1000 * sampling_rate)
        hop_length = int(hop_length_ms / 1000 * sampling_rate)
        
        num_frames = 1 + int((len(samples) - frame_length) / hop_length)

        energy = np.zeros(num_frames)
        zcr = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = samples[start:end]
            
            energy[i] = np.sum(frame**2)
            
            zcr[i] = np.sum(np.abs(np.diff(np.sign(frame.astype(float))))) / (2 * frame_length)
            
        rms_energy = np.sqrt(energy / frame_length)
            
        energy_threshold = 0.01 * np.max(energy)
        zcr_threshold = 0.15
        
        is_voiced = (energy > energy_threshold) & (zcr < zcr_threshold)

        sample_time = np.arange(len(samples)) / float(sampling_rate)
        frame_time = np.arange(num_frames) * hop_length / float(sampling_rate)

        print(f"--- Identical Analysis for {file_path} ---")
        plt.figure(figsize=(15, 4))
        plt.plot(sample_time, samples, color='black', linewidth=0.7)
        plt.title(f'Voiced/Unvoiced Segments for {file_path} (Original Style)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        start_time = 0
        current_status = is_voiced[0]
        for i in range(1, num_frames):
            if is_voiced[i] != current_status:
                end_time = frame_time[i]
                color = 'red' if current_status else 'blue'
                plt.axvspan(start_time, end_time, color=color, alpha=0.3)
                start_time = end_time
                current_status = is_voiced[i]
        plt.axvspan(start_time, frame_time[-1], color='red' if current_status else 'blue', alpha=0.3)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        plt.figure(figsize=(15, 4))
        plt.plot(frame_time, zcr, color='magenta')
        plt.title(f'Zero-Crossing Rate (ZCR) for {file_path}')
        plt.xlabel('Time (s)')
        plt.ylabel('ZCR')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        plt.figure(figsize=(15, 4))
        plt.plot(frame_time, rms_energy, color='cyan')
        plt.title(f'RMS Energy for {file_path}')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        print(f"Audio for {file_path}:")
        ipd.display(ipd.Audio(file_path))
        print("-" * 60 + "\n")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

audio_files = ["a.wav", "e.wav", "i.wav", "o.wav", "u.wav"]
for file in audio_files:
    analyze_audio_identically(file)
