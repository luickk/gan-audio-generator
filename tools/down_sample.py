from subprocess import check_output
import glob
import librosa
import os
import ntpath

# Tool to down sample .wav files to lower sample rate

def main():
	sample_rate = 500
	if not os.path.exists('data/cv-valid-train-' + str(sample_rate)):
		os.makedirs('data/cv-valid-train-' + str(sample_rate))

	for fn in glob.iglob('data/cv-valid-train/*.wav'):
		file_name_n = ntpath.basename(fn)
		if not glob.glob('data/cv-valid-train-'+str(sample_rate)+'/'+file_name_n):
			y, sr = librosa.load(str(fn).split(".")[0] + '.wav', sr=sample_rate) # Downsample 44.1kHz to <SR>
			print(file_name_n)
			librosa.output.write_wav('data/cv-valid-train-'+str(sample_rate)+'/'+file_name_n, y, sr)
			print('data/cv-valid-train-'+str(sample_rate)+'/'+file_name_n)

if __name__ == '__main__':
    main()
