from subprocess import check_output
import glob

# Tool to reformat .mp3 files to .wav files using ffmpeg

def main():
	for fn in glob.iglob('data/cv-valid-train/*.mp3'):
		print(str(fn).split(".")[0] + '.wav')
		if not glob.glob(str(fn).split(".")[0] + '.wav'):
			check_output("ffmpeg -i "+fn+" -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav "+str(fn).split('.')[0]+".wav", shell=True)


if __name__ == '__main__':
    main()
