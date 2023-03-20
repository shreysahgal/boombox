import os
import yt_dlp
import csv
from MetadataParser import MetadataParser
from MusicVectorizer import MusicVectorizer
from MusicSegmentation import MusicSegmentation
import pathlib


class SalamiMachine:
    PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
    DOWNLOADED_AUDIO_FOLDER = PROJECT_ROOT / "downloaded_audio"
    TRANSFORMED_AUDIO_FOLDER = PROJECT_ROOT / "transformed_audio"
    
    songs = []

    # Youtube download and post-processing options
    YDL_OPTS = {
        'outtmpl': os.path.join(str(DOWNLOADED_AUDIO_FOLDER), u'%(id)s.%(ext)s'),
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    def __init__(self):
        if not os.path.exists(self.DOWNLOADED_AUDIO_FOLDER):
            os.mkdir(self.DOWNLOADED_AUDIO_FOLDER)
        if not os.path.exists(self.TRANSFORMED_AUDIO_FOLDER):
            os.mkdir(self.TRANSFORMED_AUDIO_FOLDER)
        metadata_path = self.PROJECT_ROOT / "salami-data-public" / "metadata" / "metadata.csv"
        self.parser = MetadataParser(metadata_path)
        self.mv = MusicVectorizer()
        self.segmenter = MusicSegmentation()
        
    def usage(self):
        print(" Create an instance of SalamiMachine and call one of the following methods:")
        print("  download_and_transform(): Downloads and transforms all paired videos.")
        print("  get_annotation_text(salami_id): Returns annotation text(s) for the specified song.")

    def download_video(self, youtube_id, redownload=False, sleep=0):
        global ydl_opts
        if (not os.path.exists(str(self.DOWNLOADED_AUDIO_FOLDER) + "/" + youtube_id + ".mp3")) or (redownload):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    x = ydl.download(['http://www.youtube.com/watch?v='+youtube_id])
                print("Successfully downloaded ({0})".format(youtube_id))
                return "downloaded"
                time.sleep(sleep)
            except (KeyboardInterrupt):
                raise
            except:
                print("Error downloading video ({0})".format(youtube_id))
                if (not os.path.exists(str(self.DOWNLOADED_AUDIO_FOLDER) + "/" + youtube_id + ".txt")):
                    open(str(self.DOWNLOADED_AUDIO_FOLDER) + "/" + youtube_id + ".txt", 'a').close()
                return "error"
        else:
            return "downloaded"
        
    def transform_video(self):
        """Transforms all audio to match SALAMI dataset."""
        os.subprocess.call(["python", "align_audio.py"])
        
    def download_and_transform(self):
        """Downloads and transforms all paired videos."""
        pairings_path = self.PROJECT_ROOT / "matching-salami" / "salami_youtube_pairings.csv"
        with open(pairings_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = self.download_video(row["youtube_id"])
                if status == "downloaded":
                    self.songs.append({
                        "salami_id": row["salami_id"],
                        "youtube_id": row["youtube_id"],
                    })
        self.transform_video()
        
    def get_annotation_text(self, salami_id):
        if os.path.exists(str(self.TRANSFORMED_AUDIO_FOLDER) + "/" + str(salami_id) + ".mp3"):
            metadata = self.parser.metadata(salami_id)
            if(metadata[5] != ''):
                textfile_1_path = self.PROJECT_ROOT/'salami-data-public'/'annotations'/row['salami_id']/'parsed'/'textfile1_functions.txt'
                with open(textfile_1_path, 'r') as f:
                    textfile_1 = f.read()
                    print("=== Annotation 1 ===")
                    print(textfile_1)
            if(metadata[6] != ''):
                textfile_2_path = self.PROJECT_ROOT/'salami-data-public'/'annotations'/row['salami_id']/'parsed'/'textfile2_functions.txt'
                with open(textfile_2_path, 'r') as f:
                    textfile_2 = f.read()
                    print("=== Annotation 2 ===")
                    print(textfile_2)
            if(metadata[5] == '' and metadata[6] == ''):
                print("No text file found for this song")
        else:
            print("No audio file found for this song")
            
    def generate_change_points(self):
        """Generates change points for all songs."""
        for song in self.songs:
            self.segmenter.generate_change_points(song["salami_id"])
    