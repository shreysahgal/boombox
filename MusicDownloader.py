from yt_dlp import YoutubeDL
from multiprocessing import Pool
import os
from tqdm import tqdm

# PLAYLISTS = [
#     ('reggaeton', 'https://youtube.com/playlist?list=PLPPgknP7IOwXmkAE99CdZ2h4baKNEvzev', 186),
#     ('2000s_indie', 'https://youtube.com/playlist?list=PL4BrNFx1j7E7BDa131cA8Aax0QtfDbWeS', 258),
#     ('2000s_alt_rock', 'https://youtube.com/playlist?list=PL6Lt9p1lIRZ311J9ZHuzkR5A3xesae2pk', 250),
#     ('2000s_rnb', 'https://youtube.com/playlist?list=PLplXQ2cg9B_phh5cj3tEPibzumbF5hWh5', 250),
#     ('2010s_hiphop', 'https://youtube.com/playlist?list=PLF26C807923FB7B9B', 182),
#     ('2010s_pop', 'https://youtube.com/playlist?list=PL7Q2ZklqtR8B_EAUfXt5tAZkxhCApfFkL', 300),
#     ('country', 'https://youtube.com/playlist?list=PL3oW2tjiIxvQW6c-4Iry8Bpp3QId40S5S', 150),
#     ('90s_rock', 'https://youtube.com/playlist?list=PLD58ECddxRngHs9gZPQWOCAKwV1hTtYe4', 300),
#     ('80s_pop', 'https://youtube.com/playlist?list=PLmXxqSJJq-yXrCPGIT2gn8b34JjOrl4Xf', 182),
#     ('90s_hiphop', 'https://youtube.com/playlist?list=PLmXxqSJJq-yXvmRMuHu7vd2XdJEUgsr33', 201),
#     ('classical', 'https://youtube.com/playlist?list=PLxvodScTx2RtAOoajGSu6ad4p8P8uXKQk', 100),
#     ('lofi', 'https://youtube.com/playlist?list=PLOzDu-MXXLliO9fBNZOQTBDddoA3FzZUo', 250),
#     ('jazz', 'https://youtube.com/playlist?list=PL8F6B0753B2CCA128', 273),
#     ('movie_scores', 'https://youtube.com/playlist?list=PL4BrNFx1j7E5qDxSPIkeXgBqX0J7WaB2a', 200)
# ]

# PATH = '/home/shrey/Documents/eecs448-boombox/data/large_dataset_pca'

# for genre, link, count in playlists:
#     data_path = os.path.join(path, genre)
#     if not os.path.exists(data_path):
#         print("making directory: ", data_path)
#         os.mkdir(data_path)

#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'outtmpl': os.path.join(data_path, '%(title)s.%(ext)s'),
#         'playlistend': count,
#         'postprocessors': [{  # Extract audio using ffmpeg
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'mp3',
#         }],
#         'ignoreerrors': True,
#     }

#     with YoutubeDL(ydl_opts) as ydl:
#         error_code = ydl.download([link])

class MusicDownloader:

    def __init__(self, playlists, save_dir, make_subdir_if_not_exist=True):
        self.playlists = playlists
        self.save_dir = save_dir
        self.make_subdir_if_not_exist = make_subdir_if_not_exist

    def download_playlist(self, playlist : list[tuple[str, str, int]]):
        genre, link, count = playlist
        data_path = os.path.join(self.path, genre)
        if not os.path.exists(data_path) and self.make_subdir_if_not_exist:
            print("making directory: ", data_path)
            os.mkdir(data_path)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(data_path, '%(title)s.%(ext)s'),
            'playlistend': count,
            'postprocessors': [{  # Extract audio using ffmpeg
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'ignoreerrors': True,
            'quiet': True,
            'outtmpl': os.path.join(data_path, '%(id)s*%(title)s.%(ext)s'),
        }

        with YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([link])
            return error_code
    
    def download(self):
        with Pool(len(self.playlists)) as p:
            r = p.map(self.download_playlist, tqdm(self.playlists))
        return r

def main():
    playlists = [
        ('reggaeton', 'https://youtube.com/playlist?list=PLPPgknP7IOwXmkAE99CdZ2h4baKNEvzev', 186),
        ('2000s_indie', 'https://youtube.com/playlist?list=PL4BrNFx1j7E7BDa131cA8Aax0QtfDbWeS', 258),
        ('2000s_alt_rock', 'https://youtube.com/playlist?list=PL6Lt9p1lIRZ311J9ZHuzkR5A3xesae2pk', 250),
        ('2000s_rnb', 'https://youtube.com/playlist?list=PLplXQ2cg9B_phh5cj3tEPibzumbF5hWh5', 250),
        ('2010s_hiphop', 'https://youtube.com/playlist?list=PLF26C807923FB7B9B', 182),
        ('2010s_pop', 'https://youtube.com/playlist?list=PL7Q2ZklqtR8B_EAUfXt5tAZkxhCApfFkL', 300),
        ('country', 'https://youtube.com/playlist?list=PL3oW2tjiIxvQW6c-4Iry8Bpp3QId40S5S', 150),
        ('90s_rock', 'https://youtube.com/playlist?list=PLD58ECddxRngHs9gZPQWOCAKwV1hTtYe4', 300),
        ('80s_pop', 'https://youtube.com/playlist?list=PLmXxqSJJq-yXrCPGIT2gn8b34JjOrl4Xf', 182),
        ('90s_hiphop', 'https://youtube.com/playlist?list=PLmXxqSJJq-yXvmRMuHu7vd2XdJEUgsr33', 201),
        ('classical', 'https://youtube.com/playlist?list=PLxvodScTx2RtAOoajGSu6ad4p8P8uXKQk', 100),
        ('lofi', 'https://youtube.com/playlist?list=PLOzDu-MXXLliO9fBNZOQTBDddoA3FzZUo', 250),
        ('jazz', 'https://youtube.com/playlist?list=PL8F6B0753B2CCA128', 273),
        ('movie_scores', 'https://youtube.com/playlist?list=PL4BrNFx1j7E5qDxSPIkeXgBqX0J7WaB2a', 200)
    ]

    save_dir = '/home/shrey/Documents/eecs448-boombox/data/large_dataset_with_'

    downloader = MusicDownloader(playlists, save_dir)
    downloader.download()