from __future__ import print_function, division
import os
import pandas as pd
import time
import sox

# 
# 
# Zero-pads and trims all audio files to match the versions used in SALAMI.
# 
# 

matchlist_csv_filename = os.getcwd() + "/salami_youtube_pairings.csv"
downloaded_audio_folder = os.getcwd() + "/downloaded_audio"
transformed_audio_folder = os.getcwd() + "/transformed_audio"

# Specify location of downloaded audio

downloaded_audio_folder = os.getcwd() + "/downloaded_audio"
if not os.path.exists(downloaded_audio_folder):
	os.makedirs(downloaded_audio_folder)

if not os.path.exists(transformed_audio_folder):
	os.makedirs(transformed_audio_folder)

def reshape_audio(salami_id, match_data):
	row = {colname: match_data[colname][match_data.salami_id==salami_id].values[0]  for colname in match_data.columns}
	input_filename = downloaded_audio_folder + "/" + str(row["youtube_id"]) + ".mp3"
	output_filename = transformed_audio_folder + "/" + str(row["salami_id"]) + ".mp3"
	start_time_in_yt = row["onset_in_youtube"] - row["onset_in_salami"]
	# = - row["time_offset"]
	end_time_in_yt = start_time_in_yt + row["salami_length"]
	tfm = sox.Transformer()
	if end_time_in_yt > row["youtube_length"]:
		tfm.pad(end_duration=end_time_in_yt - row["youtube_length"])
	if start_time_in_yt < 0:
		tfm.pad(start_duration=-start_time_in_yt)
		start_time_in_yt = 0
	# Select portion of youtube file to match salami
	tfm.trim(start_time_in_yt, start_time_in_yt+row["salami_length"])
	tfm.build(input_filename, output_filename)

if __name__ == "__main__":
	match_data = pd.read_csv(matchlist_csv_filename, header=0)
	match_data = match_data.fillna("")
	for ind in match_data.index:
		try:
			reshape_audio(match_data.salami_id[ind], match_data)
			time.sleep(2)
		except (KeyboardInterrupt):
			raise
		except:
			print("Error while attempting to process row {0}: {1} (salami_id {2}).".format(ind,match_data.youtube_id[ind],match_data.salami_id[ind]))
