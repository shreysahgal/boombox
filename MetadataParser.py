"""Salami Dataset Metadata Parser."""
import os
import json
import csv
import pdb

class MetadataParser:
    """Parse metadata csv file into a dictionary."""
    """The fields are:
	0 SOURCE                     Either Codaich, IA (Internet Archive), or RWC - SAVE
	1 ANNOTATOR1                 ID number for first annotator
	2 ANNOTATOR2                 ID number for second annotator
    3 FILE_LOCATION              Server location of the piece
	4 SONG_DURATION              Duration of the piece, in seconds - SAVE
    5 EMPTY                      Empty field
	6 SONG_TITLE                 Title - SAVE
	7 ARTIST                     Artist - SAVE
    8 FORMAT                     Music file format - SAVE
	9 ANNOTATION_TIME1           Self-reported time to complete annotation for first annotator
	10 ANNOTATION_TIME2          Self-reported time to complete annotation for second annotator
	11 TEXTFILE1                 File path for first annotator's file - SAVE
	12 TEXTFILE2                 File path for second annotator's file - SAVE
	13 CLASS                     Broad genre (classical, jazz, popular, world, Live_Music_Archive, or unknown) - SAVE
	14 GENRE                     Narrow genre - SAVE
	15 SUBMISSION_DATE1          Date of submission of first annotation
	16 SUBMISSION_DATE2          Date of submission of second annotation
    17 SONG_WAS_PRIVATE_FLAG     Was the song private? 1 indicates yes; 0 indicates no.
    18 SONG_WAS_DISCARDED_FLAG   Was the song discarded? TRUE indicates yes; FALSE indicates no.
	19 XEQS1                     Was the first annotation converted automatically from X/= notation? X indicates yes; 0 indicates no. - SAVE
	20 XEQS2                     Was the second annotation converted automatically from X/= notation? X indicates yes; 0 indicates no. - SAVE
    """
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.metadata = self._parse_metadata()
        self._filter_metadata()
    
    def _parse_metadata(self):
        """Parse metadata csv file into a dictionary."""
        metadata = {}
        with open(self.metadata_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                metadata[row[0]] = row[1:]
        return metadata
    
    def _filter_metadata(self):
        """Filter metadata."""
        indecies_to_keep = [0, 4, 6, 7, 8, 13, 14, 19, 20]
        for key in self.metadata.keys():
            row = self.metadata[key]
            row = [row[i] for i in indecies_to_keep]
            self.metadata[key] = row
            
    def _pretty_print_row(self, key):
        """Pretty print row."""
        field_names = ['SOURCE', 'SONG_DURATION', 'SONG_TITLE', 'ARTIST', 'FORMAT', 'TEXTFILE1', 'TEXTFILE2', 'CLASS', 'GENRE', 'XEQS1', 'XEQS2']
        row = self.metadata[key]
        for i in range(len(row)):
            print(field_names[i], row[i])

if __name__ == '__main__':
    metadata_path = os.path.join('salami-data-public','metadata', 'metadata.csv')
    metadata_parser = MetadataParser(metadata_path)