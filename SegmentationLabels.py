import os
import json
import csv
import pdb

class SegmentationLabels:

    def __init__(self, salami_ids) -> None:
        self.salami_ids = salami_ids
        
    def get_annotation(self, salami_id) -> list:
        """Get annotation."""
        annotations_1 = []
        annotations_2 = []
        annotations = []
        
        if os.path.isfile(os.path.join('salami-data-public', 'annotations', str(salami_id), 'parsed', 'textfile1_functions.txt')):
            with open(os.path.join('salami-data-public', 'annotations', str(salami_id), 'parsed', 'textfile1_functions.txt')) as f:
                        reader = csv.reader(f, delimiter='\t')
                        annotations_1 = list(reader)
                        #print(annotations)
        if os.path.isfile(os.path.join('salami-data-public', 'annotations', str(salami_id), 'parsed', 'textfile2_functions.txt')):
            with open(os.path.join('salami-data-public', 'annotations', str(salami_id), 'parsed', 'textfile2_functions.txt')) as f:
                        reader = csv.reader(f, delimiter='\t')
                        annotations_2 = list(reader)
                        #print(annotations)
                        
        # if  only one exists, take that one
        if len(annotations_2) == 0:
            return annotations_1
        elif len(annotations_1) == 0:
            return annotations_2
                        
        # If both exist, take the one with less 'no_function' labels
        # if they have the same number of 'no_function' labels, take the shorter one
        # if they have the same number of 'no_function' labels and the same length, take the first one
        if len(annotations_1) != len(annotations_2):
            if annotations_1.count('no_function') > annotations_2.count('no_function'):
                return annotations_2
            elif annotations_1.count('no_function') < annotations_2.count('no_function'):
                return annotations_1
            elif annotations_1.count('no_function') == annotations_2.count('no_function'):
                if len(annotations_1) > len(annotations_2):
                    return annotations_2
                elif len(annotations_1) < len(annotations_2):
                    return annotations_1
        elif len(annotations_1) == len(annotations_2):
            return annotations_1