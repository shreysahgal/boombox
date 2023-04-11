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
                        # replace 'no_function' with the last annotation
                        for i in range(len(annotations_1)):
                            if annotations_1[i][1] == 'no_function':
                                annotations_1[i][1] = annotations_1[i-1][1]
        if os.path.isfile(os.path.join('salami-data-public', 'annotations', str(salami_id), 'parsed', 'textfile2_functions.txt')):
            with open(os.path.join('salami-data-public', 'annotations', str(salami_id), 'parsed', 'textfile2_functions.txt')) as f:
                        reader = csv.reader(f, delimiter='\t')
                        annotations_2 = list(reader)
                        # replace 'no_function' with the last annotation
                        for i in range(len(annotations_2)):
                            if annotations_2[i][1] == 'no_function':
                                annotations_2[i][1] = annotations_2[i-1][1]
                        
        # if  only one exists, take that one
        if len(annotations_2) == 0:
            return annotations_1
        elif len(annotations_1) == 0:
            return annotations_2
                                    
        if len(annotations_1) != len(annotations_2):
            # return the shorter one
            if len(annotations_1) < len(annotations_2):
                return annotations_1
            else:
                return annotations_2
        elif len(annotations_1) == len(annotations_2):
            return annotations_1
        
    def create_label(self, timestep, annotation) -> list:
        """Create a label for each segment."""
        labels = []
        for i in range(10):
            # append annotation if its time step is between timestep*i and timestep*(i+1)
            lower_bound = timestep*i
            upper_bound = timestep*(i+1)
            label = []
            for j in range(len(annotation)-1):
                annotation_start = float(annotation[j][0])
                annotation_end = float(annotation[j+1][0])
                # if the annotation start or end is contained in lower and upper bound, append the label
                if (lower_bound <= annotation_start <= upper_bound) or (lower_bound <= annotation_end <= upper_bound):
                    label.append(annotation[j][1])
                if (annotation_start <= lower_bound <= annotation_end) and (annotation_start <= upper_bound <= annotation_end):
                    label.append(annotation[j][1])
            labels.append(label)
        for i, group in enumerate(labels):
            # remove duplicates
            labels[i] = list(set(group))
        return labels
    
   