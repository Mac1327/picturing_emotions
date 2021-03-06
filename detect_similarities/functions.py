import imageio
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from os import walk

## There are only 2 major functions you might want to use for the Data base 
    ## which are the last two:
            ## - detect_duplicated_pics
            ## - removing_duplicated_pics


class Similarity:
    
    def __init__(self, path):
        self.path = path
        self.files = None
        self.files_dict = None
        self.describe_dict = None
        self.describe_df = None
        self.duplicated_pics = None
        self.cleaned_files = None


    def get_pixels(self,array):
        pixels = []
        for row in array:
            for pixel in row:
                pixels.append(pixel)
        return pixels
    
    def removing_zone_identifier(self):
        for f in os.listdir(self.path):
            if f.endswith('Zone.Identifier'):
                os.remove(f'{self.path}/{f}')

    def get_current_No_images(self):
        self.removing_zone_identifier()
        self.files = []
        for (dirpath, dirnames, filenames) in walk(self.path):
            self.files.extend(filenames)
        print(f'>>> Current folder has {len(self.files)} pictures')


    def build_a_dict(self):
        self.files_dict = {}
        self.get_current_No_images()
        
        for f in self.files:
            self.files_dict[f] = \
            self.get_pixels(imageio.imread(f'{self.path}/{f}'))


    def build_DF(self):
        self.build_a_dict()
        df = pd.DataFrame(self.files_dict)

        self.describe_dict = {}
        for col in df.columns:
            self.describe_dict[col] = df[col].describe()
        
        self.describe_df = pd.DataFrame(self.describe_dict).loc[['mean','25%','50%','75%']].T
        return self.describe_df


    def detect_duplicated_pics(self):
        self.build_DF()
        dup = self.describe_df[self.describe_df.duplicated(['mean','25%','50%','75%'])].sort_values('mean')
        self.duplicated_pics = dup.index
        print(f">>> There are {self.duplicated_pics.shape[0]} duplicated pictures been detected")
        return self.duplicated_pics


    def removing_duplicated_pics(self):
        self.detect_duplicated_pics()
        for p in os.listdir(f'{self.path}'): 
            if p in self.duplicated_pics:
                os.remove(f'{self.path}/{p}')
        self.cleaned_files = []
        for (dirpath, dirnames, filenames) in walk(f'{self.path}'):
            self.cleaned_files.extend(filenames)
        print(f'>>> Cleaned Database now has reduced to {len(self.cleaned_files)} pictures')

    