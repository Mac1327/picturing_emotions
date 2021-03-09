import imageio
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from os import walk
import cv2

class Similarity:
    
    def __init__(self, path):
        self.path = path

        #file name        
        self.files = None
        self.files_png = None
        self.files_jpg = None

        #files_name_dict
        self.files_dict = None
        self.files_jpg_dict = None
        self.files_png_dict = None
        
        #describe_df 
        self.describe_df = None
        self.describe_df_png = None
        self.describe_df_jpg = None

        #dup pictures list
        self.dup_png_pic = None
        self.dup_jpg_pic = None
        self.dup_png_and_jpg = None

        self.duplicated_pics = None
        self.cleaned_files = None


    def get_pixels(self,array):
        if len(array.shape) > 2:
            # 3D
            pixels = []
            for first_layer in array:
                for sec_layer in first_layer:
                    for pixel in sec_layer:
                        pixels.append(pixel)
        else:
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

        self.files_png = []
        self.files_jpg = []

        for (dirpath, dirnames, filenames) in walk(self.path):
            for name in filenames:
                if name.endswith('png'):
                    self.files_png.append(name)
                else:
                    self.files_jpg.append(name)
        print(f'''>>> This file has {len(self.files_png)} pictures in [.png] format, {len(self.files_jpg)} pictures in [.jpg] format''')
    
    def build_a_dict(self):
        #call functions:
        self.get_current_No_images()
  
        self.files_png_dict = {}
        for f in self.files_png:
            self.files_png_dict[f] = self.get_pixels(imageio.imread(f'{self.path}/{f}'))
        
        self.files_jpg_dict = {}
        for f in self.files_jpg:
            self.files_jpg_dict[f] = self.get_pixels(cv2.imread(f'{self.path}/{f}'))

    def build_DF(self):
        self.build_a_dict()
        df_png = pd.DataFrame(self.files_png_dict)
        df_jpg = pd.DataFrame(self.files_jpg_dict)
        
        self.describe_df_png = df_png.describe().T.loc[:, ['mean','25%','50%','75%']]
        if self.files_jpg != []:
            self.describe_df_jpg = df_jpg.describe().T.loc[:, ['mean','25%','50%','75%']]

    def detect_duplicated_pics(self):
        self.build_DF()
        
        dup_png = self.describe_df_png[self.describe_df_png.duplicated(['25%','50%','75%'])].sort_values('mean')
        self.dup_png_pic = dup_png.index
        
        if self.files_jpg != []:
            dup_jpg = self.describe_df_jpg[self.describe_df_jpg.duplicated(['mean','25%','50%','75%'])].sort_values('mean')
            self.dup_jpg_pic = dup_jpg.index 
            print(f">>> There are {self.dup_png_pic.shape[0] + self.dup_jpg_pic.shape[0]} duplicated pictures been detected:")
            print(f"   {self.dup_png_pic.shape[0]} in [.png] format, {self.dup_jpg_pic.shape[0]} in [.jpg] format")
        
        else:
            print(f">>> There are {self.dup_png_pic.shape[0]} duplicated pictures been detected")
        
        print(f">>> If you are happy with the above details, you can run method: .removing_duplicated_pics()")
   
   
    def removing_duplicated_pics(self):
        self.dup_png_and_jpg = []
        for i in self.dup_png_pic:
            self.dup_png_and_jpg.append(i)
        if self.files_jpg != []:
            for i in self.dup_jpg_pic:
                self.dup_png_and_jpg.append(i)
   
        for p in os.listdir(f'{self.path}'): 
            if p in self.dup_png_and_jpg:
                os.remove(f'{self.path}/{p}')
        self.cleaned_files = []
        for (dirpath, dirnames, filenames) in walk(f'{self.path}'):
            self.cleaned_files.extend(filenames)
        

        print(f'>>> Cleaned Database now has reduced to {len(self.cleaned_files)} pictures')
        
    

