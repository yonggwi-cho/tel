import numpy as np
import codecs
from PIL import Image, ImageOps
import os
import torch.utils.data as data

class PreProcessor():
    def __init__(self,batch_size,path):
        #self.n_sample = n_sample
        self.batch_size = batch_size
        self.path = path

    def read_file(self):
        data_list = []
        for top_dir in os.listdir(self.path):
            file_dir = os.path.join(self.path, top_dir)
            file_list = os.listdir(file_dir)
            self.n_data = len(file_list)
            data_list += [os.path.join(self.path, top_dir, file) for file in file_list[:]]
        return data_list

    def pre_process(self):
        pass

class DatasetCreater(data.Dataset):
    def __init__(self,batch_size,path):
        self.batch_size = batch_size
        self.path = path

    def count_files(self):
        self.files_path = list()
        for top_dir in os.listdir(self.path):
            file_dir = os.path.join(self.path, top_dir)
            file_list = os.listdir(file_dir)
            self.n_data = len(file_list)
            self.files_path += [os.path.join(self.path, top_dir, file) for file in file_list[:]]

    def read_file(self):
        pass

    def pre_process(self):
        pass
