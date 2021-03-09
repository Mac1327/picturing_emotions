import os
import random
import shutil
from shutil import copyfile

classes = ['surprise',
            "fear",
            "disgust",
            "anger",
            "neutrality",
            "sadness",
            "happiness"]
data_location = "raw_data/fer_ckplus_kdef"


def list_and_count(main_data_folder="raw_data/data_3200"):

    check_dr = [main_data_folder+"/train/", main_data_folder+"/val/", main_data_folder+"/test/"]
    for dr in check_dr:
        print(f"##########{dr}###########")
        for dir_ in os.listdir(dr):
            count = 0
            for f in os.listdir(f"{dr}{dir_}/"):
                count += 1
            print(f"{dir_}:   \t {count} images")


def make_emotions_dir(location_to_make_dirs, name_of_dir):
    root = location_to_make_dirs
    os.mkdir(root+f'/{name_of_dir}')
    os.mkdir(root+f'/{name_of_dir}/train/')
    os.mkdir(root+f'/{name_of_dir}/test/')
    os.mkdir(root+f'/{name_of_dir}/val/')

    for class_ in classes:
        os.mkdir(root+f'/{name_of_dir}/train/{class_}')
        os.mkdir(root+f'/{name_of_dir}/test/{class_}')
        os.mkdir(root+f'/{name_of_dir}/val/{class_}')


def split_data(data_dir, train, val, test, train_size, subsample=True, samplesize=3200):
    """
    Inputs:The current data dir for a given class, the destination train, val and test dir. 
           PLus the ratio of train size.
    Output: Copies of the the images in the correct destination file locations.
    """
    #make list of of file names
    current_data = []
    for item in os.listdir(data_dir):
        if os.path.getsize(f"{data_dir}/{item}") >0:
            current_data.append(item)
        else:
            print(item, "Source is empty!")
    #create suffeled sample
    if subsample:
        shuff = random.sample(current_data, samplesize)
    else:
        shuff = random.sample(current_data, len(current_data))
    
    #split names into train, val and test
    size = int(len(shuff)*train_size)
    train_set = shuff[:size]
    val_test_set = shuff[size:]
    
    val_test_size = int(len(val_test_set) /2)
    val_set = val_test_set[:val_test_size]
    test_set = val_test_set[val_test_size:]
    
    #copy files into correct locations
    for item in train_set:
        copyfile(data_dir + item, train + item)
    for item in val_set:
        copyfile(data_dir + item, val + item)
    for item in test_set:
        copyfile(data_dir + item, test + item)


def make_dir_split_sample(location_to_make_dirs="raw_data", name_of_dir="data_3200", 
                           raw_data_location=data_location,  train_size=0.9, samplesize=3200):


    make_emotions_dir(location_to_make_dirs, name_of_dir)

    for class_ in classes:

        data_dir = f'{raw_data_location}/{class_}/'
        train_dir = f"{location_to_make_dirs}/{name_of_dir}/train/{class_}/"
        val_dir =  f"{location_to_make_dirs}/{name_of_dir}/val/{class_}/"
        test_dir =  f"{location_to_make_dirs}/{name_of_dir}/test/{class_}/"
        split_data(data_dir, train_dir, val_dir, test_dir, train_size=train_size, samplesize=samplesize)
    print(list_and_count(f"{location_to_make_dirs}/{name_of_dir}"))



if __name__ == "__main__":
    print("test")
    raw_data_location = "raw_data/fer_ckplus_kdef"
    location_to_make_dirs="raw_data"
    name_of_dir= "data_3200_improved"
    sample_size = 3200
    make_dir_split_sample(location_to_make_dirs=location_to_make_dirs, name_of_dir=name_of_dir, 
                           raw_data_location=raw_data_location,  train_size=0.9, samplesize=sample_size)