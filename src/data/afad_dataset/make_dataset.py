import pandas as pd
import numpy as np
import os
from glob import glob
import cv2
from tqdm import tqdm
import shutil


BASE_DIR = '/home/Data/AFAD_Dataset/raw/tarball/AFAD-Full'
OUTPUT_DIR = '/home/Data/AFAD_Dataset/interim/'
OUTPUT_CSV = '/home/Data/AFAD_Dataset/processed/'

def analysis(path):
    img_heights = []
    img_widths = []
    for image_name in glob(path + '/*/*/*.jpg'):
        try:
            image = cv2.imread(image_name)
            img_size = image.shape[:2]
        except Exception as e:
            img_heights.append(0)
            img_widths.append(0)
        else:
            img_heights.append(img_size[0])
            img_widths.append(img_size[1])
    
    avg_height = sum(img_heights) / len(img_heights)
    avg_width = sum(img_widths) / len(img_widths)

    return avg_height, avg_width

def create_dataset(path):
    new_h, new_w = analysis(path)

    print(f'[INFO] Resize w, h are: {new_w}, {new_h}')
    
    output_dict = {}

    with open(os.path.join(OUTPUT_DIR, 'processed.log'), 'w') as f:
        f.write('#############################Error images#########################\n')
        f.flush()

    print('[INFO] Start processed dataset raw')

    count = 0
    for image_path in tqdm(glob(path  + '/*/*/*.jpg'), desc='Progress'):
        info = image_path.split('/')
        file_name = str(info[-1])
        age = int(info[-3])
        gender = 1 if int(info[-2]) == 111 else 0 # 1:Male , 0: Female

        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (int(new_h), int(new_w)))
        except Exception as e:
            with open(os.path.join(OUTPUT_DIR, 'processed.log'), 'a') as f:
                f.write('%s\t', str(e))
                f.write('%s\n', str(image_path))
        else:
            if 'age' not in output_dict:
                output_dict['age'] = [age]
                output_dict['gender'] = [gender]
                output_dict['file_name'] = [file_name]
            else:
                output_dict['age'].append(age)
                output_dict['gender'].append(gender)
                output_dict['file_name'].append(file_name)

            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), img)
            count += 1
    
    df = pd.DataFrame(output_dict)

    df.to_csv(os.path.join(OUTPUT_CSV, 'afad.csv'), header=True, index=False)

    f.close()

    print('[INFO] Finish')
    print('[INFO] Total of success processed image : ', count)

if __name__ == '__main__':
    create_dataset(BASE_DIR)