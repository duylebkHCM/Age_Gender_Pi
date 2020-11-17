# Preprcessing data: process data: detect, alignment 
# Format image cropped: age_gender_id.png, ex: 02_1_1.png

# ````````````````````````````````````````````````````````````````````
from ast import parse
import os 
import cv2
from PIL import Image
import scipy.io
import pandas as pd
import insightface
import numpy as np
import mxnet as mx
import argparse

class Make_Dataset():
    def __ini__(self, img_path, output_path, image_size, device, is_align = False, margin = 0, threshold = 0.5):
        self.img_path = img_path
        self.image_size = image_size

        lst_img_names = os.listdir(self.img_path)
        self.is_align = is_align
        self.margin = margin
        self.threshold = threshold
        self.landmark = {}
        self.output_img_dir = output_path
        self.device = device
        self.lst_img_paths = [os.path.join(img_path, i) for i in lst_img_names]

    def extract_face(self):
        model = insightface.model_zoo.get_model('retinaface_r50_v1')
        model.prepare(ctx_id = int(self.device), nms=0.4)

        for img_name in self.lst_img_paths:
            img = cv2.imread(img_name)

            bbox, landmark = model.detect(img, threshold=self.threshold, scale=1.0)

            landmark_new = np.reshape(landmark, (-1, 10), order='F')
            landmark_new = landmark_new.astype('int')

            self.landmark[img_name.split('/')[-1].split('.')[0]] = landmark_new

            x1 = int(bbox[0][0]) - self.margin
            y1 = int(bbox[0][1]) - self.margin
            x2 = int(bbox[0][2]) + self.margin
            y2 = int(bbox[0][3]) + self.margin
            con = bbox[0][4]

            crop_img = img[y1 : y2, x1 : x2]
            crop_img = cv2.resize(crop_img, (self.image_size, self.image_size))

            if self.is_align:
                pass
            
            cv2.imwrite(os.path.join(self.output_img_dir, img_name.split('/')[-1]), crop_img)

    def create_csv(self, save_path):
        
        pass


class Make_AAF_Dataset(Make_Dataset):
    def __init__(self):
        super(Make_AAF_Dataset, self).__init__()

    def create_csv(self, save_path):
        pass

class Make_UTK_Dataset(Make_Dataset):
    def __init__(self):
        super(Make_UTK_Dataset, self).__init__()

    def create_csv(self, save_path):
        num_row = len(os.listdir(self.output_img_dir))

        columns = ['file_name','age', 'gender', 'land_mark']
        img_names = os.listdir(self.output_img_dir)
        output_df = pd.DataFrame(index = range(num_row), columns=columns)
        for row in range(num_row):
            output_df[row]['file_name'] = img_names[row]
            output_df[row]['age'] = img_names[row].split('_')[0]
            output_df[row]['gender'] = img_names[row].split('_')[1]
            output_df[row]['land_mark'] = self.landmark[img_names[row].split('.')[0]]
        output_df.to_csv(save_path)

def process_Wiki_dataset(continue_index, image_size = 128, is_align = False, margin = 0, threshold = 0.5):
    path = os.getcwd() #/content/drive/My Drive/Colab Notebooks/Face_Application/Face-Application
    path2data_raw=os.path.join(path , 'data/raw/Wiki/wiki')
    path2data_processed = os.path.join(path , 'data/processed/Wiki')
    path2meta = os.path.join(path, 'data/raw/Wiki/wiki/meta.csv')
    path2resultdf = os.path.join(path, 'data/raw/Wiki/wiki/result.csv')

    df = pd.read_csv(path2meta)
    df.columns = ['dob', 'photo_taken', 'full_path', 'gender', 'face_loc', 'face_score', 'second_face_score', 'yob', 'age']

    if not os.path.isfile(os.path.join(path2data_raw, 'result.csv')):
        result_df = pd.DataFrame(columns = ['my_bbox', 'landmarks', 'detect_con', 'save_path'], index=range(len(df.index)))
    else:
        result_df = pd.read_csv(os.path.join(path2data_raw, 'result.csv'))
        
    print('[INFO] start process')

    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id = 0, nms=0.4)
    
    for idx, row in df.loc[continue_index : ].iterrows():
        img_path = row['full_path']
        img_path = os.path.join(path2data_raw, img_path)
        print(f'[INFO] {idx}-img_path:{img_path}')

        age = row['age']
        gender = row['gender']
        
        img = cv2.imread(img_path)

        if str(img) == 'None':
            result_df['my_bbox'][idx] = None
            result_df['landmarks'][idx] = None
            result_df['detect_con'][idx] = None
            result_df['save_path'][idx] = None
        else:
            try:
                bbox, landmark = model.detect(img, threshold=threshold, scale=1.0)

                landmark_new = np.reshape(landmark, (-1, 10), order='F')
                landmark_new = landmark_new.astype('int')

                x1 = int(bbox[0][0]) - margin
                y1 = int(bbox[0][1]) - margin
                x2 = int(bbox[0][2]) + margin
                y2 = int(bbox[0][3]) + margin
                con = bbox[0][4]

                crop_img = img[y1 : y2, x1 : x2]
                crop_img = cv2.resize(crop_img, (image_size, image_size))

                if is_align:
                    pass

                img_name = str(gender) + '_' + str(age) + '_' + str(idx) + '.jpg'
                save_path = os.path.join(path2data_processed, img_name)

                cv2.imwrite(save_path, crop_img)

                result_df['my_bbox'][idx] = ','.join([str(i) for i in bbox[0][:-1]])
                result_df['landmarks'][idx] = ','.join([str(i) for i in landmark_new[0]])
                result_df['detect_con'][idx] = con
                result_df['save_path'][idx] = save_path
            except:
                result_df['my_bbox'][idx] = None
                result_df['landmarks'][idx] = None
                result_df['detect_con'][idx] = None
                result_df['save_path'][idx] = None
            
        result_df.to_csv(path2resultdf, index=False, header=True)

    print('[INFO] Finish processed')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--img-size', required=True, help="Size of crop images")
    ap.add_argument('--img-path', required=True, help="Path of input images")
    ap.add_argument('--output-img', required=True, help="Path of csv file")
    ap.add_argument('--output-path', required=True, help="Path of csv file")
    ap.add_argument('--device', default='', help='Choose device to use')

    opt = vars(ap.parse_args())        

    utk = Make_UTK_Dataset(img_path = opt["img_path"], device = opt["device"], output_path = opt["output_img"], image_size = opt["img_size"])

    utk.extract_face()
    utk.create_csv(opt["output_path"])  