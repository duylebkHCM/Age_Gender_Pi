# Preprcessing data: process data: detect, alignment 
# Format image cropped: age_gender_id.png, ex: 02_1_1.png

# ````````````````````````````````````````````````````````````````````
from ast import parse
import os
import cv2
from PIL import Image
import scipy.io
import pandas as pd
import imageio
import insightface
import pyprind
import numpy as np
import mxnet as mx
from tqdm import tqdm
import argparse
from mlxtend.image import EyepadAlign
import traceback

class Make_Dataset(object):
    def __init__(self, img_path, output_img, image_size, device, is_align = False, margin = 0, threshold = 0.5):
        self.img_path = img_path
        self.image_size = int(image_size)
        self.is_align = is_align
        self.margin = margin
        self.threshold = threshold
        self.bboxes = {}
        self.landmark = {}
        self.confidence = {}
        self.device = device
        self.output_img_dir = output_img
        self.aligned_dir = None
        self.cropped_dir = None

        if not os.path.isdir(os.path.join(output_img, 'aligned')):
            path = os.path.join(output_img, 'aligned')
            os.makedirs(path, exist_ok=True)
            self.aligned_dir = path
        else:
            self.aligned_dir = os.path.join(output_img, 'aligned')
        if not os.path.isdir(os.path.join(output_img, 'cropped')):
            path = os.path.join(output_img, 'cropped')
            os.makedirs(path, exist_ok=True)
            self.cropped_dir = path 
        else:
            self.cropped_dir = os.path.join(output_img, 'cropped')

        if self.is_align:
            lst_img_names = os.listdir(self.aligned_dir)
        else:
            lst_img_names = os.listdir(self.img_path)

        self.lst_img_paths = [os.path.join(img_path, i) for i in lst_img_names]

    def _face_align(self, target_size = 512):
        # Get average image
        eyepad = EyepadAlign(verbose=1)
        eyepad.fit_directory(target_img_dir=self.img_path,
                            target_width=target_size,
                            target_height=target_size,
                            file_extension='.jpg')  # note the capital letters

        # Center nose of the average image
        nose_coord = eyepad.target_landmarks_[33].copy()
        disp_vec = np.array([target_size//2, target_size//2]) - nose_coord
        translated_shape = eyepad.target_landmarks_ + disp_vec

        eyepad_centnoise = EyepadAlign(verbose=1)
        eyepad_centnoise.fit_values(target_landmarks=translated_shape,
                                    target_width=target_size,
                                    target_height=target_size)

        # Align images to centered average image
        flist = [f for f in os.listdir(self.img_path) if f.endswith('.jpg')]
        pbar = pyprind.ProgBar(len(flist), title='Aligning images ...')

        for f in flist:
            pbar.update()
            img = imageio.imread(os.path.join(self.img_path, f))

            img_tr = eyepad.transform(img)
            if img_tr is not None:
                imageio.imsave(os.path.join(self.aligned_dir, f), img_tr)

    def extract_face(self):
        #Aligng first
        if self.is_align:
            self._face_align()

        model = insightface.model_zoo.get_model('retinaface_r50_v1')
        model.prepare(ctx_id = int(self.device), nms=0.4)

        print('[INFO] Start extract face...')
        for idx in tqdm(range(len(self.lst_img_paths)), desc='Progress'):
            img_name = self.lst_img_paths[idx]
            try:
                img = cv2.imread(img_name)

                img = cv2.resize(img, (512, 512)) #Resize all images to the same size (512, 512, 3)

                bbox, landmark = model.detect(img, threshold=self.threshold, scale=1.0)

                #Get bbox and landmark of face which has the biggest area
                area = (bbox[:, 2] - bbox[:, 0])*(bbox[:, 3] - bbox[:, 1])
                choose_idx = np.argmax(area, axis=-1).ravel()
                choose_idx = int(choose_idx)    
                
                print('DEBUG choose idx', choose_idx)
                print('DEBUG bbox', bbox[choose_idx])
                
                landmark_new = np.reshape(landmark, (-1, 10), order='F')
                landmark_new = landmark_new.astype('int')
                self.bboxes[img_name.split('/')[-1].split('.')[0]] = bbox[choose_idx][:-1] / 512.0
                self.confidence[img_name.split('/')[-1].split('.')[0]] = bbox[choose_idx][-1]
                self.landmark[img_name.split('/')[-1].split('.')[0]] = landmark_new[choose_idx] / 512.0

                x1 = int(bbox[choose_idx][0]) - self.margin 
                y1 = int(bbox[choose_idx][1]) - self.margin 
                x2 = int(bbox[choose_idx][2]) + self.margin 
                y2 = int(bbox[choose_idx][3]) + self.margin

                crop_img = img[y1 : y2, x1 : x2]
                crop_img = cv2.resize(crop_img, (self.image_size, self.image_size))
                
                cv2.imwrite(os.path.join(self.cropped_dir, img_name.split('/')[-1]), crop_img)
            except Exception:
                traceback.print_exc()
                # print('Exception: ', e)
                continue
 
        print('[INFO] Finish extract face...')

    def create_csv(self, save_path):
        pass


class Make_AAF_Dataset(Make_Dataset):
    def __init__(self, label_path = '/home/Data/All_Age_Faces/raw/All-Age-Faces Dataset/image sets', *args, **kwargs):
        self.label = label_path
        super(Make_AAF_Dataset, self).__init__(*args, **kwargs)

    def create_csv(self, save_path):
        img_names = os.listdir(self.cropped_dir)
        
        for type in ['train', 'val']:
            output_dict = {}
            print(f'[INFO] start create {type}.csv...')
            with open(os.path.join(self.label, type + '.txt'), 'r') as txt:
                lines = txt.readlines()
                for line in tqdm(lines, desc='Progress'):
                    file_name = line.split(' ')[0]
                    gender = line.split(' ')[1]
                    try:
                        if file_name in img_names:
                            if 'file_name' not in output_dict.keys():
                                output_dict['file_name'] = [file_name]
                                output_dict['age'] = [int(file_name[file_name.rfind("A") + 1 : ]) if file_name[file_name.rfind("A") + 1 : ] else -1]
                                output_dict['gender'] = [int(gender)]
                                output_dict['x_min'] = [float(self.bboxes[file_name][0])]
                                output_dict['y_min'] = [float(self.bboxes[file_name][1])]
                                output_dict['x_max'] = [float(self.bboxes[file_name][2])]
                                output_dict['y_max'] = [float(self.bboxes[file_name][3])]
                                output_dict['land_mark'] = [str('[') + ','.join([str(i) for i in self.landmark[file_name]]) + str(']')]
                                output_dict['confidence'] = [float(self.confidence[file_name])]
                            else:
                                output_dict['file_name'].append(file_name)
                                output_dict['age'].append(int(file_name[file_name.rfind("A") + 1 : ].split('.')[0]) if file_name[file_name.rfind("A") + 1 : ] else -1)
                                output_dict['gender'].append(int(gender))
                                output_dict['x_min'].append(float(self.bboxes[file_name][0]))
                                output_dict['y_min'].append(float(self.bboxes[file_name][1]))
                                output_dict['x_max'].append(float(self.bboxes[file_name][2]))
                                output_dict['y_max'].append(float(self.bboxes[file_name][3]))
                                output_dict['land_mark'].append(str('[') + ','.join([str(i) for i in self.landmark[file_name]]) + str(']'))
                                output_dict['confidence'].append(float(self.confidence[file_name]))
                    except Exception:
                        # print('Exception: ', e)
                        traceback.print_exc()

            df = pd.DataFrame(output_dict)
            df.to_csv(os.path.join(save_path, type + '.csv'), index=False, header=True)
            print(f'[INFO] Finish create {type}.csv')

class Make_UTK_Dataset(Make_Dataset):
    def create_csv(self, save_path):
        num_row = len(os.listdir(self.cropped_dir))

        columns = ['file_name','age', 'gender', 'x_min', 'y_min', 'x_max', 'y_max', 'land_mark', 'confidence']
        img_names = os.listdir(self.cropped_dir)
        output_df = pd.DataFrame(index = range(num_row), columns=columns)
        print('[INFO] Start create csv...')
        for row in tqdm(range(num_row), desc='Progress'):
            output_df['file_name'][row] = img_names[row]
            output_df['age'][row] = int(img_names[row].split('_')[0]) if len(img_names[row].split('_')[0]) else -1
            output_df['gender'][row] = int(img_names[row].split('_')[1]) if len(img_names[row].split('_')[1]) else 2
            output_df['x_min'][row] = float(self.bboxes[img_names[row].split('.')[0]][0])
            output_df['y_min'][row] = float(self.bboxes[img_names[row].split('.')[0]][1])
            output_df['x_max'][row] = float(self.bboxes[img_names[row].split('.')[0]][2])
            output_df['y_max'][row] = float(self.bboxes[img_names[row].split('.')[0]][3])
            output_df['land_mark'][row] = str('[') + ','.join([str(i) for i in self.landmark[img_names[row].split('.')[0]]]) + str(']')
            output_df['confidence'][row] = float(self.bboxes[img_names[row].split('.')[0]][4])
        output_df.to_csv(os.path.join(save_path,'utk_face.csv'), index=False, header=True)
        print('[INFO] Finish create csv...')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--img-size', required=True, help="Size of crop images")
    ap.add_argument('--img-path', required=True, help="Path of input images")
    ap.add_argument('--output-img', required=True, help="Path of output img file")
    ap.add_argument('--output-csv', required=True, help="Path of csv file")
    ap.add_argument('--device', default='', help='Choose device to use (ie cpu or gpu 0,1,2,3)')
    ap.add_argument('--align', default=False, help='Align images or not')
    ap.add_argument('--threshold', default=0.5, help='Threshold used to detect face')

    opt = vars(ap.parse_args())        

    aaf = Make_AAF_Dataset(img_path = opt["img_path"], device = opt["device"], output_img = opt["output_img"], is_align=opt["align"], image_size = opt["img_size"], threshold=opt['threshold'])

    aaf.extract_face()
    aaf.create_csv(opt["output_csv"])   

python anhduy/age_gender_Pi/Age_Gender_Pi/src/data/make_dataset.py --img-path 'Data/All_Age_Faces/raw/All-Age-Faces Dataset/original images' --output-img Data/All_Age_Faces/interim --output-csv Data/All_Age_Faces/processed --img-size 128 --device '0'