# Preprcessing data: process data: detect, alignment 
# Format image cropped: age_gender_id.png, ex: 02_1_1.png

# ````````````````````````````````````````````````````````````````````
import os 
# from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import scipy.io
import pandas as pd
import insightface
import numpy as np
import mxnet as mx



class Make_Dataset():
    def __ini__(self):
        pass



class Make_AAF_Dataset(Make_Dataset):
    pass


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
    f_names = os.listdir('data/processed/Wiki')
    lst = []
    for f_n in f_names:
        lst.append(int(f_n.split('.')[1].split('_')[-1]))
    
    # path2data_raw=os.path.join(os.getcwd() , 'data/raw/Wiki/wiki')
    # result_df = pd.read_csv(os.path.join(path2data_raw, 'result.csv'))

    # print(len(result_df))
    process_Wiki_dataset(continue_index=max(lst) + 1, margin=5)