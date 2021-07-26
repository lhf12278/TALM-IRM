

import os
import random
import shutil

prid_path = "G:\datasets\GRID"
prid_path_a = os.path.join(prid_path, 'single_shot/probe')
prid_path_b = os.path.join(prid_path, 'single_shot/gallery')
a_files = os.listdir(prid_path_a)
for i in range(20):
    cam_a_file = os.path.join(prid_path, 'split/cam_a_' + str(i))
    cam_b_file = os.path.join(prid_path, 'split/cam_b_' + str(i))
    train_file = os.path.join(prid_path, 'split/train_' + str(i))
    query_file = os.path.join(prid_path, 'split/query_' + str(i))
    gallery_file = os.path.join(prid_path, 'split/gallery_' + str(i))
    random.shuffle(a_files)
    train_a_files = a_files[0: 100]
    query_files = a_files[100: 200]
    train_b_files = []
    gallery_files = []
    for b_file in os.listdir(prid_path_b):
        if b_file in train_a_files:
            train_b_files.append(b_file)
        else:
            gallery_files.append(b_file)
    os.makedirs(cam_a_file)
    os.makedirs(cam_b_file)
    os.makedirs(query_file)
    os.makedirs(gallery_file)
    for train_a_file in train_a_files:
        shutil.copy(os.path.join(prid_path_a, train_a_file), cam_a_file)
    for train_b_file in train_b_files:
        shutil.copy(os.path.join(prid_path_b, train_b_file), cam_b_file)
    for query in query_files:
        shutil.copy(os.path.join(prid_path_a, query), query_file)
    for gallery in gallery_files:
        shutil.copy(os.path.join(prid_path_b, gallery), gallery_file)
    os.makedirs(train_file)
    for cam_a_train_file in os.listdir(cam_a_file):
        os.rename(os.path.join(cam_a_file, cam_a_train_file), os.path.join(cam_a_file, str(cam_a_train_file).split('.')[0]+'_'+str(0)+'.png'))
    for cam_b_train_file in os.listdir(cam_b_file):
        os.rename(os.path.join(cam_b_file, cam_b_train_file), os.path.join(cam_b_file, str(cam_b_train_file).split('.')[0]+'_'+str(1)+'.png'))
    for cam_a_query_file in os.listdir(query_file):
        os.rename(os.path.join(query_file, cam_a_query_file), os.path.join(query_file, str(cam_a_query_file).split('.')[0]+'_'+str(0)+'.png'))
    for cam_b_gallery_file in os.listdir(gallery_file):
        os.rename(os.path.join(gallery_file, cam_b_gallery_file), os.path.join(gallery_file, str(cam_b_gallery_file).split('.')[0]+'_'+str(1)+'.png'))
    for cam_a_train_file in os.listdir(cam_a_file):
        shutil.copy(os.path.join(cam_a_file, cam_a_train_file), train_file)
    for cam_b_train_file in os.listdir(cam_b_file):
        shutil.copy(os.path.join(cam_b_file, cam_b_train_file), train_file)




