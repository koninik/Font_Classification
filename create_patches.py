from os import path
import torch
from PIL import Image
import os
import torch.nn.functional as F
import torch.nn as nn
from numpy import asarray
import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import csv
import argparse
import random


crop_list = []
label_list = []


def create_patches(image_dir, csv_file, patch_size, output_dir, output_csv):
    transt = transforms.ToTensor()
    transp = transforms.ToPILImage()

    dataset_csv = pd.read_csv(csv_file)
    dataset_csv.index = dataset_csv["image"]
    print('length of dataset', len(dataset_csv))
    
    i=0
    height_list = []
    width_list = []
    for filename in os.listdir(image_dir):#dataset_csv.index.values:
        print(filename)
        
        i+=1
        if i % 100 == 0:
            print(f'{i} images processed') 
        
        image_name = os.path.join(image_dir, filename)
        
        label = dataset_csv.iloc[dataset_csv.index.get_loc(filename), 1]  
        #print('label', label)
        image = Image.open(image_name).convert('RGB')
        #print('size of image', image.size)
        w, h = image.size
        aspect_ratio = w / h
        if w < patch_size:
            wpercent = (patch_size / float(w))
            new_height = int((float(h) * float(wpercent)))
            image = image.resize((patch_size, new_height))
            #print('NEW WIDTH image size', image.size)
            w, h = image.size
            #patch_size = w
            #print('NEW patch size', patch_size)
            
        if h < patch_size:
            hpercent = (patch_size / float(h))
            new_w = int((float(w) * float(hpercent)))
            image = image.resize((new_w, patch_size))
            print('NEW HEIGHT image size', image.size)
            #patch_size = h
            #print('NEW patch size', patch_size)
        

        img_t = transt(image)
        kernel_height, kernel_width = patch_size, patch_size
        stride_height, stride_width = patch_size, patch_size
        patches = img_t.data.unfold(0, 3, 3).unfold(1, kernel_height, stride_height).unfold(2, kernel_width, stride_width) #first unfold is the channels, then height, then width
        
        height_list.append(patches.shape[1])
        width_list.append(patches.shape[2])
        save_patches(patches, filename, label, output_dir, output_csv, patches.shape[1], patches.shape[2])
        #print(patches[0][0][0])
        '''
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_t = transt(image)
        except ValueError:
                continue
        try:
            patches = img_t.data.unfold(0, 3, 3).unfold(1, 224, 224).unfold(2, 224, 224)
            #newsize = (224, 224)
            #patches = patches.resize(newsize)
            #save_patches(patches, filename, label)
        except RuntimeError:
                continue
        '''
    print('MAX HEIGHT IN LIST', max(height_list)) 
    print('MAX WIDTH IN LIST', max(width_list)) 
    print('Finished patching')

def save_patches(patches, filename, label, output_dir, csv_file, i_value, j_value):
    """Imshow for Tensor."""  
    transp = transforms.ToPILImage()
    #print(i_value, j_value)
    #try:
    for i in range(0, i_value):
        
        for j in range(0, j_value):
            
            inp = transp(patches[0][i][j])
            crop_name = os.path.splitext(filename)[0]
            #print(label)
            ext = os.path.splitext(filename)[1]
            inp.save(f"{output_dir}/{crop_name}_{i}_{j}.png")
            crop_list.append(f'{crop_name}_{i}_{j}.png')
            '''
            if ext == '.jpg':
                inp.save(f"{output_dir}/{crop_name}_{i}_{j}.jpg")
                crop_list.append(f'{crop_name}_{i}_{j}.jpg')
            
            elif ext == '.JPEG':
                inp.save(f"{output_dir}/{crop_name}_{i}_{j}.JPEG")
                crop_list.append(f'{crop_name}_{i}_{j}.JPEG')

            elif ext == '.jpeg':
                inp.save(f"{output_dir}/{crop_name}_{i}_{j}.jpeg")
                crop_list.append(f'{crop_name}_{i}_{j}.jpeg')

            elif ext == '.tif':
                inp.save(f"{output_dir}/{crop_name}_{i}_{j}.tif")
                crop_list.append(f'{crop_name}_{i}_{j}.tif')

            elif ext == '.png':
                inp.save(f"{output_dir}/{crop_name}_{i}_{j}.png")
                crop_list.append(f'{crop_name}_{i}_{j}.png')
            '''
            label_list.append(label)
                
            #except IndexError:
             #   continue
    '''
    except ValueError:
        for i in random.sample(range(0, i_value), 1):
            #print(i)
            for j in random.sample(range(0, j_value), 1):
             #   print(j)
                try:
                    inp = transp(patches[0][i][j])
                    crop_name = os.path.splitext(filename)[0]
                    #print(label)
                    ext = os.path.splitext(filename)[1]
                    inp.save(f"{output_dir}/{crop_name}_{i}_{j}.png")
                    crop_list.append(f'{crop_name}_{i}_{j}.png')
                    
                    if ext == '.jpg':
                        inp.save(f"{output_dir}/{crop_name}_{i}_{j}.jpg")
                        crop_list.append(f'{crop_name}_{i}_{j}.jpg')
                    
                    elif ext == '.JPEG':
                        inp.save(f"{output_dir}/{crop_name}_{i}_{j}.JPEG")
                        crop_list.append(f'{crop_name}_{i}_{j}.JPEG')

                    elif ext == '.jpeg':
                        inp.save(f"{output_dir}/{crop_name}_{i}_{j}.jpeg")
                        crop_list.append(f'{crop_name}_{i}_{j}.jpeg')

                    elif ext == '.tif':
                        inp.save(f"{output_dir}/{crop_name}_{i}_{j}.tif")
                        crop_list.append(f'{crop_name}_{i}_{j}.tif')

                    elif ext == '.png':
                        inp.save(f"{output_dir}/{crop_name}_{i}_{j}.png")
                        crop_list.append(f'{crop_name}_{i}_{j}.png')
                    
                    label_list.append(label)
                    
                except IndexError:
                    continue
    '''
    #print('Length of crop list', len(crop_list))
    #print('Length of label list', len(label_list))
    #print('===================================')
    #print(label_list)
    '''
    file = open("location_crops_new.csv", "w")
    writer = csv.writer(file)
    for w in range(len(label_list)):
        writer.writerow([crop_list[w], label_list[w]])
    file.close()
    '''
    with open(csv_file, 'w', newline='') as f:
        for w in range(len(label_list)):
            writer = csv.writer(f)
            #print(label_list[w])
            writer.writerow([crop_list[w], label_list[w]])
    f.close()
    
        
def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Create patches from images and save them with their labels for classification task')

    parser.add_argument('--create_patches', type=bool, default=True, help='argument True if want to create patches')
    parser.add_argument('--save_patches', type=bool, default=False, help='argument True if want to save the generated patch images and their labels')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of patches we want to create')
    parser.add_argument('--image_dir', type=str, default='/path/to/images/', help='Directory of Input Images')
    parser.add_argument('--csv_file', type=str, default='/path/to/gt.csv', help='Directory to csv file that contains image names and labels')
    parser.add_argument('--output_csv', type=str, default='/path/to/saved/patches_gt.csv', help='Directory to output csv file that contains patch image names and labels')
    parser.add_argument('--output_dir', type=str, default='/path/to/saved/patches/images/', help='Directory to save patch images')

    args = parser.parse_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.create_patches:
        print("STARTED patches")
        create_patches(args.image_dir, args.csv_file, args.patch_size, output_dir, args.output_csv)
        print("FINISHED patches")
        
if __name__ == '__main__':
    main()