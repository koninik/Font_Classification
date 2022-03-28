from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
#from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from torchvision.transforms.transforms import FiveCrop
#from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import argparse
#rom sampler import BalancedBatchSampler
from torch.optim.lr_scheduler import StepLR
import sklearn
from sklearn.model_selection import train_test_split
#from models.ResNet import resnet18
from efficientnet_pytorch import EfficientNet
#from torchsampler import ImbalancedDatasetSampler
import wandb



'''
def init_seed(opt):
    
    #Disable cudnn to maximize reproducibility
    
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

'''
#============== Dataset =========================

class DocumentDataset(nn.Module):

    def __init__(self, root, image_dir, csv_file, transform):
        self.root = root
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.dataset = pd.read_csv(csv_file) #.iloc[:, 1]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()
        #img_name = self.dataset.iloc[0]
        #label = self.dataset.iloc[1]
        image_name = os.path.join(self.image_dir, self.dataset.iloc[index, 0])
        #image_name = os.path.join(self.image_dir, self.image_files[index])
        
        image = Image.open(image_name)

        if image.mode != 'RGB': 
            image = image.convert('RGB')
        #else:
         #   image = image.convert('RGB')

        #label = self.dataset[index]
        label = self.dataset.iloc[index, 1]
        #print('image name', image_name, 'with label', label)
        if label == 'textura':
            label = 0
        elif label == 'rotunda':
            label = 1
        elif label == 'gotico_antiqua':
            label = 2
        elif label == 'bastarda':
            label = 3
        elif label == 'schwabacher':
            label = 4
        elif label == 'fraktur':
            label = 5
        elif label == 'antiqua':
            label = 6
        elif label == 'italic':
            label = 7
        elif label == 'greek':
            label = 8
        elif label == 'hebrew':
            label = 9

 
        if self.transform:
            image = self.transform(image)

        return (image, label)

class Model_Classifier(nn.Module):

    def __init__(self, model, num_classes, pretrained):
        super(Model_Classifier, self).__init__()
        self.num_classes = num_classes
        self.model = model
        #input image size: 224
        if self.model == 'resnet18':           
            self.model = models.resnet18(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
            #self.model = models.resnet101(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        if self.model == 'resnet50':           
            self.model = models.resnet50(pretrained=pretrained)
            #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
            num_ftrs = self.model.fc.in_features
            #self.model.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(num_ftrs, num_classes))
            self.model.fc = nn.Linear(num_ftrs, num_classes)


        if self.model == 'densenet':
            self.model = models.densenet201(pretrained=pretrained)
            num_ftrs = self.model.classifier.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        if self.model == 'efficientnet':
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
            num_ftrs = self.model._fc.in_features
            self.model._fc = nn.Linear(num_ftrs, num_classes)
        
        if self.model == 'vgg':
            self.model = models.vgg19_bn(pretrained)
            #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
            num_ftrs = self.model.classifier[-1].in_features 
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

        

            
    def forward(self, x):
        output = self.model(x)
        return output


#================ Performance and Loss Function ========================
def performance(pred, label):
    loss = nn.CrossEntropyLoss()
    loss = loss(pred, label)
    return loss 

#===================== Training ==========================================

def train_epoch(model, training_data, optimizer, device):
    '''Epoch operation in training phase'''
    
    model.train()
    total_loss = 0
    n_corrects = 0 
    total = 0
    for i, data in enumerate(tqdm(training_data)):
        #print('data', data)
        # prepare data
        image = data[0].to(device)
        #print('image', image, image.shape)
        label = data[1].to(device)
        #print('label', label, label.shape)

        optimizer.zero_grad()

        output = model(image)
        loss = performance(output, label)
        _, preds = torch.max(output.data, 1)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() #* image.size(0)
        total += label.size(0)
        n_corrects += (preds == label).sum().item()
        
    print('total is:', total, 'and training data length is:', len(training_data.dataset))
    
    #loss = total_loss/len(training_data.dataset)
    #accuracy = n_corrects/len(training_data.dataset)
    loss = total_loss/total
    accuracy = n_corrects/total
    
    return loss, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total = 0
    n_corrects = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_data)):

            image = data[0].to(device)
            label = data[1].to(device)

            output = model(image)
            loss = performance(output, label)  #performance
            _, preds = torch.max(output.data, 1)

            total_loss += loss.item()
            n_corrects += (preds == label.data).sum().item()
            total += label.size(0)
            
    
    #loss = total_loss/len(validation_data.dataset)
    #accuracy = n_corrects/len(validation_data.dataset)
    loss = total_loss/total
    accuracy = n_corrects/total

    return loss, accuracy


def train(model, training_data, validation_data, optimizer, scheduler, lr, device, run, args): #scheduler # after optimizer
    ''' Start training '''

    valid_accus = []
    num_of_no_improvement = 0
    best_acc = 0
    for epoch_i in range(args.epochs):
        print('[Epoch', epoch_i, ']')

        start = time.time()
        #wandb.log({'lr': scheduler.get_last_lr()})
        #print('Epoch:', epoch_i,'LR:', scheduler.get_last_lr())

        train_loss, train_acc = train_epoch(model, training_data, optimizer, device)
        print('Training: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, accu=100*train_acc,
                  elapse=(time.time()-start)/60))
        
        start = time.time()
        val_loss, val_acc = eval_epoch(model, validation_data, device)
        print('Validation: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=val_loss, accu=100*val_acc,
                  elapse=(time.time()-start)/60))

        scheduler.step()
        wandb.log({'epoch': epoch_i, 'train loss': train_loss, 'val loss': val_loss})
        wandb.log({'epoch': epoch_i, 'train acc': 100*train_acc, 'val acc': 100*val_acc})
        
        valid_accus += [val_acc]

        model_state_dict = model.state_dict()
        checkpoint = {'model': model_state_dict, 'settings': args, 'epoch': epoch_i}


        if val_acc > best_acc:
            model_name = 'best_model_task_1_bal.chkpt'
            torch.save(checkpoint, model_name)
            print('- [Info] The checkpoint file has been updated.')
            best_acc = val_acc
            torch.save(model.state_dict(), f"./trained_models/results_GAN/GAN_{args.model}_run_{run}.pth")
            num_of_no_improvement = 0
        else:
            num_of_no_improvement +=1


        if num_of_no_improvement >= args.stop:
            #with open("validation_accuracies_task_1.txt", "w") as output:
             #   output.write(str(valid_accus))

            print("Early stopping criteria met, stopping...")
            break

# ======================================================================================

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def prepare_dataloaders_random_split(image_dir, csv_file, batch_size, validation_fraction, balanced_batch):

    #========= Data augmentation and normalization for training =====#
    transform = transforms.Compose([
                        transforms.Resize(224),
                        #transforms.FiveCrop((224, 224)),
                        #transforms.CenterCrop((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        #transforms.RandomRotation(10, resample=Image.BILINEAR),
                        #transforms.RandomPerspective(distortion_scale=0.2),
                        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                        #transforms.RandomAffine(8, translate=(.15,.15)),
                        transforms.ToTensor()
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #transforms.Normalize((0.5,), (0.5,)),  #
                        ])
    
    #========= Preparing train and validation dataloaders =======#
    #for random set split uncomment this part
    '''
    df = pd.read_csv(csv_file)
    print(df.head())

    images = df['image']
    scripts = df['font']
    print(type(images))

    #images= images.as_matrix(columns=None)
    images= images.to_numpy()
    scripts = scripts.to_numpy()


    x_train, x_val, y_train, y_val = train_test_split(images, scripts, test_size=1256, random_state=4, stratify=scripts)
    '''
    dataset = DocumentDataset(Path(os.getcwd()), image_dir, 
                                    csv_file, transform=transform)


    #validation_length = int(len(dataset)*validation_fraction)
    validation_length = 1256
    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset)-validation_length, validation_length]
    )
    '''
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(train_img),torch.from_numpy(train_label))
        
    val_set = torch.utils.data.TensorDataset(torch.from_numpy(val_img),torch.from_numpy(val_label))
    '''
    print('length', len(train_set), len(val_set))
    

    if balanced_batch == True:
        print('Use of Balanced Batch Sampler')
        #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=BalancedBatchSampler(train_set)) # #)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #sampler=BalancedBatchSampler(train_set),
    # Build the validation loader using indices from 75000 to 80000
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    #val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, sampler=BalancedBatchSampler(val_set))


    print("Initializing Random Split Datasets and Dataloaders...")

    return train_loader, val_loader

    
def prepare_dataloaders(train_dir_or, train_dir_gan, val_dir, train_csv_or, train_csv_gan, val_csv, batch_size, balanced_batch):

    
    transform = transforms.Compose([
                        transforms.Resize(224),
                        #transforms.FiveCrop((224, 224)),
                        transforms.CenterCrop((224, 224)),
                        #transforms.RandomHorizontalFlip(),
                        #transforms.RandomRotation(10, resample=Image.BILINEAR),
                        #transforms.RandomPerspective(distortion_scale=0.2),
                        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                        #transforms.RandomAffine(8, translate=(.15,.15)),
                        transforms.ToTensor()
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #transforms.Normalize((0.5,), (0.5,)),  #
                        ])
    
    val_transform = transforms.Compose([
                        #transforms.Resize(224),
                        #transforms.FiveCrop((224, 224)),
                        #transforms.CenterCrop((224, 224)),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        #transforms.Normalize((0.5,), (0.5,)),  #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])


    #data_train_orig = datasets.ImageFolder(root = train_dir_or, transform=transform)
    #data_train_gan = datasets.ImageFolder(root = train_dir_gan, transform=transform)
    #concat_dataset = torch.utils.data.ConcatDataset([data_train_orig, data_train_gan])
    
    data_train_orig = DocumentDataset(Path(os.getcwd()), train_dir_or, 
                                    train_csv_or, transform=transform)
    data_train_gan = DocumentDataset(Path(os.getcwd()), train_dir_gan, 
                                    train_csv_gan, transform=transform)
    concat_dataset = torch.utils.data.ConcatDataset([data_train_orig, data_train_gan])
    val_set = DocumentDataset(Path(os.getcwd()), val_dir, 
                                    val_csv, transform=val_transform)
    
    if balanced_batch == True:
        print('Use of Balanced Batch Sampler')
        #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=BalancedBatchSampler(train_set)) # #)
    else:
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True)
    #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=BalancedBatchSampler(train_set))
    # Build the validation loader using indices from 75000 to 80000
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    print("Initializing Datasets and Dataloaders...")

    return train_loader, val_loader





#============================== Main =======================================

def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Document Classification')
    parser.add_argument('--model', type=str, default='densenet', help='type of cnn to use (resnet, densenet, etc.)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--num_classes', type=int, default=10, required=False, help='number of classes in the dataset')
    parser.add_argument('--feat_extract', type=bool, default=False, help='use of feature extractor or not')

    parser.add_argument('--image_dir', type=str, required=False, help='Path to image directory for random set split')
    parser.add_argument('--train_dir', type=str, required=False, help='Path to image directory for train set')
    parser.add_argument('--val_dir', type=str, required=False, help='Path to image directory for val set')
    parser.add_argument('--train_dir_gan', type=str, required=False, help='Path to gan image directory for train set')
    
    parser.add_argument('--csv_file', type=str, required=False, help='Path to csv directory for random set split')
    parser.add_argument('--train_csv', type=str, required=False, help='Path to csv directory for train set')
    parser.add_argument('--val_csv', type=str, required=False, help='Path to csv directory for val set')
    parser.add_argument('--train_csv_gan', type=str, required=False, help='Path to csv directory for train set')

    parser.add_argument('--epochs', type=int, default=60, help='epochs for training')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--stop', type=int, default=10, help='early stopping when validation does not improve for 10 epochs')

    parser.add_argument('-log', default=None)
    parser.add_argument('--task_name', type=str, default = 'task_1_font')
    parser.add_argument('--balanced_batch', type=bool, default=False)
    #parser.add_argument('-wandb.log', default=True)

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    
    #wandb.init(config=args)

    for run in range(1, 6):
        #runs = wandb.init(project=f"NEW_Font_Classification_+OpenGAN_{args.model}", reinit=True)
        runs = wandb.init(project=f"{args.model}_Font_Classification_GAN", reinit=True)
        wandb.config.update(args)
        print('Run', run)
        torch.manual_seed(run+12)
        #====Loading Dataset=====#
        if args.image_dir:
            train_data, val_data = prepare_dataloaders_random_split(args.image_dir, args.csv_file, args.batch_size, 0.1, args.balanced_batch)

        if args.train_dir:
            train_data, val_data = prepare_dataloaders(args.train_dir, args.train_dir_gan, args.val_dir, args.train_csv, args.train_csv_gan, args.val_csv, args.batch_size, args.balanced_batch)

        #====Preparing Model=====#

        # Detect if there is a GPU available
        device = torch.device('cuda:4' if args.cuda else 'cpu')
        #device = torch.device('cpu')
        print(device)
        #opt = main().parse_args()
        #init_seed(opt)
        model = Model_Classifier(args.model, args.num_classes, pretrained=True)
        #model = resnet18
        print(f"We use {args.model}")
        model = model.to(device)
        #print(model)
        wandb.watch(model)
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        #optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = StepLR(optimizer_ft, step_size=1, gamma=0.1)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'max', factor=0.1, patience=0)
        #scheduler = optim.lr_scheduler.StepLR(optimizer_ft, 'min', gamma=0.1)
        model= train(model, train_data, val_data, optimizer_ft, scheduler, args.lr, device, run, args)
        #model= train(model, train_data, val_data, optimizer_ft, device, args)    
        runs.finish()
if __name__ == '__main__':
    main()


### command lines ###

#doccreator
#python classifier_model_1_font_doccreator.py --batch_size 32 --num_classes 10 --train_dir /home/konnik/DAS_2022_paper/patches_224/ --train_dir_gan /home/konnik/DAS_2022_paper/doccreator_patches/ --train_csv /home/konnik/DAS_2022_paper/1000_patch_224-labels-training.csv --train_csv_gan /home/konnik/DAS_2022_paper/doccreator_labels.csv --val_dir /home/konnik/DAS_2022_paper/patches_224/ --val_csv /home/konnik/DAS_2022_paper/patch_224-labels-test.csv

#gan
#python classifier_model_1_font_doccreator.py --batch_size 32 --num_classes 10 --train_dir /home/konnik/DAS_2022_paper/patches_224/ --train_dir_gan /home/konnik/DAS_2022_paper/OpenGAN/output_gan_2/ --train_csv /home/konnik/DAS_2022_paper/1000_patch_224-labels-training.csv --train_csv_gan /home/konnik/DAS_2022_paper/opengan_new_labels.csv --val_dir /home/konnik/DAS_2022_paper/patches_224/ --val_csv /home/konnik/DAS_2022_paper/patch_224-labels-test.csv

#opengan_comb
#python classifier_model_1_font_doccreator.py --batch_size 32 --num_classes 10 --train_dir /home/konnik/DAS_2022_paper/patches_224/ --train_dir_gan /home/konnik/DAS_2022_paper/OpenGAN/output_gan_comb/ --train_csv /home/konnik/DAS_2022_paper/1000_patch_224-labels-training.csv --train_csv_gan /home/konnik/DAS_2022_paper/opengan_comb_labels.csv --val_dir /home/konnik/DAS_2022_paper/patches_224/ --val_csv /home/konnik/DAS_2022_paper/patch_224-labels-test.csv

#real
##python classifier_model_1_font_doccreator.py --batch_size 32 --num_classes 10 --train_dir /home/konnik/DAS_2022_paper/patches_224/ --train_dir_gan /home/konnik/DAS_2022_paper/patches_224/ --train_csv /home/konnik/DAS_2022_paper/1000_patch_224-labels-training.csv --train_csv_gan /home/konnik/DAS_2022_paper/60,000_patch_224-labels-training_new.csv --val_dir /home/konnik/DAS_2022_paper/patches_224/ --val_csv /home/konnik/DAS_2022_paper/patch_224-labels-test.csv
#========================== command lines ======================
