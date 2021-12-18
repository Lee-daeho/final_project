import os
import cv2
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from config import *

import matplotlib.pyplot as plt

def makedirs(path): 
    try: 
        os.makedirs(path) 
    except OSError: 
        if not os.path.isdir(path): 
            raise

def labeled_savecam(data_train, labeled_graycam, labeled_set, trial, cycle, NO_CLASSES):
    
    random.seed(25)
    
    random_indices = random.sample(range(ADDENDUM), 10) ### 

    for idx in random_indices:
        img = data_train.data[labeled_set[cycle*ADDENDUM + idx]]
        
        path = "./figure/CAM_result/Trial_"+str(trial)+"/Cycle_"+str(cycle)+"/labeled/"+str(labeled_set[cycle*ADDENDUM + idx])
        makedirs(path)
        plt.title(" ")
        plt.imshow(img)
        plt.savefig(path+"/original.png")
        
        for clas in range(NO_CLASSES):
            heatmap = labeled_graycam[idx, clas]
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_PINK)
            heatmap_img = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
            plt.imshow(heatmap_img)
            plt.savefig(path+"/{}.png".format(clas))
            

def unlabeled_savecam(data_train, unlabeled_graycam, subset, trial, cycle, NO_CLASSES, arg):
    
    random.seed(25)
#     random_indices = random.sample(range(len(subset)), 10)

    top_set = list(torch.tensor(subset)[arg][-ADDENDUM:].numpy()[:10])
    worst_set = list(torch.tensor(subset)[arg][-ADDENDUM:].numpy()[-10:])

    for i, idx in enumerate(top_set):
        img = data_train.data[idx]
        
        path = "./figure/CAM_result/Trial_"+str(trial)+"/Cycle_"+str(cycle)+"/unlabeled/top_set/"+str(i)
        makedirs(path)
        plt.title(" ")
        plt.imshow(img)
        plt.savefig(path+"/original.png")
        
        for clas in range(NO_CLASSES):
            heatmap = unlabeled_graycam[subset.index(idx), clas]
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_PINK)
            heatmap_img = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
            plt.imshow(heatmap_img)
            plt.savefig(path+"/{}.png".format(clas))
    
    for i, idx in enumerate(worst_set):
        img = data_train.data[idx]
        
        path = "./figure/CAM_result/Trial_"+str(trial)+"/Cycle_"+str(cycle)+"/unlabeled/worst_set/"+str(i)
        makedirs(path)
        plt.title(" ")
        plt.imshow(img)
        plt.savefig(path+"/original.png")
        
        for clas in range(NO_CLASSES):
            heatmap = unlabeled_graycam[subset.index(idx), clas]
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_PINK)
            heatmap_img = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
            plt.imshow(heatmap_img)
            plt.savefig(path+"/{}.png".format(clas))
            
def top_worst(arg, data_train, subset, method, models, trial, cycle):
    
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], 
                    [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])
    
    top_set = list(torch.tensor(subset)[arg][-ADDENDUM:].numpy()[:50])
    worst_set = list(torch.tensor(subset)[arg][-ADDENDUM:].numpy()[-50:])

    top_data = data_train.data[top_set] # 원본 이미지
    worst_data = data_train.data[worst_set]

    top_transf_data = torch.zeros([50, 3, 32, 32], device='cuda')
    worst_transf_data = torch.zeros([50, 3, 32, 32], device='cuda')

    top_true = np.array(data_train.targets)[top_set]
    worst_true = np.array(data_train.targets)[worst_set]

    for i in range(len(top_set)):
        top_transf_data[i] = test_transform(top_data[i])
        worst_transf_data[i] = test_transform(worst_data[i])

    with torch.no_grad():
        top_scores, _, _ = models['backbone'](top_transf_data)
        worst_scores, _, _ = models['backbone'](worst_transf_data)

        _, top_preds = torch.max(top_scores.data, 1)
        _, worst_preds = torch.max(worst_scores.data, 1)

    for i, data in enumerate(top_data):
        title = "Rank"+str(i)+"_"+"GT"+str(top_true[i])+"_"+"PT"+str(top_preds[i].item())
        plt.title(title)
        plt.imshow(data)
        
        path = "./figure/"+str(method)+"/top"+"/Trial_"+str(trial)+"/Cycle_"+str(cycle)
        makedirs(path)
        plt.savefig(path+"/{}.png".format(title))
    
    for i, data in enumerate(worst_data):
        title = "Rank"+str(i)+"_"+"GT"+str(worst_true[i])+"_"+"PT"+str(worst_preds[i].item())
        plt.title(title)
        plt.imshow(data)
        
        path = "./figure/"+str(method)+"/worst"+"/Trial_"+str(trial)+"/Cycle_"+str(cycle)
        makedirs(path)
        plt.savefig(path+"/{}.png".format(title))