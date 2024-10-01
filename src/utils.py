#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:43:29 2018

@author: lepetit
"""

import torch
import os
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import torch.nn as nn
import re
import json

########################################################################
###################### useful for siamese learning #####################
########################################################################

def label_to_sgn(label):  #0 -> 1  et 1 -> -1 
    sgn =copy.deepcopy(label)
    sgn.detach()
    sgn[label==0] = 1
    sgn[label==1] = -1
    return sgn

def loss_from_diff(a,b,x):
    m=(a+b)/2
    r= (b-a)/2
    y = torch.abs(x -m) - r
    y = torch.clamp(y,min = 0)
    y = torch.mean(y)
    return y


def loss_from_pair(a,b,x,y):
    m=(a+b)/2
    r= (b-a)/2
    z = torch.abs(x - y - m) - r
    z = torch.clamp(z , min = 0) + torch.clamp(-x , min = 0) + torch.clamp(-y , min = 0)
    z = torch.mean(z)
    return z


######################################################################
######################################################################
######################################################################

def jpg_to_json(name):
    return name[:-4] + '.json'

def json_to_jpg(name):
    return name[:-5] + '.jpg'

def lbls_in_jpg(lbls):
    lbls_jpg = {}
    for name in lbls:
        lbls_jpg[json_to_jpg(name)] = lbls[name]
    return lbls_jpg


def mkdir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

def correct_lbls(lbls):
    names = sorted(lbls.keys())
        
    for i in range(1,len(names)):
        labels = lbls[names[i]]
        prev_labels = lbls[names[i-1]]
        if (labels['ground'] in ['wet_road','dry_road']) and (labels['old snow_traces'] != 'ground') and (labels['compa'] == 'snow_up'):
#            print('yo')
            lbls[names[i]]['compa'] = 'no_comp'
        if (prev_labels['ground'] in ['wet_road','dry_road']) and (prev_labels['old snow_traces'] != 'ground') and (labels['compa'] == 'snow_down'):
#            print('yo')
            lbls[names[i]]['compa'] = 'no_comp'
   

        
def get_lbls(labels_dir,c = True):
    lbls = {}
    
    for name in os.listdir(labels_dir):
    #    try:
            
            lbl_fic = os.path.join(labels_dir, name)
    #        cam, amd, hm = re.split("_", name)
    #        date = datetime.strptime(amd+hm[:-5],"%Y%m%d%H%M%S")
    
            lbls[name] = {}
            with open(lbl_fic) as x:
                lbl = json.load(x)
            
            for key in lbl.keys():
                lbls[name][key] = lbl[key]
    
    if c:            
        correct_lbls(lbls)
                
    return lbls


def get_weigths(classes, labels_train_dir, in_cuda=True):
    weights = {}
    lbls = get_lbls(labels_train_dir)
    
    names = sorted(lbls.keys())
    
    #calcul des poids:            
    for attribute in classes:
        list_of_weights = []
        for list_of_labels in classes[attribute]:
            weight = 1 / len([name for name in names if lbls[name][attribute] in list_of_labels])
            list_of_weights.append(weight)
        #normalisation:
        x = np.array(list_of_weights)
        x = np.round(x/np.sum(x),3)
        list_of_weights = torch.tensor(x)
        if in_cuda:
            list_of_weights = list_of_weights.float().cuda()
        #affectation
        weights[attribute]=list_of_weights
    return weights



def get_dic_sampling_weights(classes, labels_train_dir, in_cuda=True):
    lbls = get_lbls(labels_train_dir)
    
    for name in os.listdir(labels_train_dir):
    #    try:
            
            lbl_fic = os.path.join(labels_train_dir, name)
    #        cam, amd, hm = re.split("_", name)
    #        date = datetime.strptime(amd+hm[:-5],"%Y%m%d%H%M%S")
    
            lbls[name] = {}
            with open(lbl_fic) as x:
                lbl = json.load(x)
            
            for key in lbl.keys():
                lbls[name][key] = lbl[key]
    
    names= sorted(lbls.keys())
    
    #calcul des poids:
    dic_sampling_weights={}            
    for attribute in classes:
        
        weights = np.zeros(len(names))
        
        set_of_weights = []
        
        for labels in classes[attribute]:  #indictrice de la classe * taille de la classe
            weights_labels = np.zeros(len(names))
            for i in range(len(names)):
                if lbls[names[i]][attribute] in labels:
                    weights_labels[i] = 1
            weight = 1/np.sum(weights_labels) 
            weights_labels = weight * weights_labels
            #update
            set_of_weights.append(weight)
            weights += weights_labels
        
        weights = np.round((1/sum(set_of_weights)) * weights,2)
        
        dic_sampling_weights[attribute] = list(weights)

    return dic_sampling_weights

def get_sampling_weights(classes, labels_dir): # "avec le min"
    #for 'train':
    ltrain, lval = (len(os.listdir(labels_dir[phase])) for phase in ['train','val'])
    weights = {'train': np.ones(ltrain), 'val':np.ones(lval)}
    
    for phase in ['train','val']:
        for attribute in classes:
            dics_sampling_weights = get_dic_sampling_weights(classes, labels_dir[phase]) 
            attribute_weights = np.array(dics_sampling_weights[attribute])
            weights[phase] = np.min(np.stack((weights[phase],attribute_weights), axis=0), axis=0)

    return weights



def get_dic_indices(classes, labels_train_dir, exclusion_cases, in_cuda=True):
    dic_indices = {}
    lbls = get_lbls(labels_train_dir)
    names = sorted(lbls.keys())
    
    indices= list(range(len(names)))
    included_indices = set(indices)
    
    for i in indices:
        for attribute in exclusion_cases:
            if lbls[names[i]][attribute] in exclusion_cases[attribute]:
                included_indices-= {i}
            
    
    
    
    #calcul des poids:            
    for attribute in classes:
        list_of_indices = []
        for list_of_labels in classes[attribute]:
            indices = [i for i in included_indices if lbls[names[i]][attribute] in list_of_labels]
            list_of_indices.append(indices)
        dic_indices[attribute]=list_of_indices

    return dic_indices



def modify_state_dict(state_dict):
    pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]



def get_list_of_labels(attribute):
    
    #ground
    if attribute == 'ground_5':
        return [['dry_road'],['wet_road'], [ 'snow_ground','snow_ground_dry_road'],['snow_road'],['white_road']]
    elif attribute == 'ground_4':
        return [['dry_road','wet_road'], [ 'snow_ground','snow_ground_dry_road'],['snow_road'],['white_road']]
    elif attribute == 'ground_2':
        return [['dry_road','wet_road'], [ 'snow_ground','snow_ground_dry_road','snow_road','white_road']]
    elif attribute == 'ground_2_separated':
        return [['dry_road','wet_road'], [ 'snow_ground_dry_road','snow_road','white_road']]

    #atmo:
#    elif attribute == 'atmo_all_negdoubt':
#        return [['no_precip','doubt'],['precip'], ['rain'],['fog','fog_or_snow'],['snow']]
    elif attribute == 'atmo_2_negdoubt':
        return [['no_precip', 'doubt'],['precip','rain','fog','fog_or_snow','snow']]
    elif attribute == 'atmo_3_negdoubt':
        return [['no_precip', 'doubt'],['precip','rain','fog','fog_or_snow'],['snow']]
    elif attribute == 'atmo_2_separated':
        return [['no_precip', 'doubt'],['snow']]


    elif attribute == 'atmo_2_nodoubt':
        return [['no_precip'],['precip','rain','fog','fog_or_snow','snow']]
    elif attribute == 'atmo_3_nodoubt':
        return [['no_precip'],['precip','rain','fog','fog_or_snow'],['snow']]


    elif attribute == 'atmo_2_posdoubt':
        return [['no_precip'],['precip','rain','fog','fog_or_snow','snow']]
    elif attribute == 'atmo_3_posdoubt':
        return [['no_precip'],['doubt','precip','rain','fog'],['fog_or_snow','snow']]
    
    #snowfall
    elif attribute == 'snowfall':
        return [['no'],['streaks']]
    elif attribute == 'snowfall_negdoubt':
        return [['no','doubt'],['streaks']]
    elif attribute == 'snowfall_posdoubt':
        return [['no'],['streaks','doubt']]
    
    elif attribute == 'mask':    #mask
        return [['no','filth'],['droplets', 'droplets_acc'],['snowflakes','snowflakes_acc']]
    elif attribute == 'mask_2':    #mask
        return [['no','filth'],['droplets', 'droplets_acc','snowflakes','snowflakes_acc']]
    elif attribute == 'mask_nofilth':    #mask
        return [['no'],['droplets', 'droplets_acc'],['snowflakes','snowflakes_acc']]

    elif attribute == 'mask_nofilth':    #mask
        return [['no'],['droplets', 'droplets_acc'],['snowflakes','snowflakes_acc']]

    elif attribute == 'time_2':
        return [['night','dark_night'],['day']]
    elif attribute == 'time_3':
        return [['night','dark_night'],['inter'],['day']]

def make_dic_of_classes(attributes_):
    classes = {}
    for attribute_ in attributes_:
        attribute = re.split('_',attribute_)[0]
        classes[attribute] =  get_list_of_labels(attribute_)
    return classes


def get_nclasses(attributes_):
    classes = make_dic_of_classes(attributes_)
    nclasses = 0
    for attribute in classes:
        nclasses += len(classes[attribute])
    return nclasses


        
def get_model_suffixe(attributes_):
    suf = ''
    for at in attributes_:
        suf+=at+'_'
    suf = suf[:-1]
    return suf


def imshow(inp, title=None):
    #Imshow for Tensor
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def get_names_in_frame(lbls, level = 'superframe'):
    names_in_frame = {}
    frames = {lbls[name][level] for name in lbls} 
    for frame in frames:
        names_in_frame[frame] = {name for name in lbls if lbls[name][level] == frame}
    return names_in_frame


def get_nearers_in_same_frame_same_time(lbls,level = 'superframe'):
    names_in_frame = get_names_in_frame(lbls,level)
    nearers = {}
    for name in lbls:
        time_ref = lbls[name]['time']
        frame_ref = lbls[name][level] 
        names = names_in_frame[frame_ref]
        if len(names) == 0:
            print(frame_ref)
        
        times_ok = [time_ref]
        if time_ref == 'inter':  #cas du crépuscule: peu d'images. On ajoute les images de jour de la frame
            times_ok.append('day')
        
        ns = {n for n in names if lbls[n]['time'] in times_ok }
        
        if len(ns)>1:
            ns -= {name}
        
        nearers[name] = list(ns)
        
    return nearers

def get_nearers_in_same_frame(lbls,level = 'superframe'):
    names_in_frame = get_names_in_frame(lbls,level)
    nearers = {}
    for name in lbls:
        frame_ref = lbls[name][level] 
        names = names_in_frame[frame_ref]
        ns = set(names)
        
        if len(ns)>1:
            ns -= {name}
        
        nearers[name] = list(ns)
        
    return nearers

def clean_lbls(lbls,exclusion_cases):  #exclusion_cases: dic atttribute - class to remove
    names_to_remove = set()
    for name in lbls:
        for attribute in exclusion_cases:
            if lbls[name][attribute] in exclusion_cases[attribute]:
                names_to_remove|={name}
    for name in names_to_remove:
        del lbls[name]
        
def invert_edges(edges):
    inverted_edges = set()
    for edge in edges:
        inverted_edges |={(edge[1], edge[0])}
    return inverted_edges

def edges_and_inverted_edges(edges):
    edges = set(edges)
    inverted_edges = invert_edges(edges)
    return edges.union(inverted_edges)
