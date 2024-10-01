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
import networkx as nx
import pickle


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





####################################################
############## Building graphs #####################



def read_gpickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except EOFError:
        print(f"Error: The file at '{path}' is empty or corrupted.")
        raise
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle the file at '{path}'. The file might be corrupted.")
        raise

def write_gpickle(graph, path):
    with open(path, 'wb') as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def exclude_from_graph(graph,exclusion_cases):
    nodes_to_remove = []
    for attribute in exclusion_cases:
        for node in graph:
            if graph.nodes[node][attribute] in exclusion_cases[attribute]:
                nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)




def kill_Id_edges(graphs):
    for graph in graphs:
        edges_to_remove =[]
        for edge in graph.edges:
            if edge[0] == edge[1]:
                edges_to_remove.append(edge)
        graph.remove_edges_from(edges_to_remove)


def oracle_from_AMOSlbls(lbls, prev_name, name, param, mode):

    compa = lbls[name]['compa']
    visi = lbls[name]['visi']
    time = lbls[name]['time']
    ground = lbls[name]['ground']
    traces = lbls[name]['old snow_traces']
    noise = lbls[name]['noise']
    
    prev_time = lbls[prev_name]['time']
    prev_ground = lbls[prev_name]['ground']
    
    #hauteur de neige, cas général:
    if param == 'sh':

        if compa == 'snow_up':
            label = 1
        elif compa == 'snow_down':
            label = 0
        elif compa == 'snow_eq':
            label = 2
        elif compa == 'snow_eq3':
            label = 3
        elif compa == 'no_comp':
            label = 3

        #cas particulier 2 changement de couverture: dépendance au mode.
        if  prev_ground != ground:
            #épaisseur + mais surface -:
            if label == 1 and prev_ground == 'white_road' and ground == 'snow_road':
                if mode == 'surface':
                    label = 0
            if label == 1 and prev_ground == 'snow_road' and ground == 'snow_ground':
                if mode == 'surface':
                    label = 0        
            #épaisseur = mais surface +:
            if label in [2,3] and prev_ground == 'snow_road' and ground == 'white_road':
                if mode == 'surface':
                    label = 1 
            if label in [2,3] and prev_ground == 'snow_ground' and ground == 'snow_road':
                if mode == 'surface':
                    label = 1
            #cas d'omission labellisation
            if prev_ground in ['snow_ground','snow_road','white_road'] and ground in ['wet_road','dry_road']:
                    label = 1
            if prev_ground in ['dry_road','wet_road'] and ground in ['wet_road','dry_road']:
                    label = 3
        
        #utilisation de new snow
        if label in [0,2,3] and traces == 'new_snow' and ground in ['snow_road','white_road']:
            if mode == 'height':
                label = 1 
    #        if label in [0] and traces == 'new_snow' and ground == 'snow_road':
    #            if mode == 'height':
    #                label = 1

        #cas particulier 3: en mode height, pas utiliser les changements de surface si snow road
        if  mode == 'height':
            if (prev_ground == 'snow_road') or (ground == 'snow_road'):
                if label != 2:
                    print('get rid of that fluctuation')
                    label = None

    #visibilité, cas général:
    if param == 'vv':
        
        if visi == 'farer':
            label = 1
        elif visi == 'lower':
            label = 0
        elif visi == 'eq':
            label = 2
        elif visi == 'eq3':
            label = 3
        elif visi == 'no_comp':
            label = 3
            
    #cas des changements de luminosité  eq -> eq3:
    if time != prev_time and label == 2:
        label = 3  #dégradation du label pour l'instant
        
    
        
    #cas des bruits:
    if noise in ['blurry','miss_rec', 'other'] and label == 2:
        label = 3  #dégradation du label pour l'instant        
#    if noise in ['surexp'] and and time = ['night'] and label == 2:
#        label = 3  #dégradation du label pour l'instant 
    return label


def oracle_from_TENEBRElbls(lbls, prev_name, name, param, mode):
    
    tresh_sh = 1  #ce qu'il faut de différence pour un label ordinal 
    tresh_vv = 0.25   # seuil en log10 -> 100 m 
    
    sh = lbls[name]['sh']
    vv = lbls[name]['vv']
    time = lbls[name]['time']
    ground = lbls[name]['ground']
    rr6 = lbls[name]['atmo']


    prev_sh = lbls[prev_name]['sh']
    prev_vv = lbls[prev_name]['vv']
    prev_rr6 = lbls[prev_name]['atmo']    
    prev_time = lbls[prev_name]['time']
    prev_ground = lbls[prev_name]['ground']




    
    #hauteur de neige, cas général:
    if param == 'sh':

        if prev_sh >= sh + tresh_sh:
            label = 0
        elif prev_sh + tresh_sh <= sh :
            label = 1
        elif prev_sh >= sh - tresh_sh and prev_sh <= sh + tresh_sh:
            label = 3
        else:
            label = None



    #visibilité, cas général:
    if param == 'vv':
        if np.log10(prev_vv) >= np.log10(vv) + tresh_vv:
            label = 0
        elif np.log10(prev_vv) + tresh_vv <= np.log10(vv) :
            label = 1
        elif np.log10(prev_vv) >= np.log10(vv) - tresh_vv and np.log10(prev_vv) <= np.log10(vv) + tresh_vv:
            label = 3
        else:
            label = None
            
    #on s'assure que les deux images sont de la même scène:
    if lbls[prev_name]['superframe'] != lbls[name]['superframe']:
        label = None

    return label



#Les trois fonctions suivantes sont les mêmes que dans tri_fonction
    

def rebuild(graphs):
    dg, ug, eg = graphs
    for edge in set(dg.edges):
        impact_new_dg_edge(graphs,edge)
    for edge in set(ug.edges):
        impact_new_ug_edge(graphs,edge)

#def impact_neweq(graphs,edge_eq):
#    
#    name0, name1 = edge_eq
#    dg, ug, eg = graphs                
#
#    component = nx.descendants(eg,name1)
#    component.add(name1)
#    nodes_up = set()
#    nodes_down= set()
#    nodes_unr = set()
#    for node in component:
#        nodes_up |= set(dg.predecessors(node))
#        nodes_down |= set(dg.successors(node))
#        nodes_unr |= set(ug.neighbors(node))
#    
#    new_dg_edges = set()
#    new_ug_edges = set()
#    
#    new_dg_edges |= {(node0,node1) for node0 in nodes_up for node1 in component}
#    new_dg_edges |= {(node0,node1) for node0 in component for node1 in nodes_down}
#    new_ug_edges |= {(node0,node1) for node0 in component for node1 in nodes_unr}
#
#    dg.add_edges_from(new_dg_edges)
#    ug.add_edges_from(new_ug_edges)
#
#    return new_dg_edges
def impact_new_dg_edge(graphs,dg_edge):
    dg, _, eg = graphs   
    name0,name1 = dg_edge
    
    #1 get connexe components from eg
    component0 = nx.node_connected_component(eg,name0)
    component1 = nx.node_connected_component(eg,name1)

    #build new links
    new_dg_edges = {(node0,node1) for node0 in component0 for node1 in component1}
    dg.add_edges_from(new_dg_edges)    
    
    
def impact_new_ug_edge(graphs,ug_edge):
    _, ug, eg = graphs   
    name0,name1 = ug_edge
        
    #1 get connexe components from eg
    component0 = nx.node_connected_component(eg,name0)
    component1 = nx.node_connected_component(eg,name1)

    #2build new links    
    new_ug_edges = {(node0,node1) for node0 in component0 for node1 in component1}
    ug.add_edges_from(new_ug_edges)    

def make_dg_from_eg(dg, eg):  #get the graph of true labels

    a = copy.deepcopy(dg)
#    get_naked([a])
    b = copy.deepcopy(eg).to_directed()
    b = nx.compose( b ,   nx.reverse(b) )   #symmetrization of b

    complete_dg = nx.compose(a, b)
    
    return complete_dg

def comprise_in(x,b0,b1):
    if x>b0 and x<= b1:
        return True

def get_sdg_from_TENEBRElbls(graphs,param,mode, tresh, size_by_interval = 50000): #pas codé pour vv
    
    if param == 'sh':
        dg = graphs[0]
        ug = graphs[1]
        eg = graphs[2]    
        names = sorted(list(dg.nodes))
    
        #adding edges:
        new_dg_edges = set()
        new_ug_edges = set()
        new_eg_edges = set()
        tresh_sh = tresh
        
        snow_names0 = [name for name in names if dg.nodes[name][param]<=2]
        snow_names1 = [name for name in names if comprise_in(dg.nodes[name][param],1,3)]
        snow_names2 = [name for name in names if comprise_in(dg.nodes[name][param],2,7)]
        snow_names5 = [name for name in names if dg.nodes[name][param] > 5]
    
        dic = dg.nodes
        
        for snow_names in [snow_names0,snow_names1,snow_names2,snow_names5]:
            print('new intervall')
            for name in snow_names:
                frame = dic[name]['frame']
    #            print('newb')
                tm = dic[name]['time']
                for name2 in snow_names:
                    if dic[name2]['frame'] == frame:
                         if dic[name2]['time'] == tm:
                             if dic[name][param] > tresh_sh + dic[name2][param]  and dic[name][param] < 2*tresh_sh + dic[name2][param]:
                                 new_dg_edges |= {(name,name2)}
            print(len(new_dg_edges))
            new_dg_edges = set(random.sample(new_dg_edges,size_by_interval))
            print(len(new_dg_edges))
            
            dg.add_edges_from(new_dg_edges)
            ug.add_weighted_edges_from(new_ug_edges)
            eg.add_weighted_edges_from(new_eg_edges)
    
        rebuild(graphs) #attention: upgrader si ug utile


    if param == 'vv':
        dg = graphs[0]
        ug = graphs[1]
        eg = graphs[2]    
        names = sorted(list(dg.nodes))
    
        #adding edges:
        new_dg_edges = set()
        new_ug_edges = set()
        new_eg_edges = set()
        tresh_vv = tresh
        
        POM_names0 = [name for name in names if dg.nodes[name][param]<=200]
        POM_names1 = [name for name in names if comprise_in(dg.nodes[name][param],100,500)]
        POM_names2 = [name for name in names if comprise_in(dg.nodes[name][param],250,1000)]
        POM_names3 = [name for name in names if comprise_in(dg.nodes[name][param],500,5000)]
        POM_names4 = [name for name in names if comprise_in(dg.nodes[name][param],2500,10000)]
        POM_names5 = [name for name in names if dg.nodes[name][param] > 5000]
    
        dic = dg.nodes
        
        for POM_names in [POM_names0 ,POM_names1,POM_names2,POM_names3, POM_names4,POM_names5]:
            print('new intervall')
            for name in POM_names:
                frame = dic[name]['frame']
    #            print('newb')
                tm = dic[name]['time']
                for name2 in POM_names:
                    if dic[name2]['frame'] == frame:
                         if dic[name2]['time'] == tm:
                             if np.log2(dic[name][param]) > tresh_vv + np.log2(dic[name2][param]) and np.log2(dic[name][param]) < 2*tresh_vv + np.log2(dic[name2][param]):
                                 new_dg_edges |= {(name,name2)}
            l = len(new_dg_edges)
            print(len(new_dg_edges))
            new_dg_edges = set(random.sample(new_dg_edges,min(size_by_interval,l)))
            print(len(new_dg_edges))
            
            dg.add_edges_from(new_dg_edges)
            ug.add_weighted_edges_from(new_ug_edges)
            eg.add_weighted_edges_from(new_eg_edges)
    
        rebuild(graphs) #attention: upgrader si ug utile
    

    return graphs






def get_sdg_from_TENEBRElbls(graphs,param,mode, tresh, size_by_interval = 50000): #pas codé pour vv
    
    if param == 'sh':
        dg = graphs[0]
        ug = graphs[1]
        eg = graphs[2]    
        names = sorted(list(dg.nodes))
    
        #adding edges:
        new_dg_edges = set()
        new_ug_edges = set()
        new_eg_edges = set()
        tresh_sh = tresh
        
        snow_names0 = [name for name in names if dg.nodes[name][param]<=2]
        snow_names1 = [name for name in names if comprise_in(dg.nodes[name][param],1,3)]
        snow_names2 = [name for name in names if comprise_in(dg.nodes[name][param],2,7)]
        snow_names5 = [name for name in names if dg.nodes[name][param] > 5]
    
        dic = dg.nodes
        
        for snow_names in [snow_names0,snow_names1,snow_names2,snow_names5]:
            print('new intervall')
            for name in snow_names:
                frame = dic[name]['frame']
    #            print('newb')
                tm = dic[name]['time']
                for name2 in snow_names:
                    if dic[name2]['frame'] == frame:
                         if dic[name2]['time'] == tm:
                             if dic[name][param] > tresh_sh + dic[name2][param]  and dic[name][param] < 2*tresh_sh + dic[name2][param]:
                                 new_dg_edges |= {(name,name2)}
            print(len(new_dg_edges))
            new_dg_edges = set(random.sample(new_dg_edges,size_by_interval))
            print(len(new_dg_edges))
            
            dg.add_edges_from(new_dg_edges)
            ug.add_weighted_edges_from(new_ug_edges)
            eg.add_weighted_edges_from(new_eg_edges)
    
        rebuild(graphs) #attention: upgrader si ug utile


    if param == 'vv':
        dg = graphs[0]
        ug = graphs[1]
        eg = graphs[2]    
        names = sorted(list(dg.nodes))
    
        #adding edges:
        new_dg_edges = set()
        new_ug_edges = set()
        new_eg_edges = set()
        tresh_vv = tresh
        
        POM_names0 = [name for name in names if dg.nodes[name][param]<=200]
        POM_names1 = [name for name in names if comprise_in(dg.nodes[name][param],100,500)]
        POM_names2 = [name for name in names if comprise_in(dg.nodes[name][param],250,1000)]
        POM_names3 = [name for name in names if comprise_in(dg.nodes[name][param],500,5000)]
        POM_names4 = [name for name in names if comprise_in(dg.nodes[name][param],2500,10000)]
        POM_names5 = [name for name in names if dg.nodes[name][param] > 5000]
    
        dic = dg.nodes
        
        for POM_names in [POM_names0 ,POM_names1,POM_names2,POM_names3, POM_names4,POM_names5]:
            print('new intervall')
            for name in POM_names:
                frame = dic[name]['frame']
    #            print('newb')
                tm = dic[name]['time']
                for name2 in POM_names:
                    if dic[name2]['frame'] == frame:
                         if dic[name2]['time'] == tm:
                             if np.log2(dic[name][param]) > tresh_vv + np.log2(dic[name2][param]) and np.log2(dic[name][param]) < 2*tresh_vv + np.log2(dic[name2][param]):
                                 new_dg_edges |= {(name,name2)}
            l = len(new_dg_edges)
            print(len(new_dg_edges))
            new_dg_edges = set(random.sample(new_dg_edges,min(size_by_interval,l)))
            print(len(new_dg_edges))
            
            dg.add_edges_from(new_dg_edges)
            ug.add_weighted_edges_from(new_ug_edges)
            eg.add_weighted_edges_from(new_eg_edges)
    
        rebuild(graphs) #attention: upgrader si ug utile
    

    return graphs



def get_sdg_from_AMOSlbls(graphs, lbls,param,mode, level = 'superframe'):
    names = sorted(list(lbls.keys()))
    dg = graphs[0]
    ug = graphs[1]
    eg = graphs[2]
    

    intraframe = True
    
    #adding edges:
    new_dg_edges = set()
    new_ug_edges = set()
    new_eg_edges = set()

    # in the time order
        
    for i in range(1,len(names)):
        prev_name = names[i-1]
        name= names[i]
        if lbls[name][level] == lbls[prev_name][level] or not intraframe:
            label = oracle_from_AMOSlbls(lbls, prev_name, name, param, mode)
            if label == 0:
                new_dg_edges |= {(prev_name, name,0)}
            elif label == 1:
                new_dg_edges |= {(name, prev_name,0)}
            elif label == 2:
                new_ug_edges |= {(name, prev_name,0)}       
                new_eg_edges |= {(name, prev_name,0)}
            elif label == 3:
                new_ug_edges |= {(name, prev_name,0)}
                
    #        if dg.nodes[name]['visi'] == 'no_comp' and label != None:
    #            print(label)

    
    dg.add_weighted_edges_from(new_dg_edges)
    ug.add_weighted_edges_from(new_ug_edges)
    eg.add_weighted_edges_from(new_eg_edges)
    
    #ajouter l'info d'égalité
    rebuild(graphs) #attention: upgrader qd ug utile
    kill_Id_edges(graphs)

    return graphs



def get_sdg_from_levels(graphs, lbls, param):
    names = sorted(list(lbls.keys()))
    seqs = sorted({lbls[name]['sequence'] for name in names})
    seq2cpls = {seq: [(n,lbls[n]['level' + param])  \
                      for n in lbls if lbls[n]['sequence'] == seq] 
                for seq in seqs}

    dg = graphs[0]
    ug = graphs[1]
    eg = graphs[2]

    intraframe = True
    
    #adding edges:
    new_dg_edges = set()
    new_ug_edges = set()
    new_eg_edges = set()

    # Pour vv. pour chaqeue nom: on regarde au dessus
    if param == 'vv':
        for seq in seqs:
            print(seq)
            cpls = seq2cpls[seq]

            for n0, level0 in cpls:
                
                # si pas annoté:
                if level0 is None:
                    pass
                else:
                    level0 = level0.replace('p','')
                    level0 = level0.replace('m','f')
                    zmax0 = level0[0]
                    # si borne non fournie ou changement de caméra: 
                    if (zmax0 == 'n') or ('c' in level0):
                        pass
                    elif zmax0 in ['f']:
                        new_ug_edges |= {(n0, cpl[0]) for cpl in cpls \
                                        if cpl[0] != n0}

                    else:
                        for n1, level1 in cpls:
                            if (n0 != n1) and (level1 is not None):
                                level1 = level1.replace('p','')
                                level1 = level1.replace('m','f')
                                zmin1 = level1[-1]
                                if zmin1 not in ['n', 'f'] and not ('c' in level1):
                                    # on intervertit lo'rdre normal car 
                                    # vv noté de 0 à 9 avec 9 = brouillard
                                    if int(zmax0) > int(zmin1):
                                        new_dg_edges |= {(n1, n0)}

    elif param == 'sc':
        for seq in seqs:
            print(seq)
            cpls = seq2cpls[seq]

            for n0, level0 in cpls:
                
                # si pas annoté:
                if level0 is None:
                    pass
                else:
                    level0 = level0.replace('p','')
                    level0 = level0.replace('m','f')
                    zmax0 = level0[0]
                    # si borne non fournie ou changement de caméra: 
                    if (zmax0 == 'n') or ('c' in level0):
                        pass
                    #elif zmax0 in ['f']:
                    #     new_ug_edges |= {(n0, cpl[0]) for cpl in cpls \
                    #                    if cpl[0] != n0}

                    else:
                        for n1, level1 in cpls:
                            if (n0 != n1) and (level1 is not None):
                                level1 = level1.replace('p','')
                                level1 = level1.replace('m','f')
                                zmin1 = level1[-1]
                                if zmin1 not in ['n', 'f'] and not ('c' in level1):
                                    if int(zmax0) > int(zmin1):
                                        new_dg_edges |= {(n1, n0)}

            two_cycles = new_dg_edges.intersection(invert_edges(new_dg_edges))
            if len(two_cycles) > 0 :
                 # print(new_dg_edges)
                 print(two_cycles)
                 print(seq2cpls[seq])

                 raise Exception('')
    print(new_dg_edges)      

    dg.add_edges_from(new_dg_edges)
    ug.add_edges_from(new_ug_edges)
    eg.add_edges_from(new_eg_edges)
    
    rebuild(graphs) #attention: upgrader qd ug utile
    kill_Id_edges(graphs)

    return graphs




def get_sdg_from_levels_ss(graphs, lbls, param, mode, level = 'sequence'):
    names = sorted(list(lbls.keys()))
    seqs = sorted({lbls[name][level] for name in names})
    seq2cpls = {seq: [(n,lbls[n]['levelss'])  \
                      for n in lbls if lbls[n][level] == seq] 
                for seq in seqs}

    dg = graphs[0]
    ug = graphs[1]
    eg = graphs[2]

    intraframe = True
    
    #adding edges:
    new_dg_edges = set()
    new_ug_edges = set()
    new_eg_edges = set()



    dg.add_edges_from(new_dg_edges)
    ug.add_edges_from(new_ug_edges)
    eg.add_edges_from(new_eg_edges)
    
    rebuild(graphs) #attention: upgrader qd ug utile
    kill_Id_edges(graphs)

    return graphs



def get_new_graphs(lbls):
    names = sorted(list(lbls.keys()))
    
    dg1 = nx.DiGraph()
    ug1 = nx.Graph()
    eg1 = nx.Graph()
    
    #adding nodes:
    dg1.add_nodes_from(names)
    ug1.add_nodes_from(names)
    eg1.add_nodes_from(names)
 
    #adding attributes on dg:
    nx.set_node_attributes(dg1,lbls)
    nx.set_node_attributes(ug1,lbls)
    nx.set_node_attributes(eg1,lbls) 
    
    return [dg1, ug1, eg1]





def new_get_wdg_from_AMOSlbls(graphs, param ,mode, intra_frame = False, nb_by_frame = 100, level = 'superframe'):
    dg = graphs[0]
    ug = graphs[1]
    eg = graphs[2]    
    
    frames = {dg.nodes[name][level] for name in dg}
    names = sorted(list(dg.nodes))
    lbls = dg.nodes
#    print(frames)
    #ug = graphs[1]
    #eg = graphs[2]
        
    #initialization:
    new_dg_edges = set()
    new_ug_edges = set()
    new_eg_edges = set()

    
    if param == 'sh':

        for frame in frames:
            new_frame_edges = set()
            
            
            names_to_pick = names  #ON NE VEUT PAS DEUX FOIS LA MËME IMAGE
            
            for name in names:
            
                
                
                if lbls[name][level] == frame and lbls[name]['ground'] in ['snow_ground','snow_road','white_road']:
                    for name2 in names_to_pick:
                        if (lbls[name2][level] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                           if lbls[name2]['ground'] in ['dry_road','wet_road']:                
                                new_frame_edges |= {(name, name2, 1)}
#                                names_to_pick.remove(name2)
            
            l = len(new_frame_edges)
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame, l))
            new_dg_edges |= set(new_frame_edges)
#            print(len(new_dg_edges))
        
        if mode == 'surface':
            for frame in frames:
                new_frame_edges = set()
                
                
                names_to_pick = names  #ON NE VEUT PAS DEUX FOIS LA MËME IMAGE
                
                for name in names:
                
                    
                    
                    if lbls[name][level] == frame and lbls[name]['ground'] in ['snow_road','white_road']:
                        for name2 in names_to_pick:
                            if (lbls[name2][level] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                               if lbls[name2]['ground'] in ['snow_ground', 'no_snow_road']:                
                                    new_frame_edges |= {(name, name2, 1)}
    #                                names_to_pick.remove(name2)
                
                l = len(new_frame_edges)
                new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame, l))
                new_dg_edges |= set(new_frame_edges)
    #            print(len(new_dg_edges))
            

            
            for frame in frames:
                new_frame_edges = set()
                
                for name in names:
                    if lbls[name][level] == frame and lbls[name]['ground'] in ['white_road']:
                        for name2 in names:
                            if (lbls[name2][level] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                               if lbls[name2]['ground'] in ['snow_road']:                
                                    new_frame_edges |= {(name, name2, 1)}
                l = len(new_frame_edges)                    
                new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame,l))
                new_dg_edges |= set(new_frame_edges)                            
#            print(len(new_dg_edges))
            
        dg.add_weighted_edges_from(new_dg_edges)     

        #case of ug/eg

        for frame in frames:
            names_frame = [name for name in names if lbls[name][level] == frame]            
            new_frame_edges = {(n0,n1) for n0 in names_frame if (lbls[n0]['sh']==0) for n1 in names_frame if (lbls[n1]['sh']==0) }
#            print(len(new_frame_edges))
            l = len(new_frame_edges)   
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame,l))
            new_ug_edges |= set(new_frame_edges)   
            new_eg_edges |= set(new_frame_edges)                          
            print(len(new_ug_edges))
            print(len(new_eg_edges))
            
        ug.add_edges_from(new_ug_edges)                          
        eg.add_edges_from(new_eg_edges)                                   

    if param == 'vv':

        for frame in frames:
            new_frame_edges = set()
            
            for name in names:
                if lbls[name][level] == frame and lbls[name]['atmo'] in ['no_precip']:
                    for name2 in names:
                        if (lbls[name2][level] == frame) or ( (lbls[name2][level] != frame) and not intra_frame):  #intra_frame précise si on cherche dans ou en dehors de la frame
                           if lbls[name2]['atmo'] in ['precip','rain','fog','snow','fog_or_snow']:                
                                new_frame_edges |= {(name, name2, 1)}
            l = len(new_frame_edges)                    
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame,l))
            new_dg_edges |= set(new_frame_edges)                          
            print(len(new_dg_edges))         
        dg.add_weighted_edges_from(new_dg_edges)           
        
        
    return graphs



def get_wug_from_AMOSlbls(graphs, param ,mode, intra_frame = False, nb_by_frame = 100):
    
    ug = graphs[1]    
    frames = {ug.nodes[name]['superframe'] for name in ug}
    names = sorted(list(ug.nodes))
    lbls = ug.nodes

    new_ug_edges = set()

    
    if param == 'sh':

        for frame in frames:
            new_frame_edges = set()
            
            
            names_to_pick = names  #ON NE VEUT PAS DEUX FOIS LA MËME IMAGE
            
            for name in names:
            
                
                
                if lbls[name]['superframe'] == frame and lbls[name]['ground'] in ['dry_road','wet_road'] and lbls[name]['old snow_traces'] !='ground':
                    for name2 in names_to_pick:
                        if (lbls[name2]['superframe'] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                           if lbls[name2]['ground'] in ['dry_road','wet_road'] and lbls[name2]['old snow_traces'] !='ground':                
                                new_frame_edges |= {(name, name2,1)}
#                                names_to_pick.remove(name2)
            
            l = len(new_frame_edges)
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame, l))
            new_ug_edges |= set(new_frame_edges)
#            print(len(new_dg_edges))


            
        ug.add_weighted_edges_from(new_ug_edges)
        kill_Id_edges(graphs)

    if param == 'vv':

        for frame in frames:
            new_frame_edges = set()
            
            for name in names:
                if lbls[name]['superframe'] == frame and lbls[name]['atmo'] in ['no_precip']:
                    for name2 in names:
                        if (lbls[name2]['superframe'] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                           if lbls[name2]['atmo'] in ['no_precip']:                
                                new_frame_edges |= {(name, name2,1)}
            l = len(new_frame_edges)                    
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame,l))
            new_ug_edges |= set(new_frame_edges)                          
#            print(len(new_dg_edges))

        ug.add_weighted_edges_from(new_ug_edges)
        kill_Id_edges(graphs)
         
    return graphs


def add_self_edges(graph):
    for node in graph:
        graph.add_edge(node,node)


#%%
def touches_snow_road(dg,e):  #says if an edge has one of its node in snow_road        
    if dg.nodes[e[0]]['ground']=='snow_road' or dg.nodes[e[1]]['ground']=='snow_road':
        print('edge detroyed')
        return True
    else:
        return False
    
def fill_graph_from_splitted(graphs,splitted_dir, ext = 'vvday', forbidden_weights= None, only_eg = False, except_snow_road = False):
    sequences = os.listdir(splitted_dir)
    
    tdg,tug,teg = graphs
    
    for sequence in sequences:
        try:
            path = os.path.join(splitted_dir,sequence,'labels_ord')
            dg_path = os.path.join(path, 'dg_' + ext + '.gpickle')
            ug_path = os.path.join(path, 'ug_' + ext + '.gpickle')
            eg_path = os.path.join(path, 'eg_' + ext + '.gpickle')
            
            local_dg = nx.read_gpickle(dg_path) 
            local_ug = nx.read_gpickle(ug_path)
            local_eg =nx.read_gpickle(eg_path)
#            print(len(local_ug.edges))
            
            if forbidden_weights is not None:
                dg_edges_to_remove = [e for e in local_dg.edges if local_dg.edges[e].get('weight') in forbidden_weights]
                ug_edges_to_remove = [e for e in local_ug.edges if local_ug.edges[e].get('weight') in forbidden_weights]
                eg_edges_to_remove = [e for e in local_eg.edges if local_eg.edges[e].get('weight') in forbidden_weights]
                
                local_dg.remove_edges_from(dg_edges_to_remove)
                local_ug.remove_edges_from(ug_edges_to_remove)
                local_eg.remove_edges_from(eg_edges_to_remove)

            if except_snow_road:
                dg_edges_to_remove2 = [e for e in local_dg.edges if touches_snow_road(local_dg,e)]
                ug_edges_to_remove2 = [e for e in local_ug.edges if touches_snow_road(local_dg,e)]
                local_dg.remove_edges_from(dg_edges_to_remove2)
                local_ug.remove_edges_from(ug_edges_to_remove2)


            if not only_eg:
                tdg = nx.compose(tdg, local_dg)
                tug = nx.compose(tug, local_ug)    
            

                
            
            teg = nx.compose(teg, local_eg) 
            
            
        except:
            pass
            print('nothing in sequence: ' + str(sequence))
            
    return (tdg, tug, teg)
    
        
def make_graph_from_splitted(splitted_dir, ext = 'vvday'):
    sequences = os.listdir(splitted_dir)
    
    tdg = nx.DiGraph()
    tug = nx.Graph()
    teg = nx.Graph()
    
    for sequence in sequences:
        try:
            path = os.path.join(splitted_dir,sequence,'labels_ord')
            dg_path = os.path.join(path, 'dg_' + ext + '.gpickle')
            ug_path = os.path.join(path, 'ug_' + ext + '.gpickle')
            eg_path = os.path.join(path, 'eg_' + ext + '.gpickle')
            
            local_dg = nx.read_gpickle(dg_path) 
            local_ug = nx.read_gpickle(ug_path)
            local_eg = nx.read_gpickle(eg_path)

            tdg = nx.compose(tdg, local_dg)
            tug = nx.compose(tug, local_ug)    
            teg = nx.compose(teg, local_eg) 
#            print(len(tdg.edges))
            
        except:
            print('nothing in sequence: ' + str(sequence))
            
    return (tdg, tug, teg)
            

#%% for qual_graph

def init_qual_graph(lbls):
    names = sorted(list(lbls.keys()))
    
    graph = nx.Graph()
    
    #adding nodes:
    graph.add_nodes_from(names)

    #adding attributes on dg:
    nx.set_node_attributes(graph,lbls)
    
    return graph


def fill_qual_graph(graph):
    level = 'superframe'
    long = 3
    names = sorted(list(graph.nodes))
    lbls = graph.nodes
#    print(frames)
    #ug = graphs[1]
    #eg = graphs[2]
        
    #initialization:
    new_graph_edges = set()
    L = len(names)
    for i in range(L):
        for k in range(i+1,min(i+long,L)):
            if lbls[names[i]][level] == lbls[names[k]][level]:
                if lbls[names[i]]['time'] == lbls[names[k]]['time'] or lbls[names[k]]['time'] == 'inter'  or lbls[names[i]]['time'] == 'inter':
                    new_graph_edges|= {(names[i],names[k])}
    

    print(str(len(new_graph_edges)) +' edges in the graph')
    graph.add_edges_from(new_graph_edges)
    


