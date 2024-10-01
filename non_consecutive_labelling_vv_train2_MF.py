# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:59:32 2020

@author: Deep Yawner
"""
#%%

import sys
#path_code = r'/home/userdev/shared_with_vm/annotation/prepare_dataset'
#sys.path.append(path_code)
import threading
from threading import Thread
import time
import shutil
import os
from os.path import isdir, isfile, join
import urllib
#import cdsapi
#import netCDF4
#from netCDF4 import Dataset
#import urllib.request
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta, datetime, date
from matplotlib.pyplot import plot_date
from matplotlib.dates import drange
import pandas
import numpy as np
import random
import pickle
import subprocess
import pytz
#from tzwhere import tzwhere
import zipfile
#tzwhere = tzwhere.tzwhere()
import re
import json 
from datetime import timedelta, datetime, date

path_code = r'C:\Users\Deep Yawner\Desktop\tri_images\prepare_dataset'
os.chdir(path_code)
from utile import *
from utile_to_get_the_graph import *
path_code = r'C:\Users\Deep Yawner\Desktop\tri_images\prepare_dataset\POSET_mergesort'
os.chdir(path_code)
from fts_utiles_tri5_levelvv import *
#%% 

#%Local
root = r"D:\sniff_webcams\AMOS"
#root = r'/home/userdev/shared_with_vm'
#hal3
#root = r"/ssd/lepetit/neige"

dir_images = os.path.join(root,
                          r"BAMOSvv")
# train_dir = r"D:\sniff_webcams\datasets\AMOSDIR_17000"
# val_dir = os.path.join(root,r"AMOS_2000")
# test_dir = os.path.join(root, r"DIR_7000")
dir_models  = os.path.join(root,r"models")
dir_figures = os.path.join(root,r"figures")
dir_experiences = os.path.join(root,r"experiences")

dic_of_dirs = {'train': dir_images,
               'models': dir_models,
               'figures': dir_figures,
               'experiences': dir_experiences}


#%%

label_path = os.path.join(root, 
                    r'labels_imagewise_BAMOSvv_corr.pickle')


with open(label_path, 'rb') as f:
    lbls = pickle.load(f)

for name in lbls:
    if 'p' in lbls[name]['levelvv']:
        print(name)
    if lbls[name]['levelvv'] == '31':
        lbls[name]['levelvv'] = '13'
        print(name)


#lbls = {}
#for seq in lblsBySeq:
#    path_images = os.path.join(dir_images, seq, 'images')
#    names = sorted(os.listdir(path_images))
#    for i,name in enumerate(names):
#        try:
#            lbls[name] = {'levelvv': lblsBySeq[seq][i],
#                          'sequence': seq}
#        except:
#            print(seq)
#            lbls[name] = {'levelvv': None,
#                          'sequence': seq}




    
#%%
          
dir_splitted = os.path.join(root,
                            'BAMOSvv2')

print(os.listdir(dir_splitted))
#%% Step 1: define critere and mode

level =  'sequence'
sequences = sorted({lbls[name]['sequence'] for name in lbls})  
critere = 'vv'
dir_splitted_intra = dir_splitted
param = 'vv'
mode = ''
modes = [mode]
subgroup = 'day'
times = ['day']

grounds = [r'snow_ground', r'snow_road', r'white_road', r'snow_ground_dry_road']

seq2images = {}
for sequence in sequences:
    if critere == 'vv':
        seq2images[sequence] = {name for name in lbls \
            if (lbls[name]['sequence'] == sequence)
                }

    # elif critere =='snow':
    #     names_of_superframe[superframe] = {name for name in lbls \
    #         if (lbls[name][level] == superframe) \
    #             and (lbls[name]['ground'] in grounds) and lbls[name]['noise'] != r'miss_rec'}
         

sequences_to_sort = [seq for seq in seq2images  if len(seq2images[seq])>1]

print(sequences_to_sort, len(sequences_to_sort), len(sequences))
#%%step 2 Get the global sdg/ug/eg from AMOS lbls to prefill the frame graphs

graphs  = get_new_graphs(lbls)

#for g in graphs:
#    for name in g.nodes:
#        g.nodes[name]["levelvv"] = lbls[name]["levelvv"]

tdg, tug, teg = get_sdg_from_levels(graphs, lbls, param, mode)


two_cycles = set(tdg.edges).intersection(invert_edges(set(tdg.edges)))
two_cycles = edges_and_inverted_edges(two_cycles)
print(two_cycles)

#%%
#for cy in two_cycles:
#    print(lbls[cy[0]]['levelvv'], lbls[cy[1]]['levelvv'])

#%%new labels
print("Add new labels")


graphs= (tdg, tug, teg)
tdg, tug, teg = fill_graph_from_splitted(graphs, dir_splitted, ext =  'vvday')
print(len(tdg.edges), len(tug.edges), len(teg.edges))


#washing
two_cycles = set(tdg.edges).intersection(invert_edges(set(tdg.edges)))
two_cycles = edges_and_inverted_edges(two_cycles)
print(two_cycles)
#%%
print(str(len(two_cycles)) + ' 2-cycles')
tdg.remove_edges_from(two_cycles)
tdg = nx.transitive_closure(tdg)
two_cycles = set(tdg.edges).intersection(invert_edges(set(tdg.edges)))
two_cycles = edges_and_inverted_edges(two_cycles)
print(str(len(two_cycles)) + ' 2-cycles after washing')


vs03 = edges_and_inverted_edges(tug.edges).intersection(set(tdg.edges))
print(str(len(vs03)) + ' 0vs3')
vs03eg = {edge for edge in vs03 if edge in teg.edges}
tdg.remove_edges_from(vs03)
tug.remove_edges_from(vs03)
teg.remove_edges_from(vs03eg)

tdg = nx.transitive_closure(tdg)

vs03 = edges_and_inverted_edges(tug.edges).intersection(set(tdg.edges))
print(str(len(vs03)) + ' 0vs3 after washing')
print(len(tdg.edges), len(tug.edges), len(teg.edges))

#%%

# vendredi:
# - savoir d'où viennent ces 6 deux cycles ok

# - ajouter les incomparabilités sauf si l ou r
# - ajouter l'annot. manuelle au niveau des noeuds ok 
# - savoir pourquoi il y a des paires pas automatiquement traitées
# (ex im 0 085710, image1 090736). Pour ça: faire apparaître l'annot
# manuelle et le nom de l'image.ok (à cause d'un 19r)
#%%step 3: restriction to the nodes


for i, sequence in enumerate(['00010657_7002']):#enumerate(sequences_to_sort[400:]):  #[-42:-41]:
    print('sequence: ' + str(sequence), i)
    root_dataset = dir_splitted
    #cam = r"nancy2"
    
    
    dataset = os.path.join(root_dataset, str(sequence))
    images_dataset = join(dataset, 'images')
    labels_dataset = join(dataset, 'labels')
    root_cs = join(dataset,r'labels_ord')    
    
    if not isdir(root_cs):
        os.mkdir(root_cs)

    #init frame_lbls and dic of subgroup
    frame_lbls = {}
    for name in os.listdir(images_dataset):
            frame_lbls[name] = lbls[name]
    
    dic_of_subgroups = {}
    # list_snow = [r'snow_ground', r'snow_road', r'white_road', r'snow_ground_dry_road']
    # list_time = [r'day', r'inter', r'night', r'dark_night'] 
    
    # if critere == 'vv':
    #     for time in list_time:
    #         dic_of_subgroups[time] = [name for name in frame_lbls if lbls[name]['time'] == time and frame_lbls[name]['noise']] # != r'miss_rec']
    #     dic_of_subgroups['all_times'] = [name for name in frame_lbls if frame_lbls[name]['time'] in ['day', 'inter']] # and frame_lbls[name]['noise'] != r'miss_rec']


    # elif critere =='snow':
    #     for time in list_time:
    #         dic_of_subgroups[time] = [name for name in frame_lbls if frame_lbls[name][r'time'] == time and (frame_lbls[name][r'ground'] in list_snow or frame_lbls[name][r'old snow_traces'] in [r'ground'])]
    #     dic_of_subgroups['all_times'] = [name for name in frame_lbls if  (frame_lbls[name][r'ground'] in list_snow or frame_lbls[name][r'old snow_traces'] in [r'ground'])]

    dic_of_subgroups['day'] = [name for name in frame_lbls]
    
    labeling_root= r"D:\sniff_webcams\AMOS\tmp"
    temp_images_dir= os.path.join(labeling_root, r"images")
    temp_labels_dir= os.path.join(labeling_root, r"labels")


    
    
    
    #set mode depends on what is already done
    
    name_poset_vv = r'poset_vv.pickle'
    name_poset_snowsurface =r'poset_snow_surface.pickle'
    name_poset_snowheight = r'poset_snow_height.pickle'
    
    #%def graph paths
    suffixe = get_suffixe(critere,subgroup,mode)        
    graphs_names = [os.path.join(graph + suffixe+r'.gpickle') for graph in ["dg","ug","eg"]]
    graphs_paths = [os.path.join(root_cs, graph_name) for graph_name in graphs_names]


    #def nodes

    nodes = sorted(dic_of_subgroups[subgroup])


    print('labelling of ' + str(len(nodes)) + ' nodes')

    #clean images_dir
    for name in os.listdir(temp_images_dir):
        os.remove(os.path.join(temp_images_dir,name))

    #copy in the temp image dir
    for node in nodes:
        shutil.copy(os.path.join(dataset, 'images', node), os.path.join(temp_images_dir))


    
    if graphs_names[0] not in os.listdir(root_cs):
        print("need to init graphs")
        #restrict tdg, tug, teg:
        
        dg = tdg.subgraph(nodes).copy()
        ug = tug.subgraph(nodes).copy()                
        eg = teg.subgraph(nodes).copy()  
        
        #use the jpg as nodes
#        mapping = {node: json_to_jpg(node) for node in lbls}
#
#        dg = nx.relabel_nodes(dg, mapping)
#        ug = nx.relabel_nodes(ug, mapping)
#        eg = nx.relabel_nodes(eg, mapping)
        
        
        #save the graphs
        nx.write_gpickle(dg, graphs_paths[0])
        nx.write_gpickle(ug, graphs_paths[1])
        nx.write_gpickle(eg, graphs_paths[2])

    else:
        pass


    #COREN_2019-01-23_08_40.jpg
    #get decomposition and poset:
    
    kwargs = {'root_cs': root_cs,
              'critere':  critere,
              'subgroup': subgroup,
              'mode' : mode,
              'graphs_paths': graphs_paths,
              'images_dir' : temp_images_dir
              }
   
    decomposition =  labelling_mode(**kwargs)


#%%

    
#clean:
#00010268_0_20131114_081603.jpg
ntr = ['00010657_7002_20130721_180838.jpg']

clean_nodes(ntr, **kwargs)

#%%
path2 = os.path.join(root_cs, 'decomposition_vvday.pickle')
os.remove(path2)
#%%
#for name in os.listdir(splitted_dir):
#    path = os.path.join(splitted_dir,name,'labels_ord')
#    if 'decomposition_snowday_surface.pickle' in os.listdir(path):
#        path2 = os.path.join(path, 'decomposition_snowday_surface.pickle')
#        os.remove(path2)
#%%
# Correction de certain noms:
corrected_lbls = copy.deepcopy(lbls)

def get_dg2vvday(root_cs):
  
    path_of_dg2 = os.path.join(root_cs, "poset_vvday" + r".gpickle" )  
    dg2 = nx.read_gpickle(path_of_dg2)     
    return dg2

def save_dg2vvday(dg2, root_cs):
  
    path_of_dg2 = os.path.join(root_cs, "poset_vvday" + r".gpickle" )  
    nx.write_gpickle(dg2, path_of_dg2)
    print("graph dg2 saved")

def save_graphs(graphs, graphs_paths, **kwargs):
    dg, ug, eg = graphs
    nx.write_gpickle(dg, graphs_paths[0])
    nx.write_gpickle(ug, graphs_paths[1])
    nx.write_gpickle(eg, graphs_paths[2])
    print("graphs saved")

doublons  =[]
for i, sequence in enumerate(sequences_to_sort):  #[-42:-41]:
    
    root_dataset = dir_splitted
    #cam = r"nancy2"
    
    
    dataset = os.path.join(root_dataset, str(sequence))
    images_dataset = join(dataset, 'images')
    labels_dataset = join(dataset, 'labels')
    root_cs = join(dataset,r'labels_ord')    

    
    for name in os.listdir(images_dataset):
        if 'Copie' in name:
            print('sequence: ' + str(sequence), i, name)
            
            correct_name = name.split(' - ')[0] + '.jpg'
            level = copy.deepcopy(corrected_lbls[name])
            
            
            print(correct_name, level, correct_name in os.listdir(images_dataset))
            
            if correct_name in os.listdir(images_dataset):
                doublons.append(name)
                
            else:
                corrected_lbls[correct_name] = level
#            
            del corrected_lbls[name]
            
# Vérification:
for name in corrected_lbls:
    if 'Copie' in name:
        print(name)

#%% correction des graphs:
for i, sequence in enumerate(sequences_to_sort):  
    modify_graph = False
    root_dataset = dir_splitted
    #cam = r"nancy2"
    
    
    dataset = os.path.join(root_dataset, str(sequence))
    images_dataset = join(dataset, 'images')
    labels_dataset = join(dataset, 'labels')
    root_cs = join(dataset,r'labels_ord')    
    name_poset_vv = r'poset_vv.pickle'
    name_poset_snowsurface =r'poset_snow_surface.pickle'
    name_poset_snowheight = r'poset_snow_height.pickle'
    
    #%def graph paths
    suffixe = get_suffixe(critere,subgroup,mode)        
    graphs_names = [os.path.join(graph + suffixe+r'.gpickle') for graph in ["dg","ug","eg"]]
    graphs_paths = [os.path.join(root_cs, graph_name) for graph_name in graphs_names]
  
    
    for name in os.listdir(images_dataset):
        if 'Copie' in name:
            modify_graph = True
    
    if modify_graph:
        
        dg, ug, eg = get_graphs(graphs_paths)
        dg2 = get_dg2vvday(root_cs)  
    
        for name in os.listdir(images_dataset):
            if 'Copie' in name:

                if name in doublons:
                    print(name + ' was in doublons')
                    dg.remove_node(name)
                    ug.remove_node(name)
                    eg.remove_node(name)
                    dg2.remove_node(name)
                else:
                    correct_name = name.split(' - ')[0] + '.jpg'
                    mapping = {name:correct_name}
                    print(sequence, name, len(dg.nodes))
                    dg = nx.relabel_nodes(dg, mapping)
                    ug = nx.relabel_nodes(ug, mapping)                
                    eg = nx.relabel_nodes(eg, mapping)
                    dg2 = nx.relabel_nodes(dg2, mapping)
                    print(name, len(dg.nodes))
                    
        save_graphs((dg, ug, eg), graphs_paths)
        save_dg2vvday(dg2, root_cs)
        
#%% Vérification:

for i, sequence in enumerate(sequences_to_sort):  
    modify_graph = False
    root_dataset = dir_splitted
    #cam = r"nancy2"
    
    
    dataset = os.path.join(root_dataset, str(sequence))
    images_dataset = join(dataset, 'images')
    labels_dataset = join(dataset, 'labels')
    root_cs = join(dataset,r'labels_ord')    
    
    #%def graph paths
    suffixe = get_suffixe(critere, subgroup, mode)        
    graphs_names = [os.path.join(graph + suffixe + r'.gpickle') \
                    for graph in ["dg","ug","eg"]]
    graphs_paths = [os.path.join(root_cs, graph_name) \
                    for graph_name in graphs_names]
  
    
    
      
    dg, ug, eg = get_graphs(graphs_paths)
    dg2 = get_dg2vvday(root_cs)                    
    
    for g in [dg, ug, eg, dg2]:
        for n in g.nodes:
            if "Copie" in n:
                print(n)                
        
#%% renommer les images 
for i, sequence in enumerate(sequences_to_sort):  #[-42:-41]:
    
    root_dataset = dir_splitted
    #cam = r"nancy2"
    
    
    dataset = os.path.join(root_dataset, str(sequence))
    images_dataset = join(dataset, 'images')
    labels_dataset = join(dataset, 'labels')
    root_cs = join(dataset,r'labels_ord')    

    
    for name in os.listdir(images_dataset):
        if 'Copie' in name:
            print('sequence: ' + str(sequence), i, name)
            
            correct_name = name.split(' - ')[0] + '.jpg'
            
            
            if name in doublons:
                print("delete name")
                os.remove(join(images_dataset, name))
            else:
                src = join(images_dataset, name)
                dst = join(images_dataset, correct_name)
                print("move " + src + " to " + dst)
                shutil.move(src,dst)
                
        
#%% Vérif que correct_lbls contient toutes les images:
for i, sequence in enumerate(sequences_to_sort):  #[-42:-41]:
    
    root_dataset = dir_splitted
    #cam = r"nancy2"
    
    
    dataset = os.path.join(root_dataset, str(sequence))
    images_dataset = join(dataset, 'images')
    labels_dataset = join(dataset, 'labels')
    root_cs = join(dataset,r'labels_ord')    

    seq_names = set([n for n in corrected_lbls if corrected_lbls[n]['sequence'] == sequence])
    
    if not seq_names == set(os.listdir(images_dataset)):
        print(seq)              

#%% sauvegarde de corrected_labels:
        
label_path2 = os.path.join(root, 
                    r'labels_imagewise_BAMOSvv_corr2.pickle')


with open(label_path2, 'wb') as f:
    pickle.dump(corrected_lbls, f)

          
#%% Fabrication des ug2:
def save_ug2vvday(ug2, root_cs):
  
    path_of_ug2 = os.path.join(root_cs, "posetbar_vvday" + r".gpickle" )  
    nx.write_gpickle(ug2, path_of_ug2)
    print("graph ug2 saved")

        

counter = 0

for i, sequence in enumerate(sequences_to_sort):  
    modify_graph = False
    root_dataset = dir_splitted
    #cam = r"nancy2"
    
    
    dataset = os.path.join(root_dataset, str(sequence))
    images_dataset = join(dataset, 'images')
    labels_dataset = join(dataset, 'labels')
    root_cs = join(dataset,r'labels_ord')    
    name_poset_vv = r'poset_vv.pickle'
    name_poset_snowsurface =r'poset_snow_surface.pickle'
    name_poset_snowheight = r'poset_snow_height.pickle'
    
    #%def graph paths
    suffixe = get_suffixe(critere,subgroup,mode)        
    graphs_names = [os.path.join(graph + suffixe+r'.gpickle') for graph in ["dg","ug","eg"]]
    graphs_paths = [os.path.join(root_cs, graph_name) for graph_name in graphs_names]
  
        
    dg, ug, eg = get_graphs(graphs_paths)
    dg2 = get_dg2vvday(root_cs)  
    
    # test que dg2 est bien construit 
    if len(set(dg2.nodes) - set(dg.nodes)) > 0:
        print(sequence)
    # (en fait, pour trois séquences, ça a merdouillé 
    # - sans doute manip de decomp. dans ce cas, on ne ug2 = ug)
    
    if len(nx.transitive_closure(dg2).edges) < len(nx.transitive_closure(dg).edges):
        print(sequence)
        ug2 = ug
        dg2 = dg
        save_dg2vvday(dg2, root_cs)
        save_ug2vvday(ug2, root_cs)
        
    else:
        tdg = nx.transitive_closure(dg)
        tdgu = tdg.to_undirected()
        ug2_ = nx.complement(tdgu)
        
        # vérif que ug2_ contient bien ug (aux auto edges près)
        for e in ug.edges:
            if not ug2_.has_edge(e[0], e[1]) and (e[0] != e[1]):
                print('aië')      
        
        
        # ajout des edges de ug2_
        new_ug_edges = set([(e[0], e[1], -1) for e in ug2_.edges\
                         if not ug.has_edge(*e)])
        counter += len(new_ug_edges)
        print(len(new_ug_edges))
        
        ug.add_weighted_edges_from(new_ug_edges)
        
        save_ug2vvday(ug, root_cs)
        
print(counter)

#%% Vérification que les arêtes additionnelles
# sont correctes:

liste_verif = [('00010618_6_20161209_150114.jpg', '00010618_6_20161201_103013.jpg', -1),
('00010621_0_20120109_143302.jpg', '00010621_0_20140621_100310.jpg', -1),
('00010628_7_20120107_090956.jpg', '00010628_7_20120107_080955.jpg', -1),
('00010628_7_20120107_090956.jpg', '00010628_7_20120107_073955.jpg', -1),
('00010628_7_20120107_073955.jpg', '00010628_7_20120107_080955.jpg', -1),
('00010657_7002_20130925_103845.jpg', '00010657_7002_20130721_180838.jpg', -1),
('00010660_36_20140417_054134.jpg', '00010660_36_20140524_184135.jpg', -1),
('00010726_5_20150518_091628.jpg', '00010726_5_20150415_174635.jpg', -1),
('00010853_0_20100812_170528.jpg', '00010853_0_20110128_095112.jpg', -1),
('00010853_0_20110508_142110.jpg', '00010853_0_20110710_085110.jpg', -1),
('00010871_0_20110120_123852.jpg', '00010871_0_20110126_113854.jpg', -1),
('00010874_19_20131211_121201.jpg', '00010874_19_20131220_124158.jpg', -1),
('00010926_53_20140521_180312.jpg', '00010926_53_20140609_173316.jpg', -1),
('00011287_0_20100504_022356.jpg', '00011287_0_20100401_201949.jpg', -1),
('00011287_0_20100305_211757.jpg', '00011287_0_20100522_171919.jpg', -1),
('00011351_1_20150121_220130.jpg', '00011351_1_20161005_210131.jpg', -1),
('00011351_1_20150121_220130.jpg', '00011351_1_20170107_170127.jpg', -1),
('00011377_2_20100718_170031.jpg', '00011377_2_20100531_140021.jpg', -1),
('00011385_10001_20150418_120436.jpg', '00011385_10001_20150430_153441.jpg', -1),
('00011385_10001_20150418_120436.jpg', '00011385_10001_20150504_203437.jpg', -1)]
compas = []
for e in liste_verif:
    name0 = e[0]
    name1 = e[1]
    sequence = e[0].split('_')[0] + '_' + e[0].split('_')[1]
    dataset = os.path.join(root_dataset, sequence)
    images_dataset = join(dataset, 'images')
    compas.append(compare(name0, name1, mode, images_dataset, critere))
#%% images fully noisy

root = r"D:\sniff_webcams\AMOS"

dir_noisy_images = os.path.join(root,
                          r"full_noisy_images")



# 1: Ajout de toutes les images annotées 'f' dans bamosvv2 dans le pool
for name_img in corrected_lbls:
    if 'f' in corrected_lbls[name_img]['levelvv']:
        sequence = name_img.split('_')[0] + '_' + name_img.split('_')[1]
        seq = corrected_lbls[name_img]['sequence']
        src = join(dir_splitted, seq, 'images', name_img)
        dst = join(dir_noisy_images, name_img)
        shutil.copy(src,dst)

# à la main: sélection des images annotées 'f' universelles (ie: ne comportant aucune info, et ne correspondant pas à un changement de scène)
# -> nuit noires, et surtout caméras "bouchées".
#%% Construction de BAMOSvv3: images/graphe        
dir_bamos = os.path.join(root,
                            'BAMOSvv3')
dir_bamos_images = join(dir_bamos, 'images')
     
for sequence in os.listdir(dir_splitted):
    dir_sequence = join(dir_splitted, sequence, 'images')

    
    images = os.listdir(dir_sequence)
    
    for name_image in images:
        src = join(dir_sequence, name_image)
        dst = join(dir_bamos_images, name_image)
        shutil.copy(src, dst)
# 5374 images
#%% 2) pool_BAMOSvv2 -> BAMOSvv3
        

dir_pool_BAMOSvv = os.path.join(root,
                            'pool_BAMOSvv2')
     
for sequence in os.listdir(dir_pool_BAMOSvv):
    dir_pool_images = join(dir_pool_BAMOSvv, sequence, 'images')
    dir_sequence = join(dir_splitted, sequence, 'images')
    
    try:
        images = os.listdir(dir_sequence)
    except:
        images = []
    
    for name_image in os.listdir(dir_pool_images):
        if (not 'Copie' in name_image) and (name_image not in images):
            src = join(dir_pool_images, name_image)
            dst = join(dir_bamos_images, name_image)
            shutil.copy(src, dst)

#%% 3) full noisy images -> BAMOSvv3
dir_noisy_images = os.path.join(root,
                          r"full_noisy_images")
     
for name_image in os.listdir(dir_noisy_images):
    src = join(dir_noisy_images, name_image)
    dst = join(dir_bamos_images, name_image)
    shutil.copy(src, dst)            

# 41736 images
#%% Construction des graphs complets:

# 1) Aggrégation des graphes venant de BAMOSvv2:

tdg = nx.DiGraph()
tug = nx.Graph()
teg = nx.Graph()

for sequence in os.listdir(dir_splitted):
    try:
        dir_labels = os.path.join(dir_splitted,sequence,'labels_ord')
        # attention: poset_vvday a perdu les poids. Mais ils sont dans dg_vv
        # qui contient la même information
        dg_path = os.path.join(dir_labels, 'dg_vvday.gpickle')
        ug_path = os.path.join(dir_labels, 'posetbar_vvday.gpickle')
        eg_path = os.path.join(dir_labels, 'eg_vvday.gpickle')
        
        local_dg = nx.read_gpickle(dg_path) 
        local_ug = nx.read_gpickle(ug_path)
        local_eg = nx.read_gpickle(eg_path)
    #            print(len(local_ug.edges))
        
        tdg = nx.compose(tdg, local_dg)
        tug = nx.compose(tug, local_ug)    
        teg = nx.compose(teg, local_eg) 
        
        
    except:
        pass
        print('nothing in sequence: ' + str(sequence))
       
            

#%% Contrôle:
        
print(len(tdg.edges), len(tug.edges), len(teg.edges)) 


#washing
two_cycles = set(tdg.edges).intersection(invert_edges(set(tdg.edges)))
two_cycles = edges_and_inverted_edges(two_cycles)
print(str(len(two_cycles)) + ' 2-cycles')


tdg.remove_edges_from(two_cycles)
tdg = nx.transitive_closure(tdg)
two_cycles = set(tdg.edges).intersection(invert_edges(set(tdg.edges)))
two_cycles = edges_and_inverted_edges(two_cycles)
print(str(len(two_cycles)) + ' 2-cycles after washing')


vs03 = edges_and_inverted_edges(tug.edges).intersection(set(tdg.edges))
print(str(len(vs03)) + ' 0vs3')


vs03eg = {edge for edge in vs03 if edge in teg.edges}
vs03 = edges_and_inverted_edges(tug.edges).intersection(set(tdg.edges))
tdg.remove_edges_from(vs03)
tug.remove_edges_from(vs03)
teg.remove_edges_from(vs03eg)
print(str(len(vs03)) + ' 0vs3 after washing')
print(len(tdg.edges), len(tug.edges), len(teg.edges)) 


# 26841 12489 3373
#%% Ajout des edges "noisy"
images = sorted(os.listdir(dir_bamos_images))
new_edges = []
for noisy_image in os.listdir(dir_noisy_images):
    sequence = noisy_image.split('_')[0]
    same_seq_images = [img for img in images if img.split('_')[0] == sequence]
    new_edges += [(noisy_image, img, -2) for img in same_seq_images]
    
print(len(new_edges)) #141138
tug.add_weighted_edges_from(new_edges)
#%% contrôle
vs03 = edges_and_inverted_edges(tug.edges).intersection(set(tdg.edges))
print(str(len(vs03)) + ' 0vs3')
tdg.remove_edges_from(vs03)
#%%
print(len(tdg.edges), len(tug.edges), len(teg.edges)) 
#26839 150096 3373
#%% Enregistrement des graphes
    
path_tdg = os.path.join(dir_bamos, 'tdg_bamosvv_041123')
path_tug = os.path.join(dir_bamos, 'tug_bamosvv_041123')
path_teg = os.path.join(dir_bamos, 'teg_bamosvv_041123')

nx.write_gpickle(tdg, path_tdg)
nx.write_gpickle(tug, path_tug)
nx.write_gpickle(teg, path_teg) 
    