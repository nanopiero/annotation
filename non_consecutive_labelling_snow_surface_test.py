# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:59:32 2020

@author: Deep Yawner
"""

import random
import sys
import threading
from threading import Thread
import time
import shutil
import os
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
import pickle
import subprocess
import pytz
#from tzwhere import tzwhere
import zipfile
#tzwhere = tzwhere.tzwhere()
import re
import json 
from datetime import timedelta, datetime, date

path_code = r'C:\Users\Deep Yawner\Desktop\tri_images\prepare_dataset\ordinal_labeling'
os.chdir(path_code)

from utile import *
from utile_to_get_the_graph import *

path_code = r'C:\Users\Deep Yawner\Desktop\tri_images\prepare_dataset\POSET_mergesort'
os.chdir(path_code)
from fts_utiles_tri5_et_test import *


#%% 

#%Local
root = r"D:\sniff_webcams\datasets"

#hal3
#root = r"/ssd/lepetit/neige"

train_dir = os.path.join(root,r"AMOSDIR_16000")
#train_dir = r"D:\sniff_webcams\datasets\AMOSDIR_17000"
val_dir = os.path.join(root,r"VAL")
test_dir = os.path.join(root, r"TEST")
models_dir  = os.path.join(root,r"models")
figures_dir = os.path.join(root,r"figures")
experiences_dir = os.path.join(root,r"experiences")

dic_of_dirs = {'train': train_dir,
               'val':val_dir,
               'test': test_dir,
               'models': models_dir,
               'figures':figures_dir,
               'experiences':experiences_dir}

label_train_dir= os.path.join(train_dir,"labels") 
image_train_dir= os.path.join(train_dir,"images") 

label_val_dir = os.path.join(val_dir,"labels") 
image_val_dir= os.path.join(val_dir,"images") 

label_test_dir = os.path.join(test_dir,"labels") 
image_test_dir= os.path.join(test_dir,"images") 


#%%

label_dir = label_test_dir

lbls = {}

for name in os.listdir(label_dir):
#    try:
        
        lbl_fic = os.path.join(label_dir, name)
#        cam, amd, hm = re.split("_", name)
#        date = datetime.strptime(amd+hm[:-5],"%Y%m%d%H%M%S")

        lbls[name] = {}
        with open(lbl_fic) as x:
            lbl = json.load(x)
        
        for key in lbl.keys():
            lbls[name][key] = lbl[key]

lbls = lbls_in_jpg(lbls)

##%%
splitted_name ='TEST_splitted'
splitted_dir = os.path.join(root, splitted_name)



#%% Step 1: define critere and mode

level =  'sequence'
superframes = {lbls[name][level] for name in lbls}
 

critere = 'snow'
param = 'sh'
mode = 'surface'
modes = [mode]
subgroup = 'day'
times = ['day']

grounds = [r'snow_ground', r'snow_road', r'white_road', r'snow_ground_dry_road']

names_of_superframe = {}
for superframe in superframes:
    if critere == 'vv':
        names_of_superframe[superframe] = {name for name in lbls if (lbls[name][level] == superframe) and (lbls[name]['time'] in times) and lbls[name]['noise'] != r'miss_rec'}
    elif critere =='snow':
        names_of_superframe[superframe] = {name for name in lbls if (lbls[name][level] == superframe) and (lbls[name]['ground'] in grounds) and lbls[name]['noise'] != r'miss_rec'}
         

superframes_to_sort = {superframe for superframe in superframes if len(names_of_superframe[superframe])>=0}

superframes_to_sort = sorted(superframes_to_sort)
#%%step 2 Get the global sdg/ug/eg from AMOS lbls to prefill the frame graphs


graphs  = get_new_graphs(lbls)
tdg, tug, teg = get_sdg_from_AMOSlbls(graphs, lbls, param, mode, level = 'sequence')
tdg = nx.transitive_closure(tdg)


#old labels
print("Add old labels")
print(len(tdg.edges), len(tug.edges), len(teg.edges))

graphs= (tdg, tug, teg)
splitted_dir = os.path.join(root,splitted_name)




#new labels
print("Add new labels")
print(len(tdg.edges), len(tug.edges), len(teg.edges))

graphs= (tdg, tug, teg)
tdg, tug, teg = fill_graph_from_splitted(graphs, splitted_dir, ext =  'snowday_surface')
tdg = nx.transitive_closure(tdg)
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
tdg.remove_edges_from(vs03)
tug.remove_edges_from(vs03)
teg.remove_edges_from(vs03eg)

tdg = nx.transitive_closure(tdg)

vs03 = edges_and_inverted_edges(tug.edges).intersection(set(tdg.edges))
print(str(len(vs03)) + ' 0vs3 after washing')


#%%weak edges
#print("add w-edges")
#
# 
#wgraphs = get_new_graphs(lbls)
#intra_frame=True
#nb_by_frame = 1000000
#wdg,_,_ = new_get_wdg_from_AMOSlbls(wgraphs, param, mode, intra_frame =intra_frame, nb_by_frame = nb_by_frame, level = 'sequence')
#wdg = nx.transitive_closure(wdg)        
#
#
##washing
#vs01 = invert_edges(set(wdg.edges)).intersection(set(tdg.edges))
#vs01w = invert_edges(vs01)
#print(str(len(vs01)) + ' 0vs1 betweent dg and wdg')
#tdg.remove_edges_from(vs01)
#wdg.remove_edges_from(vs01w)
#
#
#
#vs03 = edges_and_inverted_edges(tug.edges).intersection(set(wdg.edges))
#vs02 = {edge for edge in vs03 if edge in teg.edges}
#print(str(len(vs03)) + ' 0vs3u between ug and wdg')
#print(str(len(vs02)) + ' 0vs2u between eg and wdg')
#wdg.remove_edges_from(vs03)
#tug.remove_edges_from(vs03)
#teg.remove_edges_from(vs02)
#vs03 = edges_and_inverted_edges(tug.edges).intersection(set(wdg.edges))
#vs02 = {edge for edge in vs03 if edge in teg.edges}
#print(str(len(vs03)) + ' 0vs3u between ug and wdg')
#print(str(len(vs02)) + ' 0vs2u between eg and wdg')
#
#
#
##fusion
#tdg = nx.compose(tdg,wdg)
#tdg = nx.transitive_closure(tdg)
#print(len(tdg.edges), len(tug.edges), len(teg.edges))



#%%Load model:
#arch = 'vgg16_scratch'
#nclasses = 3
#nchannels = 6
#
#device = torch.device("cuda:0")
#
#model_name = '2806_relative_012_'+'vv'+ '_' + 'AMOSDIR_16000' + '_day_' +arch +'_' + str(100)+ '_bm'
#try:
#    PATH = os.path.join(models_dir, model_name) 
#except:
#    raise NameError('model not trained')
#    
#
#model = charge_model(arch, nchannels, nclasses, models_dir)
#
#model.load_state_dict(torch.load(PATH))
#
#
#model = model.to(device)
#
#model.eval()

#%%step 3: restriction to the nodes

model = None
device = None



#%%
i=114
for superframe in superframes_to_sort[i-1:]:

    print('count :' + str(i))

    print('superfame: ' + str(superframe))

    
    
    dataset = os.path.join(splitted_dir, str(superframe))
    images_dataset = os.path.join(dataset, 'images')
    labels_dataset = os.path.join(dataset, 'labels')
    root_cs = os.path.join(dataset,r'labels_ord')    

  
    #init frame_lbls and dic of subgroup
    frame_lbls = {}
    for name in os.listdir(images_dataset):
            frame_lbls[name]= lbls[name]
    
    dic_of_subgroups = {}
    list_snow = [r'snow_ground', r'snow_road', r'white_road', r'snow_ground_dry_road']
    list_time = [r'day', r'inter', r'night', r'dark_night'] 
    
    if critere == 'vv':
        for time in list_time:
            dic_of_subgroups[time] = [name for name in frame_lbls if lbls[name]['time'] == time and frame_lbls[name]['noise']] # != r'miss_rec']
        dic_of_subgroups['all_times'] = [name for name in frame_lbls if frame_lbls[name]['time'] in ['day', 'inter']] # and frame_lbls[name]['noise'] != r'miss_rec']

    elif critere =='snow':
        for time in list_time:
            dic_of_subgroups[time] = [name for name in frame_lbls if frame_lbls[name][r'time'] == time and (frame_lbls[name][r'ground'] in list_snow or frame_lbls[name][r'old snow_traces'] in [r'ground'])]
        dic_of_subgroups['all_times'] = [name for name in frame_lbls if  (frame_lbls[name][r'ground'] in list_snow or frame_lbls[name][r'old snow_traces'] in [r'ground'])]

    
    #Step 4: copy the images in the temp_images_dir

#    passage à COREN_2019-01-23_08_40.jpj
    labeling_root= r"C:\Users\Deep Yawner\Desktop\tri_images"
    temp_images_dir= os.path.join(labeling_root, r"images")
    temp_labels_dir= os.path.join(labeling_root, r"labels")


    
    
    
    #set mode depends on what is already done
    
    name_poset_vv = r'poset_vv.pickle'
    name_poset_snowsurface =r'poset_snow_surface.pickle'
    name_poset_snowheight = r'poset_snow_height.pickle'
    
    #%def graph paths
    suffixe = get_suffixe(critere,subgroup,mode)       
    graphs_names = [os.path.join(graph + suffixe+r'.gpickle') for graph in ["dg","ug", "eg"]]
    graphs_paths = [os.path.join(root_cs,graph_name) for graph_name in graphs_names]


    #def nodes

    nodes = sorted(dic_of_subgroups[subgroup])



    print('labelling of ' + str(len(nodes)) + ' nodes')

    #clean images_dir
    for name in os.listdir(temp_images_dir):
        os.remove(os.path.join(temp_images_dir,name))

    #copy in the temp image dir
    for node in nodes:
        shutil.copy(os.path.join(dataset, 'images', node), os.path.join(temp_images_dir,node))


    
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
              'images_dir' : temp_images_dir,
              'test_model':False,
              'model':model,
              'device':device
              }
    if len(nodes) <= 25:
        decomposition =  labelling_mode(**kwargs)
    else:
        decomposition =  labelling_mode_without_dg2(**kwargs)
    i+=1
#%%
ntr = ['violay_2019-01-23_14_28.jpg' ]
clean_nodes(ntr, **kwargs)
#%%
for superframe in superframes_to_sort:

    print('count :' + str(i))

    print('superfame: ' + str(superframe))

    
    
    dataset = os.path.join(splitted_dir, str(superframe))
    images_dataset = os.path.join(dataset, 'images')
    labels_dataset = os.path.join(dataset, 'labels')
    root_cs = os.path.join(dataset,r'labels_ord')    

  
    #init frame_lbls and dic of subgroup
    frame_lbls = {}
    for name in os.listdir(images_dataset):
            frame_lbls[name]= lbls[name]

    list_snow = [r'snow_ground', r'snow_road', r'white_road', r'snow_ground_dry_road']
    times = [r'day', r'inter']  #en test, pour le no snow, on prend aussi inter 
 
    

#    passage à COREN_2019-01-23_08_40.jpj
    labeling_root= r"C:\Users\Deep Yawner\Desktop\tri_images"
    temp_images_dir= os.path.join(labeling_root, r"images")
    temp_labels_dir= os.path.join(labeling_root, r"labels")



    
    #%def graph paths
    suffixe = get_suffixe(critere,subgroup,mode)       
    graphs_names = [os.path.join(graph + suffixe+r'.gpickle') for graph in ["dg","ug", "eg"]]
    graphs_paths = [os.path.join(root_cs,graph_name) for graph_name in graphs_names]

    new_graphs_names = [os.path.join(graph + suffixe+'_complete'+r'.gpickle') for graph in ["dg","ug", "eg"]]
    new_graphs_paths = [os.path.join(root_cs,new_graph_name) for new_graph_name in new_graphs_names]



    dg = nx.read_gpickle(graphs_paths[0])
    ug = nx.read_gpickle(graphs_paths[1])
    eg = nx.read_gpickle(graphs_paths[2])


    nodes = sorted([name for name in frame_lbls if frame_lbls[name][r'time'] in times])
    
    snow = sorted([name for name in frame_lbls if frame_lbls[name][r'time'] in times and (frame_lbls[name][r'ground'] in list_snow)])
    no_snow = sorted([name for name in frame_lbls if frame_lbls[name][r'time'] in times and (frame_lbls[name][r'ground'] in ['dry_road', 'wet_road'] and frame_lbls[name][r'old snow_traces'] not in [r'ground', 'road'])])


    new_dg_edges = [(name_snow, name_no_snow, -1) for name_snow in snow for name_no_snow in no_snow]
    
    new_eg_edges  = []
    for i in range(len(no_snow)):
        for j in range(len(no_snow) - i -1):
            new_eg_edges.append((no_snow[i],no_snow[i+j+1], -1))
            
    new_ug_edges = new_eg_edges
    
    print(len(dg.nodes))
    print(len(dg.edges) , len(new_dg_edges))
    print(len(ug.edges) , len(new_ug_edges))
    print(len(eg.edges) , len(new_eg_edges))
    
    
    dg.add_weighted_edges_from(new_dg_edges)
    ug.add_weighted_edges_from(new_ug_edges)    
    eg.add_weighted_edges_from(new_eg_edges)   
    
    print(len([dg.edges[e] for e in dg.edges if dg.edges[e].get('weight') == -1]))

    nx.write_gpickle(dg, new_graphs_paths[0])
    nx.write_gpickle(ug, new_graphs_paths[1])
    nx.write_gpickle(eg, new_graphs_paths[2])


#%%
#        dg = tdg.subgraph(nodes).copy()
#        ug = tug.subgraph(nodes).copy()                
#        eg = teg.subgraph(nodes).copy()  
#        
#        #use the jpg as nodes
##        mapping = {node: json_to_jpg(node) for node in lbls}
##
##        dg = nx.relabel_nodes(dg, mapping)
##        ug = nx.relabel_nodes(ug, mapping)
##        eg = nx.relabel_nodes(eg, mapping)
#        
#        
#        #save the graphs
#        nx.write_gpickle(dg, graphs_paths[0])
#        nx.write_gpickle(ug, graphs_paths[1])
#        nx.write_gpickle(eg, graphs_paths[2])