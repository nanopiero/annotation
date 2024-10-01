#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:02:37 2021
but: sélectionner des images de chaque dossier et les passer rapidement
pour dire si:
sol/pas sol  change/pas change
    
@author: plepetit
"""



import random
import sys
import threading
from threading import Thread
import time
import shutil
import os
from os.path import join, isdir, isfile
import urllib
#import cdsapi
#import netCDF4
#from netCDF4 import Dataset
#import urllib.request
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import datetime
from datetime import timedelta, datetime, date
from matplotlib.pyplot import plot_date
from matplotlib.dates import drange
from PIL import Image, ImageTk
from PIL.Image import open as imopen
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas
import numpy as np
import pickle
import zipfile
import re
import json 
from datetime import timedelta, datetime, date

#path_code = r'C:\Users\Deep Yawner\Desktop\tri_images\training_on_AMOS\src'
#os.chdir(path_code)

#from utile_mtl import *
#from fts_utiles_comb import *
#from transforms_AMOS import *
#import networkx as nx
#import time
#%




#%%Local
root = r"C:\Users\Deep Yawner\Desktop\poste_dso\AMOS_exemple"
root = r"D:\sniff_webcams\AMOS"
#root = r"/ssd/lepetit/neige/datasets"



old_corpus_1_events = os.path.join(root, 'pseudo_inovv_splitted')
corpus_1_events = os.path.join(root, 'inovv')

print(len(os.listdir(old_corpus_1_events)))

#%%
for seq in os.listdir(old_corpus_1_events):
    imgs = os.listdir(join(old_corpus_1_events, seq))
    print(len(imgs))
    if len(imgs) > 5:
        break

 
#%% D'abord, refaire des sous-rep:
def mkdir(path):
    if not isdir(path):
        os.mkdir(path)
        
mkdir(corpus_1_events)

for i, seq in enumerate(os.listdir(old_corpus_1_events)):
    newseq = seq #str(i)
    subseq = seq.split('_')[1]
    mkdir(join(corpus_1_events, newseq))
    mkdir(join(corpus_1_events, newseq, '0'))
    for file in os.listdir(join(old_corpus_1_events, seq)):
        src = join(old_corpus_1_events, seq, file)
        dest = join(corpus_1_events, newseq, '0', file)
        shutil.copy(src, dest)
    

#%%


corpus = corpus_1_events

def format(seq):
    l = seq.split('_')
    l = l[0] + codesubseq  + codeday
    return l


seqs = sorted([(seq) for seq in  os.listdir(corpus)])

dict_path = os.path.join(root, 'labels_inovv', 'labels_settlings.pickle')

#%%tidy the reps:
"""
def clean_rep(seq_path):
    subseqs = os.listdir(seq_path)
    prev_names = []
    prev_subseq_path = ''
  
    
    
    for subseq in subseqs:
        subseq_path = os.path.join(seq_path, subseq)
        names = os.listdir(subseq_path)
        if names[0] in prev_names:
            for name in names:
                shutil.copy(os.path.join(subseq_path, name), prev_subseq_path)
            prev_names += names
            shutil.rmtree(subseq_path)
        else:
            prev_names = names
            prev_subseq_path = os.path.join(seq_path, subseq)
            
         

for seq in seqs:
    seq_path = os.path.join(corpus, str(seq))
    clean_rep(seq_path)

"""

#%%
from tkinter import *



seqn= len(seqs)
seq0 = seqs[0]
seqpath0 = os.path.join(corpus, str(seq0))
subseqs0 = sorted(os.listdir(seqpath0))
subseqn = len(subseqs0)
subseq0 = subseqs0[0]
subseqpath0 = os.path.join( seqpath0, subseq0)
imgnames0 = sorted(os.listdir(subseqpath0))
imgsn = len(imgnames0)
imgpaths0 = [os.path.join( subseqpath0, name) for name in imgnames0]

w,h = 1000, 800 #width and height


labels_subseq = {i:{0:''} for i in seqs}
#labels_seq    = {i:{} for i in seqs}
#
#with open(dict_path, 'wb') as f:
#    pickle.dump(labels_subseq,f)

with open(dict_path, 'rb') as f:
    labels_subseq = pickle.load(f)
    

def find_start(labels_subseq):
    for i,seq in enumerate(sorted(labels_subseq.keys())):
        if labels_subseq[seq].get(0) == '':
            break
    return i
            

def load_seq():
    global i,j,k, imgs, labelsubseq, imgsn, subseqn
    seq = seqs[i]
    seqpath = os.path.join(corpus, str(seq))
    subseqs = sorted(os.listdir(seqpath))
    imgs = {}
    print(seq)
    for j in range(len(subseqs)):
        subseq = subseqs[j]
        subseqpath = os.path.join( seqpath, subseq)
        imgnames = sorted(os.listdir(subseqpath))
        imgpaths = [os.path.join( subseqpath, name) for name in imgnames]
        imgs[j] = [ImageTk.PhotoImage(imopen(imgpath).resize((w,h)), master = root) for imgpath in imgpaths]

#    print(imgs)
    j=0
    k=0

    subseqn = len(subseqs)    
    imgsn = len(imgs[j])
    
    texti.set("seq n°" + str(i+1)  + ' / ' + str(seqn))
    textj.set("subseq n°" + str(j+1) + ' / ' + str(subseqn))
    textk.set("img n° "+ str(k+1)  + ' / ' + str(imgsn) )   
    textss.set(str(labels_subseq[seq].get(int(subseqs[0]))))
    canvas.itemconfig(image_id, image=imgs[j][k])


def save_labels():
    with open(dict_path, 'wb') as f:
        pickle.dump(labels_subseq,f) 
    print('labels saved')

def change_seq():
    global i
    i+=1
    load_seq()

def back_seq():
    global i
    i-=1
    load_seq()
    
def change_subseq():
    global i,j,k, imgs, imgsn, subseqn
    
    # next image
    j += 1
    print(j)
    # return to first image
    if j == subseqn:
        print('changing seq')
        j = 0
        change_seq()
        save_labels()
    else:
        print('changing subseq')
        k=0
        imgsn = len(imgs[j])
        textj.set("subseq n°" + str(j+1) + ' / ' + str(subseqn))
        textk.set("img n° "+ str(k+1)  + ' / ' + str(imgsn) )
        textss.set(str(labels_subseq[seqs[i]].get(j)))
        canvas.itemconfig(image_id, image=imgs[j][k])

def change_img():
    global j,k, imgs, imgsn
    
    # next image
    k += 1
    print(k)
    # return to first image
    if k == len(imgs[j]):
        print('end')
        k = 0

    canvas.itemconfig(image_id, image=imgs[j][k])
    textk.set("img n° "+ str(k+1)  + ' / ' + str(imgsn) )  

def a(event):
    change_img()

def z(event):
    global i
    change_subseq()
    print(i)
    
def e(event):
    change_seq()

def r(event):
    back_seq()

def s(event):
    save_labels()

#0 grow 1 melts 3=incomp  2=eq  4=no snow   5=incomp noisy  6=eq noisy  7=changing scene  8=2 surexp  9=3 surexp  10=complex

def p0(event):
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '0'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('0000')

def p1(event):
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '1'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('1111')
def p2(event):
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '2'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('2222')

def p4(event):  #2 0 neige
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '4'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('4444 2 and no snow')

def p3(event):
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '3'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('3333')

def p5(event):   #5 noisy
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '5'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('5555 noisy')

def p6(event):   #6 2 noisy
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '6'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('6666 2 noisy')

def p7(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '7'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('7777 3 changing scene')

def p8(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '8'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('8888 2 surexp')
    
def p9(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += '9'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('9999 3 surexp')

def pc(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'c' 
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('c visi croissante')
    
def pd(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'd'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('d visi décroissante')

def pi(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'i'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('i inclusion croissante')
    
    
def pj(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'j'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('j inclusion décroissante')

def pk(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'k'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('k incomparabilité sans ordre')

def pg(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'g'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('g égalité')

    
def po(event):   
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'o'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('ombres au sol')

def pb(event):   
    global labels_subseq, i , j
    try:
        labels_subseq[seqs[i]][j] = labels_subseq[seqs[i]][j][:-1] 
    except:
        print('impossible de supprimer un label')
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('suppr. dernier label')


def pm(event):   #mixt
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'm'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('m mixte, pas de label clair')



def pn(event):   #mixt
    global labels_subseq, i , j
    labels_subseq[seqs[i]][j] += 'n'
    textss.set(str(labels_subseq[seqs[i]].get(j)))
    print('n  image nuit ')
#%%
    

    
root = Tk()
#init:
i =find_start(labels_subseq) 
imgs = {0:[ImageTk.PhotoImage(imopen(imgpath).resize((w,h)), master = root) for imgpath in imgpaths0]}

j = 0
k = 0



#p = PanedWindow(fenetre, orient=VERTICAL)
#p.pack(side=TOP, expand=Y, fill=BOTH, pady=2, padx=2)
#p.add(Label(p, text='Volet 1', background='blue', anchor=CENTER))
#p.add(Label(p, text='Volet 2', background='white', anchor=CENTER) )
#p.add(Label(p, text='Volet 3', background='red', anchor=CENTER) )
#p.pack()


frame0 = Frame(root, borderwidth=2, relief=GROOVE)
frame0.pack(side = TOP, padx=10, pady=0)

frame1 = Frame(root, borderwidth=2, relief=GROOVE)
frame1.pack(side=BOTTOM, padx=10, pady=0)

frame2 = Frame(root, borderwidth=2, relief=GROOVE)
frame2.pack(side=BOTTOM, padx=10, pady=0)
frame2.grid_columnconfigure(3,weight=1)

canvas = Canvas(frame1,  width=w, height=h)
canvas.pack()


texti = StringVar()
textj = StringVar()
textk = StringVar()
textss = StringVar()


#textss = StringVar()

li = LabelFrame(frame2, padx=20, pady=20)
li.grid(row=0,column=0)
#li.pack(padx=0, pady=0)
labi = Label(li, textvariable=texti, bg="yellow")
labi.pack(padx=50, pady=20)


lj = LabelFrame(frame2,  padx=50, pady=20)
lj.grid(row=0,column=1)
labj= Label(lj, textvariable=textj, bg="yellow")
labj.pack( padx=50, pady=20)

lk = LabelFrame(frame2,  padx=20, pady=20)
lk.grid(row=0,column=2)
labk = Label(lk, textvariable=textk , bg="yellow")
labk.pack(padx=50, pady=20)


texti.set("seq n°" + str(i+1)  + ' / ' + str(seqn) + ' (' + seqs[i] + ')')
textj.set("subseq n°" + str(j+1) + ' / ' + str(subseqn))
textk.set("img n° "+ str(k+1)  + ' / ' + str(imgsn) )

textss.set(str(labels_subseq[seqs[i]].get(j)))

#lss = LabelFrame(root,  padx=50, pady=20)
#lss.pack( side = TOP, padx=50, pady=20)
labss = Label(frame0, textvariable=textss , bg="yellow")
labss.pack(side = TOP, padx=50, pady=20)


#Label(l, text="A l'intérieure de la frame").pack()
#
root.bind("<KeyPress-0>",p0)
root.bind("<KeyPress-1>",p1)
root.bind("<KeyPress-2>",p2)
root.bind("<KeyPress-3>",p3)
root.bind("<KeyPress-4>",p4)
root.bind("<KeyPress-5>",p5)
root.bind("<KeyPress-c>",pc)
root.bind("<KeyPress-d>",pd)
root.bind("<KeyPress-i>",pi)
root.bind("<KeyPress-j>",pj)
root.bind("<KeyPress-k>",pk)
root.bind("<KeyPress-o>",po)
root.bind("<KeyPress-g>",pg)
root.bind("<KeyPress-b>",pb)
root.bind("<KeyPress-n>",pn)
root.bind("<m>",pm)


root.bind("<a>",a)
root.bind("<z>",z)
root.bind("<e>",e)
root.bind("<s>",s)
root.bind("<r>",r)

#root.bind("<e>",e) #"ground ok"
#root.bind("<r>",r) #"no ground"
#

#root.bind("<4>",p4)
#root.bind("<5>",p5)


    
# set first image on canvas
image_id = canvas.create_image(0, 0, anchor='nw', image= imgs[0][0])


load_seq()


root.mainloop()

#%%
#i =75
print(i)
print(seqs[i])
print(labels_subseq[seqs[i]])
    
    
