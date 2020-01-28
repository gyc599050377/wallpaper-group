#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import argparse, cv2, os, sys, math, random

from pathlib import *

from matplotlib import pyplot as plt

from PIL import Image

from scipy import ndimage


# In[2]:


# p1:

# General Function:

def rec_crop(image,width,height): # crop a rectangular area 
    h,w = image.shape[:2]
    x=int(np.random.rand(1)*(w-width))
    y=int(np.random.rand(1)*(h-height))
    img_cropped = np.copy(image[y:y+height, x:x+width])
    return img_cropped

def translate_normal(image, W, H):
    (h,w)=image.shape[:2]
    times_H = int(H/h)+1
    times_W = int(W/w)+1  
    p1=image
    for j in range(1,times_H):
        p1=np.concatenate((p1,image))       
    p2=p1   
    for i in range(1,times_W):
        p2=np.concatenate((p2, p1),axis=1)   
    img_trans = p2[:H, :W] 
    return img_trans

def obli_crop(crop, theta): #theta is angle of oblique
    b=crop.shape[0]
    crop1=np.copy(crop)
    (h,w)=crop1.shape[:2]
    theta = math.radians(theta)
    tan = math.tan(theta)
    for i in range (0,h):
        crop1[i,0:int((h-i)/tan),:]=0    
    crop1=np.rot90(crop1,2)
    for i in range (0,h):
        crop1[i,0:int((h-i)/tan),:]=0            
    crop1=np.rot90(crop1,2)
    crop1 = crop1/255.
    return crop1

def translate_obli(pri_cell, W, H):
    h,w = pri_cell.shape[:2]
    rh = h
    start,end = 0,0
    rw_set = list(np.ones((h))*w)
    
    for j in range (0,h-1):
        for i in range (0,w-1):
            if np.sum(pri_cell[j,i,:])==0 and np.sum(pri_cell[j,i+1,:])!=0:
                start = i
            if np.sum(pri_cell[j,i,:])!=0 and np.sum(pri_cell[j,i+1,:])==0:
                end = i
            rw_set[j]=end-start-1
    rw = int(max(rw_set, key = rw_set.count)) #return most frequent number
    
    extra_times = int(w/rw)+1
    times_w=int(W/rw+2*extra_times)
    img = np.zeros((h, int(rw*times_w), 3))
    img[:h,:w,:]=np.copy(pri_cell)
    
    for i in range (0,h):
        for j in range (rw, rw*times_w-rw):
            if np.sum(img[i,j:(j+rw),:])==0 and np.sum(img[i,j-1,:])!=0:
                img[i,j:(j+rw),:]=img[i,(j-rw):j,:]
    start_w = rw*(int(w/rw)+1)
    img = img[:,start_w:int(rw*times_w-rw),:]

    rolled = np.copy(img)
    times_h=int(H/h+1)
    for t in range (0, times_h):
        rolled = np.roll(rolled, 2*rw-w, axis=1)
        img = np.vstack((img,rolled))

    img = img[:H,:W,:]  
    return img

def rhomb_crop(image):
    (h,w)=image.shape[:2]
    crop2=np.copy(image)
    for n in range (0,2):
        for i in range (0,int(h/2)):
            crop2[i,0:int(w/2-i*w/h),:]=0
            crop2[i,int(w/2+i*w/h):w,:]=0
        crop2=np.rot90(crop2,2)
    return crop2

def rhombic_unit(crop):  
    #roll and fill the four corner to create a unit that can be translated
    h,w=crop.shape[:2]
    roll0 = np.roll(np.roll(crop, round(h/2), axis=0), round(w/2), axis=1)
    merged=np.copy(crop)
    for i in range (0,h):
        for j in range (0,w):
            if np.sum(merged[i,j,:])==0:
                merged[i,j,:]=np.copy(roll0[i,j,:])
    return merged

def border(image): # vague the border
    h,w = image.shape[:2]
    for i in range (0,h-1):
        for j in range (1,w-1):
            if np.sum(image[i,j-1,:])!=0 and np.sum(image[i,j+1,:])!=0 and np.sum(image[i,j,:])==0:
                image[i,j,:] = np.copy(image[i,j+1,:])
            if np.sum(image[i-1,j,:])!=0 and np.sum(image[i+1,j,:])!=0 and np.sum(image[i,j,:])==0:
                image[i,j,:] = np.copy(image[i+1,j,:])
    for i in range (0,h):
        if np.sum(image[i,0,:])==0:
            image[i,0,:]=np.copy(image[i,1,:])
        if np.sum(image[i,w-1,:])==0:
            image[i,w-1,:]=np.copy(image[i,w-2,:])
    for j in range (0,w):
        if np.sum(image[h-1,j,:])==0:
            image[h-1,j,:]=np.copy(image[h-2,j,:])
    return image

def remove_border(img):
    copy = np.copy(img)
    for i in range (0,4):
        hh = copy.shape[:1]
        while np.sum(copy[0,:,:])==0:
            copy = np.copy(copy[1:hh])
#         copy = np.copy(copy[1:hh])
        copy = np.rot90(copy)
    return copy

def rhombic_edge(image):
    img = np.copy(image)
    h,w = img.shape[:2]
    times = 1
    for i in range (0,int(h/2)):
        for j in range (0,int(w/2)):
            if np.sum(img[i,j,:])==0:
                t=1
                i_h = i
                if i_h < h-1:
                    while np.sum(img[i_h+1,j,:])==0:
                        t=t+1
                        i_h = i_h+1
                    if t > times:
                        times = t
    for q in range(0,times):
        img_1 = np.roll(img,1,axis=0)
        for i in range (0,h):
            for j in range (0,w):
                if np.sum(img[i,j,:])==0:
                    img[i,j,:]=np.copy(img_1[i,j,:])
    return img

# Transformations:

def p1_rectangular(image,unit_w,unit_h,Width,Height): 
    rec_unit = rec_crop(image,unit_w,unit_h)
    rec = translate_normal(rec_unit,Width,Height) 
    return rec

def p1_square(image,side_length,Width,Height):  # only need side length
    squ_unit = rec_crop(image,side_length, side_length)
    squ = translate_normal(squ_unit,Width,Height) 
    return squ

def p1_oblique(image, unit_width, unit_height, angle, Width, Height):
     # Angle is for left bottom corner
    rec = rec_crop(image, unit_width, unit_height)  
    crop = obli_crop(rec, angle) 
    obli = translate_obli(crop, Width, Height)
    return obli

def p1_hexagonal(image, side_length, Width, Height): 
    # cropped rectangular width/height=sqrt(3)
    rec = rec_crop(image,int(side_length * math.sqrt(3)),side_length) 
    crop = obli_crop(rec,60) # Hexagonal is oblique with 60 degree corner
    hexa = translate_obli(crop,Width,Height)
    hexa = border(hexa) # take care of border of unit cell
    return hexa

def p1_rhombic(image, unit_width, unit_height, Width, Height):
    rec = rec_crop(image,unit_width, unit_height)
    crop = rhomb_crop(rec) 
    rhomb_unit = rhombic_unit(crop)
    rhomb = translate_normal(rhomb_unit,Width,Height) 
    rhomb = rhombic_edge(rhomb) # take care of image's wierd edge
    return rhomb


# In[3]:


# p2: 

# Specfic function:

def rec_unit_p2(crop):
    h=crop.shape[0]
    crop_o = np.copy(crop[:int(h/2),:,:])
    crop_r = np.rot90(crop_o,2)
    crop = np.vstack((crop_o,crop_r)) 
    return crop 

def obli_p2_unit(image):
    # make it a p2 unit:
    copy=np.copy(image)
    h,w=copy.shape[:2]
    x = 0
    for i in range (0,w):
        if np.sum(copy[0,i-1,:])==0 and np.sum(copy[0,i,:])!=0:
            x=i
    copy1 = np.zeros((h,w+x,3))
    copy1[:h,:w,:] = np.copy(copy)
    copy2=np.rot90(copy1,2)
    image=np.vstack((copy2,copy1))
    return image

def rhomb_crop_p2(crop):
    crop1=np.copy(crop)
    (b,a)=crop1.shape[:2]
    if a>b :
        w,h = a,b
    else :
        w,h=a, int(a/2)
    for i in range (0,int(h/2)):
        crop1[i,0:int(w/2-i*w/h),:]=0
        crop1[i,int(w/2+i*w/h):w,:]=0
    crop2=np.copy(crop1[:int(h/2),:w,:])
    crop3=np.rot90(crop2, 2)
    crop4=np.vstack((crop2,crop3))
    return crop4

# Transformations:

def p2_rectangular(image,unit_w,unit_h,Width,Height):
    crop = rec_crop(image,unit_w,unit_h)
    crop = rec_unit_p2(crop)
    rec = translate_normal(crop,Width,Height)   
    return rec

def p2_square(image,side_length,Width,Height):  # only input side length
    squ_crop = rec_crop(image,side_length,side_length)
    squ_unit = rec_unit_p2(squ_crop)
    squ = translate_normal(squ_unit,Width,Height)   
    return squ

def p2_oblique(image, unit_width, unit_height, angle, Width, Height):
    crop = rec_crop(image,unit_width,unit_height)
    crop_obli = obli_crop(crop,angle) # half the height
    unit = obli_p2_unit(crop_obli) # form a whole size image
    obli=translate_obli(unit,Width,Height)
    return obli  

def p2_hexagonal(image, side_length, Width, Height): 
    #cropped rectangular width/height=sqrt(3), and angle always equals to 60 degree
    rec = rec_crop(image,int(side_length*math.sqrt(3)),side_length) 
    crop_hexa = obli_crop(rec,60) 
    unit = obli_p2_unit(crop_hexa)
    hexa=translate_obli(unit,Width,Height)
    hexa=border(hexa)
    return hexa  

def p2_rhombic(image, unit_width, unit_height, Width, Height):
    crop = rec_crop(image, unit_width, unit_height)
    crop1 = rhomb_crop_p2(crop)
    rhomb_unit = rhombic_unit(crop1)
    rhomb = translate_normal(rhomb_unit,Width,Height) 
    rhomb = rhombic_edge(rhomb)
    
    return rhomb


# In[4]:


# pm: 
def unit_pm_hori(img,W,H):
    h,w = img.shape[:2]
    
    times = int(math.log2(H/h)+1)
    for i in range (0, times):
        img_1=np.copy(img)
        img_2 = cv2.flip(img_1,0,dst=None) 
        img = np.vstack((img_1,img_2))
    img = img[:H,:,:]
    return img

def unit_pm_verti(img,W,H):
    h,w = img.shape[:2]
    
    times = int(math.log2(W/w)+1)
    for i in range (0, times):
        img_1=np.copy(img)
        img_2 = cv2.flip(img_1,1,dst=None) 
        img = np.hstack((img_1,img_2))
    img = img[:W,:,:]
    return img

def pm_horizontal(image,unit_w,unit_h,Width,Height):
    crop_pm = rec_crop(image,unit_w,unit_h)

    hori_unit = unit_pm_hori(crop_pm,Width,Height)
    img_hori = translate_normal(hori_unit,Width,Height)
    return img_hori

def pm_vertical(image,unit_w,unit_h,Width,Height):
    crop_pm = rec_crop(image,unit_w,unit_h)

    verti_unit = unit_pm_verti(crop_pm,Width,Height)
    img_verti = translate_normal(verti_unit,Width,Height)
    return img_verti


# In[5]:


# pg:
def pg_hori(img,W,H):
    h,w = img.shape[:2]
    img_1=np.copy(img)
    
    times = int(W/w+1)
    for i in range (0, times):
        img_1 = cv2.flip(img_1,0,dst=None) 
        img = np.hstack((img,img_1))
    img = img[:W,:,:]
    return img

def pg_verti(img,W,H):
    h,w = img.shape[:2]
    img_1=np.copy(img)
    
    times = int(H/h+1)
    for i in range (0, times):
        img_1 = cv2.flip(img_1,1,dst=None) 
        img = np.vstack((img,img_1))
    img = img[:H,:,:]
    return img

def pg_horizontal(image,unit_w,unit_h,Width,Height):
    crop_pg=rec_crop(image,unit_w,unit_h)

    hori_unit_pg=pg_hori(crop_pg,Width,Height)
    img_hori_pg=translate_normal(hori_unit_pg,Width,Height)
    return img_hori_pg 

def pg_vertical(image,unit_w,unit_h,Width,Height):
    crop_pg=rec_crop(image,unit_w,unit_h)

    verti_unit_pg=pg_verti(crop_pg,Width,Height)
    img_verti_pg=translate_normal(verti_unit_pg,Width,Height)
    return img_verti_pg  


# In[6]:


# pmm:
def pmm(image,unit_w,unit_h,Width,Height):
    crop = rec_crop(image, unit_w, unit_h)

    s1 = cv2.flip(crop,1,dst=None) 
    s2 = np.hstack((crop,s1))
    s3 = cv2.flip(s2,0,dst=None) 
    s4 = np.vstack((s2,s3))
    img_pmm = translate_normal(s4, Width, Height)
    return img_pmm


# In[7]:


# pmg:
def pmg_horizontal(image,unit_w,unit_h,Width,Height):
    crop = rec_crop(image,unit_w,unit_h)
    h1 = np.rot90(crop,2)
    h2 = np.hstack((crop,h1))
    h3 = cv2.flip(h2,0,dst=None) 
    h4 = np.vstack((h2,h3))
    h5 = np.rot90(h4,2)
    h6 = np.hstack((h4,h5))
    img_pmg_h=translate_normal(h6,Width,Height)
    return img_pmg_h

def pmg_vertical(image,unit_w,unit_h,Width,Height):
    crop = rec_crop(image,unit_w,unit_h)
    v1 = np.rot90(crop,2)
    v2 = np.vstack((crop,v1))
    v3 = cv2.flip(v2,1,dst=None) 
    v4 = np.hstack((v2,v3))
    v5 = np.rot90(v4,2)
    v6 = np.vstack((v4,v5))
    img_pmg_v=translate_normal(v6,Width,Height)
    return img_pmg_v


# In[8]:


# cm: 
def rhomb_crop(image):
    crop1 = np.copy(image)
    
    (b,a)=crop1.shape[:2]
    w,h = a,b
    crop2=np.copy(crop1)
    for n in range (0,2):
        for i in range (0,int(h/2)):
            crop2[i,0:int(w/2-i*w/h),:]=0
            crop2[i,int(w/2+i*w/h):w,:]=0
        crop2=np.rot90(crop2,2)     
    return crop2

def cm_rec_hori(crop1): #crop a rhombic shape image 
    h = crop1.shape[0]
    img_1=np.copy(crop1[:int(h/2),:,:])
    img_2 = cv2.flip(img_1,0,dst=None) 
    img = np.vstack((img_1,img_2))        
    return img

def rhombic_unit(crop):  
    #roll and fill the four corner to create a unit that can be translated
    h,w=crop.shape[:2]
    roll0 = np.roll(np.roll(crop, round(h/2), axis=0), round(w/2), axis=1)
    merged=np.copy(crop)
    for i in range (0,h):
        for j in range (0,w):
            if np.sum(merged[i,j,:])==0:
                merged[i,j,:]=np.copy(roll0[i,j,:])
    return merged

def rhombic_edge(image):
    img = np.copy(image)
    h,w = img.shape[:2]
    times = 1
    for i in range (0,int(h/2)):
        for j in range (0,int(w/2)):
            if np.sum(img[i,j,:])==0:
                t=1
                i_h = i
                if i_h < h-1:
                    while np.sum(img[i_h+1,j,:])==0:
                        t=t+1
                        i_h = i_h+1
                    if t > times:
                        times = t
    for q in range(0,times):
        img_1 = np.roll(img,1,axis=0)
        for i in range (0,h):
            for j in range (0,w):
                if np.sum(img[i,j,:])==0:
                    img[i,j,:]=np.copy(img_1[i,j,:])
    return img

def cm_horizontal(image,unit_w,unit_h,Width,Height):
    crop_rec = rec_crop(image, unit_w, unit_h)
    img_cm_rec_hori = cm_rec_hori(crop_rec)
    img_cm_rhomb_hori = rhomb_crop(img_cm_rec_hori)
    unit_cm_hori = rhombic_unit(img_cm_rhomb_hori)
    cm_hori = translate_normal(unit_cm_hori,Width,Height)   
    cm_hori = rhombic_edge(cm_hori) # take care of edge of unit cells
    return cm_hori

def cm_vertical(image,unit_w,unit_h,Width,Height):
    crop_rec = rec_crop(image, unit_w, unit_h)
    crop_rec = np.rot90(crop_rec)
    # rotate to vertical direction and do the same procedures
    img_cm_rec_verti = cm_rec_hori(crop_rec)
    img_cm_rhomb_verti = rhomb_crop(img_cm_rec_verti)
    unit_cm_verti = rhombic_unit(img_cm_rhomb_verti)
    unit_cm_verti = np.rot90(unit_cm_verti)
    cm_verti = translate_normal(unit_cm_verti,Width,Height)   
    cm_verti = rhombic_edge(cm_verti)
    return cm_verti


# In[9]:


# pgg:
def pgg_rec(crop1): #crop a pgg rhombic shape image 
    (b,a)=crop1.shape[:2]
    w,h = a,b
    crop2=np.copy(crop1[:int(h/2),:w,:])
    crop3=np.rot90(crop2,2)
    crop4=np.vstack((crop2,crop3))
    return crop4

def pgg_rhombic_unit(rhombic):  
    #roll and fill the four corner to create a unit that can be translated
    h,w=rhombic.shape[:2]
    p1 = np.copy(rhombic[:int(h/2),:,:])
    p2 = np.copy(rhombic[int(h/2):h,:,:])   
    r_p1 = cv2.flip(p1,0,dst=None) # horizontal flip
    r_p2 = cv2.flip(p2,0,dst=None)
    r = np.vstack((r_p2,r_p1))      
    h,w=r.shape[:2]
    rolled = np.roll(np.roll(r, int(h/2), axis=0), int(w/2), axis=1)
    for i in range (0,h):
        for j in range (0,w):
            if np.sum(rhombic[i,j,:])==0:
                rhombic[i,j,:]=np.copy(rolled[i,j,:])
    return rhombic

def pgg(image,unit_w,unit_h,Width,Height):
    crop_rec = rec_crop(image, unit_w, unit_h)
    img_pgg_rec = pgg_rec(crop_rec)
    img_pgg_rhomb = rhomb_crop(img_pgg_rec)
    unit_pgg = pgg_rhombic_unit(img_pgg_rhomb)
    pgg = translate_normal(unit_pgg,Width,Height)
    pgg = rhombic_edge(pgg)
    return pgg


# In[10]:


# cmm:   when width = height, it would be cmm square
def rhombic_unit_cmm(crop):      
    h,w=crop.shape[:2]    
    if h==w:
        u1 = Image.fromarray(crop)
        u2 = u1.rotate(45)
        u3 = np.array(u2)
        for i in range (0,4):
            hh,ww = u3.shape[:2]
            while np.sum(u3[0,:,:])==0:
                u3 = np.copy(u3[1:hh])
            u3 = np.copy(u3[1:hh])
            u3 = np.rot90(u3)
        unit = np.copy(u3)
    else:    
    #roll and fill the four corner to create a unit that can be translated
        h,w=crop.shape[:2]
        rolled = np.roll(np.roll(crop, int(h/2), axis=0), int(w/2), axis=1)
        merged=np.copy(crop)
        for i in range (0,h):
            for j in range (0,w):
                if np.sum(merged[i,j,:])==0:
                    merged[i,j,:]=np.copy(rolled[i,j,:])
        unit = np.copy(merged)
    return unit

def rhomb_crop(image):
    (h,w)=image.shape[:2]
    
    crop2=np.copy(image)
    for n in range (0,2):
        for i in range (0,int(h/2)):
            crop2[i,0:int(w/2-i*w/h),:]=0
            crop2[i,int(w/2+i*w/h):w,:]=0
        crop2=np.rot90((crop2),2)
        
    return crop2

def cmm_rhomb(reg_rhomb):
    h,w = reg_rhomb.shape[:2]
    left = np.copy(reg_rhomb[:, :int(w/4), :])
    right = np.copy(reg_rhomb[:, int(3*w/4):w, :])
    top = np.copy(reg_rhomb[:int(h/4), int(w/4):int(3*w/4), :])
    under = np.copy(reg_rhomb[int(3*h/4):h, int(w/4):int(3*w/4), :])  
    c1 = np.copy(reg_rhomb[int(h/4):int(h/2), int(w/4):int(w/2), :])
    c2 = cv2.flip(c1,1,dst=None) 
    c3 = cv2.flip(c1,0,dst=None) 
    c4 = cv2.flip(c2,0,dst=None)
    b,a = c1.shape[:2]  
    central = np.copy(reg_rhomb[int(h/4):int(3*h/4), int(w/4):int(3*w/4),:])
    central [:b, -a:, :] = np.copy(c2)
    central [-b:, :a, :] = np.copy(c3)
    central [-b:, -a:, :] = np.copy(c4)
    middle = np.vstack((top, central, under))
    cmm_rhomb = np.hstack((left, middle, right))
    return cmm_rhomb

def cmm(image,unit_w,unit_h,Width,Height):
    crop_rec = rec_crop(image, unit_w, unit_h)
    rhomb = rhomb_crop(crop_rec)
    crop_rhomb_cmm = cmm_rhomb(rhomb)
    unit_cmm = rhombic_unit_cmm(crop_rhomb_cmm)
    img_cmm = translate_normal(unit_cmm,Width,Height)
    img_cmm = border_cmm(img_cmm)
    return img_cmm

def border_cmm(image): # vague the border
    h,w = image.shape[:2]
    for i in range (0,h-1):
        for j in range (1,w-1):
            if np.sum(image[i,j,:])==0:
                if np.sum(image[i,j-1,:])!=0 and np.sum(image[i,j+1,:])!=0:
                    image[i,j,:] = np.copy(image[i,j+1,:])
                if np.sum(image[i-1,j,:])!=0 and np.sum(image[i+1,j,:])!=0:
                    image[i,j,:] = np.copy(image[i+1,j,:])
                
    for i in range (1,h):
        for j in range (1,w-1): 
            if np.sum(image[i,j,:])==0:
                if np.sum(image[i-1,1,:])!=0 and np.sum(image[i,j-1,:])!=0:
                    image[i,j,:] = np.copy(image[i-1,j-1,:])
                if np.sum(image[i-1,j,:])!=0 and np.sum(image[i,j+1,:])!=0:
                    image[i,j,:] = np.copy(image[i-1,j+1,:])
                
    for i in range (0,h-1):
        for j in range (1,w-1): 
            if np.sum(image[i,j,:])==0:
                if np.sum(image[i+1,j,:])!=0 and np.sum(image[i,j+1,:])!=0:
                    image[i,j,:] = np.copy(image[i+1,j+1,:])
                if np.sum(image[i+1,j,:])!=0 and np.sum(image[i,j-1,:])!=0:
                    image[i,j,:] = np.copy(image[i+1,j-1,:])
                
    for i in range (1,h-1):
        for j in range (1,w):           
            if np.sum(image[i,j,:])==0:
                if np.sum(image[i-1,1,:])!=0 and np.sum(image[i,j-1,:])!=0:
                    image[i,j,:] = np.copy(image[i-1,j-1,:])
                if np.sum(image[i+1,j,:])!=0 and np.sum(image[i,j-1,:])!=0:
                    image[i,j,:] = np.copy(image[i+1,j-1,:])
                                       
    for i in range (1,h-1):
        for j in range (0,w-1): 
            if np.sum(image[i,j,:])==0:
                if np.sum(image[i+1,j,:])!=0 and np.sum(image[i,j+1,:])!=0:
                    image[i,j,:] = np.copy(image[i+1,j+1,:])
                if np.sum(image[i-1,j,:])!=0 and np.sum(image[i,j+1,:])!=0:
                    image[i,j,:] = np.copy(image[i-1,j+1,:]) 
                                       
    return image


# In[11]:


# p4:
def p4_unit(img):  
    p1 = np.copy(img)
    p2 = np.rot90(p1)
    p3 = np.hstack((p1,p2))
    p4 = np.rot90(p3,2)
    img_unit = np.vstack((p4,p3))
    return img_unit

def p4(image, side_length, Width, Height): # only input side_length
    crop_rec = rec_crop(image, side_length, side_length)
    unit = p4_unit(crop_rec)
    img_p4 = translate_normal(unit, Width, Height)
    return img_p4


# In[12]:


# p4m:
def p4m_unit(img):
    squ = np.copy(img)
    h,w = squ.shape[:2]
    for i in range (0,h):
        squ[i,0:int(w-i*w/h),:] = 0
    s1 = np.rot90(squ)
    s2 = cv2.flip(s1,1,dst=None)
    s3 = np.roll(s2,1,axis=1)
    s4 = np.roll(s2,-1,axis=1)
    for i in range (0,h):  # from here
        for j in range (0,w):
            if np.sum(squ[i,j,:])==0:
                squ[i,j,:]=np.copy(s2[i,j,:])
            if np.sum(squ[i,j,:])==0:
                squ[i,j,:]=np.copy(s3[i,j,:])
            if np.sum(squ[i,j,:])==0:
                squ[i,j,:]=np.copy(s4[i,j,:])    
    if np.sum(squ[h-1,w-1,:])==0:
        squ[h-1,w-1,:]= np.copy(squ[h-2,w-2,:])
    if np.sum(squ[h-1,0,:])==0:
        squ[h-1,0,:]= np.copy(squ[h-2,0,:])
    if np.sum(squ[h-1,w-1,:])==0:
        squ[0,w-1,:]= np.copy(squ[0,w-2,:])
    if np.sum(squ[h-1,w-1,:])==0:
        squ[0,0,:]= np.copy(squ[1,1,:])  # to here, need to be shorten
        
    s3 = cv2.flip(squ,1,dst=None)
    s4 = np.hstack((squ,s3))
    s5 = cv2.flip(s4,0,dst=None)
    s6 = np.vstack((s5,s4))
    return s6

def p4m(image,side_length,Width,Height):
    crop_rec = rec_crop(image, side_length, side_length)
    unit = p4m_unit(crop_rec)
    img_p4m = translate_normal(unit, Width, Height)   
    return img_p4m


# In[13]:


# p4g:
def p4g_unit(img):
    squ = np.copy(img)
    h,w = squ.shape[:2]
    for i in range (0,h):
        squ[i,0:int(w-i*w/h),:] = 0
    s1 = np.rot90(squ)
    s2 = cv2.flip(s1,1,dst=None)
    s3 = np.roll(s2,1,axis=1)
    s4 = np.roll(np.roll(s2,-1,axis=1),-1,axis=0)

    for i in range (0,h):
        for j in range (0,w):
            if np.sum(squ[i,j,:])==0:
                squ[i,j,:]=np.copy(s2[i,j,:])
            if np.sum(squ[i,j,:])==0:
                squ[i,j,:]=np.copy(s3[i,j,:])
    if np.sum(squ[h-1,w-1,:])==0:
        squ[h-1,w-1,:]= np.copy(squ[h-2,w-2,:])
    if np.sum(squ[h-1,0,:])==0:
        squ[h-1,0,:]= np.copy(squ[h-2,0,:])
    if np.sum(squ[h-1,w-1,:])==0:
        squ[0,w-1,:]= np.copy(squ[0,w-2,:])
    if np.sum(squ[h-1,w-1,:])==0:
        squ[0,0,:]= np.copy(squ[1,1,:])
    p1 = np.copy(squ)
    p2 = np.rot90(p1,3)
    p12 = np.hstack((p1,p2))
    p34 = np.rot90(p12,2)
    unit = np.vstack((p12,p34))
    
    return unit

def p4g(image, side_length, Width, Height):
    crop_rec = rec_crop(image, side_length, side_length)
    unit = p4g_unit(crop_rec)
    img_p4g = translate_normal(unit, Width, Height)
    return img_p4g


# In[14]:


# p3:
def translate_p3(pri_cell, W, H):
    h,w = pri_cell.shape[:2]
    rh=h
    rw = int(2*w/3)
    
    times_w=int(W/rw+(w/rw)+1)
    img = np.zeros((h, int(rw*times_w), 3))
    img[:h,:w,:]=np.copy(pri_cell)
    
    img_rolled = np.copy(img)
    for times in range (0,times_w+1):
        img_rolled = np.roll(img_rolled,rw,axis=1)
        for i in range (0,h):
            for j in range (0, rw*times_w):
                if np.sum(img[i,j,:])==0:
                    img[i,j,:] = np.copy(img_rolled[i,j,:]) 
    
    rolled = np.copy(img)
    times_h=int(H/h+1)
    for t in range (0, times_h):
        rolled = np.roll(rolled, 2*rw-w, axis=1)
        img = np.vstack((img,rolled))
        
    img = img[:H,:W,:] 
    return img 

def p3_unit(obli):   # fix the edge problem later
    obli = obli*255.
    r1 = np.rot90(obli)
    r1_im = Image.fromarray(np.uint8(r1)) # have to transfer with uint8 type
    r1 = np.array(r1_im)
    r2_im = r1_im.rotate(120)
    r2 = np.array(r2_im)
    r4 = np.copy(r2)

    r3_im = r2_im.rotate(120, expand=True)
    r3 = np.array(r3_im)
    for i in range (0,4):
        hh = r3.shape[0]
        while np.sum(r3[0,:,:])==0:
            r3 = r3[1:hh]
        r3 = np.rot90(r3)
    h3,w3 = r3.shape[:2]
    r3_top = r3[:int(h3/2)]
    r3_bottom = r3[-int(h3/2):]

    r124 = np.hstack((r2,r1,r4))
    h,w = r124.shape[:2]
    r3_resize = np.zeros((h,w,3))
    r3_resize[-int(h3/2):,:w3,:] = np.copy(r3_top)
    r3_resize[:int(h3/2),-w3:,:] = np.copy(r3_bottom)
    r3_resize1 = np.roll(r3_resize,1,axis=1)
    r3_resize2 = np.roll(r3_resize,-1,axis=1)

    for i in range (0,h):
        for j in range (0,w):
            if np.sum(r124[i,j,:])==0:
                r124[i,j,:]=np.copy(r3_resize[i,j,:])
            if np.sum(r124[i,j,:])==0:
                r124[i,j,:]=np.copy(r3_resize1[i,j,:])
            if np.sum(r124[i,j,:])==0:
                r124[i,j,:]=np.copy(r3_resize2[i,j,:])
    unit = obli_crop(r124, 60)

    return unit

def p3(image, unit_h, Width, Height):
    rec = rec_crop(image,int(unit_h*math.sqrt(3)),unit_h) 
    # width = height*sqrt(3)
    crop = obli_crop(rec, 60)
    unit_p3 = p3_unit(crop)
    obli_p3=translate_p3(unit_p3,Width,Height)
    obli_p3=border(obli_p3)
    return obli_p3


# In[15]:


# p3m1:
def rhomb_p3m1(crop):
    h,w = crop.shape[:2]
    for i in range (0,w):
        if np.sum(crop[0,i-1,:])==0 and np.sum(crop[0,i,:])!=0:
            x=i 
    left = np.copy(crop[:,:(w-x),:])
    left = cv2.flip(left, 1, dst=None)

    ww = left.shape[1]
    for i in range (0,h):
        left[i,0:int(ww/2-i*ww/2/h),:]=0    
    right = np.copy(left)
    left = cv2.flip(left, 1, dst=None)
    right_im = Image.fromarray((right * 255).astype(np.uint8))
    right_im = right_im.rotate(60,expand=True)
    right = np.array(right_im)
    for i in range (0,2):
        hh = right.shape[0]
        while np.sum(right[0,:,:])==0:
            right = right[1:hh]
        right = np.rot90(right,2)
    
    hh,ww = left.shape[:2]
    right = np.copy(right[-hh:,-ww:,:])
    hhr, wwr = right.shape[:2]
    
    
    right_r = np.zeros((hh,ww,3))
    if hhr < hh or wwr < ww:
        right_r[:hhr,:wwr,:] = np.copy(right)
        if hhr < hh:
            right_r[-(hh-hhr):,:,:] = np.copy(right[-(hh-hhr):,:,:])
        if wwr < ww :
            right_r[-(ww-wwr):,:,:] = np.copy(right[-(ww-wwr):,:,:])
    else:
        right_r = np.copy(right)
        
    unit_core = np.zeros((h,w,3))
    unit_core[:hh,:ww,:] = np.copy(((left * 255).astype(np.uint8)))
    unit_core_1 = np.copy(unit_core)
    unit_core_1[-hh:,-ww:,:] = np.copy(right)
    for i in range (0,h):
        for j in range (0,w):
            if np.sum(unit_core[i,j,:])==0:
                unit_core[i,j,:]=np.copy(unit_core_1[i,j,:])
    return unit_core

def p3m1(image, unit_h, Width, Height):
    rec = rec_crop(image,int(unit_h*math.sqrt(3)),unit_h)  
    crop = obli_crop(rec, 60)
    crop1 = rhomb_p3m1(crop)  # crop a rhombic for p3m1
    unit_p3m1 = p3_unit(crop1) # form a p3 unit
    obli_p3m1=translate_p3(unit_p3m1,Width,Height)
    obli_p3m1=border(obli_p3m1) 
    obli_p3m1=rhombic_edge(obli_p3m1) # remove black dot
    return obli_p3m1


# In[16]:


# p31m:

def rhomb_p31m(crop):
    h,w = crop.shape[:2]
    p1 = np.copy(crop)
    for i in range (0,h):
        p1[i,0:int(w-i*w/h),:]=0 
    p2 = cv2.flip(p1,0,dst=None)
    p2_im = Image.fromarray((p2 * 255).astype(np.uint8))
    p2_im = p2_im.rotate(60,expand=True)
    p2 = np.array(p2_im)
#     p2 = ndimage.rotate(p2,60) #rotation angle in degree
    
    for i in range (0,4):
        hh = p2.shape[0]
        while np.sum(p2[0,:,:])==0:
            p2 = p2[1:hh]
        p2 = np.rot90(p2,1)
#     p2 = p2[-h:,:,:] 
    for i in range (0,4):
        hh = p1.shape[0]
        while np.sum(p1[0,:,:])==0:
            p1 = p1[1:hh]
        p1 = np.rot90(p1,1)
    h,w = p1.shape[:2]

    rhombic = np.copy(((p1 * 255).astype(np.uint8)))
    for i in range (0,h):
        for j in range (0,w):
            if np.sum(rhombic[i,j,:])==0:
                rhombic[i,j,:]=np.copy(p2[i,j,:])
    return rhombic

def p31m(image, unit_h, Width, Height): 
    rec = rec_crop(image,int(unit_h*math.sqrt(3)),unit_h)
    crop = obli_crop(rec, 60)
    rhombic_p31m= rhomb_p31m(crop) # create p31m rhombic
    unit_p31m = p3_unit(rhombic_p31m) # form a p3 unit
    unit_p31m = border(unit_p31m) # remove black dot
    obli_p31m=translate_p3(unit_p31m,Width,Height) #translate
#     obli_p31m=border(obli_p31m) # remove black dot
    obli_p31m=rhombic_edge(obli_p31m)
    return obli_p31m


# In[17]:


# p6: 

def rhomb_p6(crop):
    h,w = crop.shape[:2]
    p1 = np.copy(crop)
    for i in range (0,h):
        p1[i,0:int(w-i*w/h),:]=0 
    p2 = np.rot90(p1,2)
    p2_roll = np.roll(p2,1,axis=1)
    
    rhombic = np.copy(p1)
    for i in range (0,h):
        for j in range (0,w):
            if np.sum(rhombic[i,j,:])==0:
                rhombic[i,j,:]=np.copy(p2_roll[i,j,:])
            if np.sum(rhombic[i,j,:])==0:
                rhombic[i,j,:]=np.copy(p2[i,j,:])
    return rhombic

def p6(image, unit_h, Width, Height):
    rec = rec_crop(image,int(unit_h*math.sqrt(3)),unit_h)
    crop = obli_crop(rec, 60) 
    rhombic_p6= rhomb_p6(crop) # create p6 rhombic
    unit_p6 = p3_unit(rhombic_p6) # form a p3 unit
    obli_p6=translate_p3(unit_p6,Width,Height) # translate
    obli_p6=border(obli_p6) # remove black dot
    return obli_p6


# In[18]:


# p6m:
def p6m(image, unit_h, Width, Height):
    rec = rec_crop(image,int(unit_h*math.sqrt(3)),unit_h)
    crop = obli_crop(rec, 60) 
    rhombic_p3m1 = rhomb_p3m1(crop)  

    # crop a rhombic for p3m1 :
    rhombic_p6m= rhomb_p6(rhombic_p3m1) 

    # perform p6 rhombic crop, get a p6m rhombic :
    unit_p6m = p3_unit(rhombic_p6m)
    unit_p6m = border(unit_p6m)
    
    # form a p6m unit:
    obli_p6m=translate_obli(unit_p6m,256,256) # translate

    obli_p6m=rhombic_edge(obli_p6m)
    obli_p6m=border(obli_p6m)
    return obli_p6m


# In[ ]:




