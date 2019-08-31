#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wp_src_code as wp

import cv2, math, os, random
import numpy as np

from fastai.vision import * # after install fastai

from joblib import parallel_backend, Parallel, delayed


# In[2]:


pwd


# In[3]:


# wget -i image.txt


# In[4]:


cd /home/yig319/Documents/image_source


# In[5]:


def wallpaper_group_generate(source, output):
    
    imagelist = os.listdir(source) 
    l=len(imagelist)
    start_number = 0
    # random.randint(1,l-1)
    
    for i in range(start_number,l-1):
        file=imagelist[i]
        dest=file[:-4]
        folder_str = output + dest
        path_folder = Path(output + dest)

        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        angle = random.randint(20,85)
        unit_h = random.randint(30,80)
        lower_bound = int(unit_h/math.tan(math.radians(angle)))
        if lower_bound > 65:
            unit_w = random.randint(lower_bound+15, lower_bound+75)
        else: 
            unit_w = random.randint(lower_bound+15,85)

        ori = cv2.imread(imagelist[i])
        image = np.copy(ori)

        try: 
            p1_1 = wp.p1_rectangular(image,unit_w,unit_h,256,256)
            p1_2 = wp.p1_square(image,unit_w,256,256)
            p1_3 = wp.p1_oblique(image,unit_w,unit_h,angle,256,256) 
            p1_4 = wp.p1_hexagonal(image,unit_w,256,256)
            p1_5 = wp.p1_rhombic(image,unit_w,unit_h,256,256) 

            cv2.imwrite(folder_str+ '/' +'p1_1.png',p1_1)
            cv2.imwrite(folder_str+ '/' +'p1_2.png',p1_2)
            cv2.imwrite(folder_str+ '/' +'p1_3.png',p1_3)
            cv2.imwrite(folder_str+ '/' +'p1_4.png',p1_4)
            cv2.imwrite(folder_str+ '/' +'p1_5.png',p1_5)
        except:
            pass

        try: 
            p2_1 = wp.p2_rectangular(image,unit_w,unit_h,256,256)
            p2_2 = wp.p2_square(image,unit_w,256,256)
            p2_3 = wp.p2_oblique(image,unit_w,unit_h,angle,256,256) 
            p2_4 = wp.p2_hexagonal(image,unit_w,256,256)
            p2_5 = wp.p2_rhombic(image,unit_w,unit_h,256,256)

            cv2.imwrite(folder_str+ '/' +'p2_1.png',p2_1)
            cv2.imwrite(folder_str+ '/' +'p2_2.png',p2_2)
            cv2.imwrite(folder_str+ '/' +'p2_3.png',p2_3)
            cv2.imwrite(folder_str+ '/' +'p2_4.png',p2_4)
            cv2.imwrite(folder_str+ '/' +'p2_5.png',p2_5)
        except:
            pass


        try: 
            pm_1 = wp.pm_horizontal(image,unit_w,unit_h,256,256)
            pm_2 = wp.pm_vertical(image,unit_w,unit_h,256,256)
            cv2.imwrite(folder_str+ '/' +'pm_1.png',pm_1)
            cv2.imwrite(folder_str+ '/' +'pm_2.png',pm_2)
        except:
            pass


        try: 
            pg_1 = wp.pg_horizontal(image,unit_w,unit_h,256,256)
            pg_2 = wp.pg_vertical(image,unit_w,unit_h,256,256)
            cv2.imwrite(folder_str+ '/' +'pg_1.png',pg_1)
            cv2.imwrite(folder_str+ '/' +'pg_2.png',pg_2)
        except:
            pass

        try: 
            pmm_1 = wp.pmm(image,unit_w,unit_h,256,256)
            cv2.imwrite(folder_str+ '/' +'pmm_1.png',pmm_1)
        except:
            pass

        try: 
            pmg_1 = wp.pmg_horizontal(image,unit_w,unit_h,256,256)
            pmg_2 = wp.pmg_vertical(image,unit_w,unit_h,256,256)
            cv2.imwrite(folder_str+ '/' +'pmg_1.png',pmg_1)
            cv2.imwrite(folder_str+ '/' +'pmg_2.png',pmg_2)
        except:
            pass

        try: 
            cm_1 = wp.cm_horizontal(image,unit_w,unit_h,256,256) 
            cm_2 = wp.cm_vertical(image,unit_w,unit_h,256,256)
            cv2.imwrite(folder_str+ '/' +'cm_1.png',cm_1)
            cv2.imwrite(folder_str+ '/' +'cm_2.png',cm_2)
        except:
            pass

        try: 
            pgg_1 = wp.pgg(image,unit_w,unit_h,256,256)
            cv2.imwrite(folder_str+ '/' +'pgg_1.png',pgg_1) 
        except:
            pass

        try: 
            cmm_1 = wp.cmm(image,unit_w,unit_h,256,256)
            cv2.imwrite(folder_str+ '/' +'cmm_1.png',cmm_1)  
        except:
            pass


        try: 
            p4_1 = wp.p4(image,unit_w,256,256)
            cv2.imwrite(folder_str+ '/' +'p4_1.png',p4_1)  
        except:
            pass


        try: 
            p4m_1 = wp.p4m(image,unit_w,256,256)
            cv2.imwrite(folder_str+ '/' +'p4m_1.png',p4m_1)
        except:
            pass

        try: 
            p4g_1 = wp.p4g(image,unit_w,256,256)
            cv2.imwrite(folder_str+ '/' +'p4g_1.png',p4g_1)
        except:
            pass

        try: 
            p3_1 = wp.p3(image,unit_w,256,256)
            cv2.imwrite(folder_str+ '/' +'p3_1.png',p3_1)
        except:
            pass


        try: 
            p3m1_1 = wp.p3m1(image,unit_w,256,256)  
            cv2.imwrite(folder_str+ '/' +'p3m1_1.png',p3m1_1)
        except:
            pass

        try: 
            p31m_1 = wp.p31m(image,unit_w,256,256)
            cv2.imwrite(folder_str+ '/' +'p31m_1.png',p31m_1)
        except:
            pass

        try: 
            p6_1 = wp.p6(image,unit_w,256,256)
            cv2.imwrite(folder_str+ '/' +'p6_1.png',p6_1)
        except:
            pass

        try: 
            p6m_1 = wp.p6m(image,unit_w,256,256) 
            cv2.imwrite(folder_str+ '/' +'p6m_1.png',p6m_1)
        except:
            pass

    return ()


# In[6]:


source = '/home/yig319/Documents/image_source/'


# In[7]:


root_path = '/home/yig319/Documents/image_results/'

folders = ['folder 25', 
           'folder 26', 'folder 27', 'folder 28', 'folder 29', 'folder 30',
           'folder 31', 'folder 32', 'folder 33', 'folder 34', 'folder 35',
           'folder 36', 'folder 37', 'folder 38', 'folder 39', 'folder 40',
           'folder 41', 'folder 42', 'folder 43', 'folder 44', 'folder 45', 
           'folder 46', 'folder 47', 'folder 48', 'folder 49', 'folder 50', 
           'folder 51', 'folder 52', 'folder 53', 'folder 54', 'folder 55', 
           'folder 56', 'folder 57', 'folder 58', 'folder 59', 'folder 60',
           'folder 61', 'folder 62', 'folder 63', 'folder 64', 'folder 65',
           'folder 66', 'folder 67', 'folder 68', 'folder 69', 'folder 70',]

for folder in folders:

    if not os.path.exists(os.path.join(root_path, folder)):
        os.mkdir(os.path.join(root_path, folder))

folder_list = os.listdir(root_path) 
output = copy(folders)

for i in range(25): output[i] = root_path + folders[i] +'/'


# In[ ]:


with parallel_backend('loky', n_jobs=23):
    Parallel()(delayed(wallpaper_group_generate)(source, output[i]) for i in range(46))


# In[ ]:





# In[10]:


print("total number is" ,l)
print("the starting number is", start_number)


# In[ ]:





# In[ ]:


# imagelist = os.listdir(source) 
# l=len(imagelist)
# start_number = 0
# # random.randint(1,l-1)

# for i in range(start_number,l-1):
#     file=imagelist[i]
#     dest=file[:-4]
#     folder_str = output1 + dest
#     path_folder = Path(output1 + dest)
    
#     if not os.path.exists(path_folder):
#         os.makedirs(path_folder)

#     angle = random.randint(20,85)
#     unit_h = random.randint(30,80)
#     lower_bound = int(unit_h/math.tan(math.radians(angle)))
#     if lower_bound > 65:
#         unit_w = random.randint(lower_bound+15, lower_bound+75)
#     else: 
#         unit_w = random.randint(lower_bound+15,85)

#     ori = cv2.imread(imagelist[i])
#     image = np.copy(ori)
    
#     try: 
#         p1_1 = wp.p1_rectangular(image,unit_w,unit_h,256,256)
#         p1_2 = wp.p1_square(image,unit_w,256,256)
#         p1_3 = wp.p1_oblique(image,unit_w,unit_h,angle,256,256) 
#         p1_4 = wp.p1_hexagonal(image,unit_w,256,256)
#         p1_5 = wp.p1_rhombic(image,unit_w,unit_h,256,256) 

#         cv2.imwrite(folder_str+ '/' +'p1_1.png',p1_1)
#         cv2.imwrite(folder_str+ '/' +'p1_2.png',p1_2)
#         cv2.imwrite(folder_str+ '/' +'p1_3.png',p1_3)
#         cv2.imwrite(folder_str+ '/' +'p1_4.png',p1_4)
#         cv2.imwrite(folder_str+ '/' +'p1_5.png',p1_5)
#     except:
#         pass
    
#     try: 
#         p2_1 = wp.p2_rectangular(image,unit_w,unit_h,256,256)
#         p2_2 = wp.p2_square(image,unit_w,256,256)
#         p2_3 = wp.p2_oblique(image,unit_w,unit_h,angle,256,256) 
#         p2_4 = wp.p2_hexagonal(image,unit_w,256,256)
#         p2_5 = wp.p2_rhombic(image,unit_w,unit_h,256,256)

#         cv2.imwrite(folder_str+ '/' +'p2_1.png',p2_1)
#         cv2.imwrite(folder_str+ '/' +'p2_2.png',p2_2)
#         cv2.imwrite(folder_str+ '/' +'p2_3.png',p2_3)
#         cv2.imwrite(folder_str+ '/' +'p2_4.png',p2_4)
#         cv2.imwrite(folder_str+ '/' +'p2_5.png',p2_5)
#     except:
#         pass

    
#     try: 
#         pm_1 = wp.pm_horizontal(image,unit_w,unit_h,256,256)
#         pm_2 = wp.pm_vertical(image,unit_w,unit_h,256,256)
#         cv2.imwrite(folder_str+ '/' +'pm_1.png',pm_1)
#         cv2.imwrite(folder_str+ '/' +'pm_2.png',pm_2)
#     except:
#         pass
    

#     try: 
#         pg_1 = wp.pg_horizontal(image,unit_w,unit_h,256,256)
#         pg_2 = wp.pg_vertical(image,unit_w,unit_h,256,256)
#         cv2.imwrite(folder_str+ '/' +'pg_1.png',pg_1)
#         cv2.imwrite(folder_str+ '/' +'pg_2.png',pg_2)
#     except:
#         pass
    
#     try: 
#         pmm_1 = wp.pmm(image,unit_w,unit_h,256,256)
#         cv2.imwrite(folder_str+ '/' +'pmm_1.png',pmm_1)
#     except:
#         pass
    
#     try: 
#         pmg_1 = wp.pmg_horizontal(image,unit_w,unit_h,256,256)
#         pmg_2 = wp.pmg_vertical(image,unit_w,unit_h,256,256)
#         cv2.imwrite(folder_str+ '/' +'pmg_1.png',pmg_1)
#         cv2.imwrite(folder_str+ '/' +'pmg_2.png',pmg_2)
#     except:
#         pass
    
#     try: 
#         cm_1 = wp.cm_horizontal(image,unit_w,unit_h,256,256) 
#         cm_2 = wp.cm_vertical(image,unit_w,unit_h,256,256)
#         cv2.imwrite(folder_str+ '/' +'cm_1.png',cm_1)
#         cv2.imwrite(folder_str+ '/' +'cm_2.png',cm_2)
#     except:
#         pass
    
#     try: 
#         pgg_1 = wp.pgg(image,unit_w,unit_h,256,256)
#         cv2.imwrite(folder_str+ '/' +'pgg_1.png',pgg_1) 
#     except:
#         pass

#     try: 
#         cmm_1 = wp.cmm(image,unit_w,unit_h,256,256)
#         cv2.imwrite(folder_str+ '/' +'cmm_1.png',cmm_1)  
#     except:
#         pass
    
    
#     try: 
#         p4_1 = wp.p4(image,unit_w,256,256)
#         cv2.imwrite(folder_str+ '/' +'p4_1.png',p4_1)  
#     except:
#         pass
    
    
#     try: 
#         p4m_1 = wp.p4m(image,unit_w,256,256)
#         cv2.imwrite(folder_str+ '/' +'p4m_1.png',p4m_1)
#     except:
#         pass
    
#     try: 
#         p4g_1 = wp.p4g(image,unit_w,256,256)
#         cv2.imwrite(folder_str+ '/' +'p4g_1.png',p4g_1)
#     except:
#         pass

#     try: 
#         p3_1 = wp.p3(image,unit_w,256,256)
#         cv2.imwrite(folder_str+ '/' +'p3_1.png',p3_1)
#     except:
#         pass

      
#     try: 
#         p3m1_1 = wp.p3m1(image,unit_w,256,256)  
#         cv2.imwrite(folder_str+ '/' +'p3m1_1.png',p3m1_1)
#     except:
#         pass
    
#     try: 
#         p31m_1 = wp.p31m(image,unit_w,256,256)
#         cv2.imwrite(folder_str+ '/' +'p31m_1.png',p31m_1)
#     except:
#         pass
     
#     try: 
#         p6_1 = wp.p6(image,unit_w,256,256)
#         cv2.imwrite(folder_str+ '/' +'p6_1.png',p6_1)
#     except:
#         pass
    
#     try: 
#         p6m_1 = wp.p6m(image,unit_w,256,256) 
#         cv2.imwrite(folder_str+ '/' +'p6m_1.png',p6m_1)
#     except:
#         pass

   

