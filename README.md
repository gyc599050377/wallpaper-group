# Introduction

Wallpaper-group code is used to generate 17 different types of wallpaper group images(256*256).

It start form random position of original image and crop random unit size (with size restrictions) to do 17 trannsformations.

The transformation list is : cm, cmm, p1, p2, p3, p31m, p3m1, p4, p4g, p4m, p6, p6m, pg, pgg, pm, pmg, pmm.

And based on the shape and transformation direction of unit cell, 17 groups are seperated to 29 catagories. 
The list is: cm_1, cm_2, cmm, p1_1, p1_2, p1_3, p1_4, p1_5, p2_1, p2_2, p2_3, p2_4, p2_5, p3, p31m, p3m1, p4, p4g, p4m, p6, p6m, pg_1, pg_2, pgg, pm_1, pm_2, pmg_1, pmg_2, pmm.

It has been labeled with number according to wikipedia page: https://en.wikipedia.org/wiki/Wallpaper_group, which indicates the wallpaper group type and the unit shape. 
