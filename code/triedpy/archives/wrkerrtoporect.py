# -*- coding: cp1252 -*-
import sys
import time as time
import numpy as np
TRIEDPY = "C:/Users/Charles/Documents/FAD/FAD_Charles";sys.path.append(TRIEDPY);
from   triedpy import triedtools as tls
from   triedpy import triedsompy as SOM
#----------------------------------------------------------------------
#te : Topographic error, the proportion of all data vectors
#       for which first AND second BMUs are not adjacent units.
#              0    1    2    3     4
#                        |
#              5    6--- 7 ---8     9
#                        |
#             10   11    12  13    14
#
# par ex les voisins de 7 :  7-5, 7-1, 7+1, 7+5  (5 etant le nbr de col)
# c'est donc  :  (7, 7, 7, 7) + (-5, -1, +1, +5) = (2, 6, 8, 12)
# supposons que le 2ème bmus après 7 soit :
#   13 -> +1 non adjacent
#    8 -> +0 non adjacent  
def errtopo(sm,bmus2) : # dans le cas 'rect' uniquement
    # plm, bmus2 est sensé etre un tableau Nx2
#   ncol    = sm.mapsize[0];
#   voisin  = np.array([-ncol, -1, +1, +ncol])
    ncol, nlig = sm.mapsize;
    nn = ncol; # Il se confirme que ncol doit être la bonne version et
               # que la numérotation de l'exemple ci-dessus serait correcte.
    voisin  = np.array([-nn, -1, +1, +nn])
    
    unarray = np.array([1, 1, 1, 1])
    not_adjacent = 0.0;
    for i in np.arange(sm.dlen) :
        vecti = unarray * bmus2[i,0]; #vecti = unarray * i;
        ivois = vecti + voisin
        #if bmus2[i,0] not in ivois or bmus2[i,1] not in ivois :
        if bmus2[i,1] not in ivois :
            not_adjacent = not_adjacent + 1.0;
        #print(ivois, bmus2[i,0], bmus2[i,1], not_adjacent);
    et = not_adjacent / sm.dlen
    return et
#
#def ctk_bmus (sm, Data=None) :
def mbmus (sm, Data=None, narg=1) :
    # multiple bmus
    if Data==[] or Data is None: # par defaut on prend celle de sm suppsé etre celles d'App.
        Data = sm.data
    nbdata   = np.size(Data,0);
    MBMUS    = np.ones((nbdata,narg)).astype(int)*-1;
    distance = np.zeros(sm.nnodes);
    for i in np.arange(nbdata) :
        for j in np.arange(sm.nnodes) :
            C = Data[i,:] - sm.codebook[j,:];
            distance[j] = np.dot(C,C); 
        #Imindist = np.argmin(distance);
        Imindist = np.argsort(distance);
        MBMUS[i]  = Imindist[0:narg];
    return MBMUS
#
#----------------------------------------------------------------------
varname = np.array(["JAN","FEV","MAR","AVR","MAI","JUI",
                    "JUI","AOU","SEP","OCT","NOV","DEC"]);
Data = np.loadtxt("sst_mmoyB.txt")
Dapp = tls.centred(Data); # Centrage et réduction
#======================================================================
#                       Carte Topologique
tseed = 0; print(tseed); np.random.seed(tseed);
#----------------------------------------------------------------------
# Création de la structure de la carte
#--------------------------------------
nbl = 3;  nbc = 5; # Taille de la carte
sm  = SOM.SOM('sm', Dapp, mapsize=[nbl, nbc], norm_method='data', \
              initmethod='random', varname=varname)
# Apprentissage de la carte                                   QE                      TE        
Parm_app = ( 3, 3.,1.,  12,1.,1.0);    # 3.193289  0.896840    0.440515  0.331517  0.6019
epoch1,radini1,radfin1,epoch2,radini2,radfin2 = Parm_app  
etape1=[epoch1,radini1,radfin1]; etape2=[epoch2,radini2,radfin2];
sm.train(etape1=etape1,etape2=etape2, verbose='on');
#
bmus2 = mbmus (sm, Data=None, narg=2);
et    = errtopo(sm, bmus2); # dans le cas 'rect' uniquement
print("erreur topologique = %.4f" %et)
#
x=0
A = SOM.grid_dist(sm,x); print(A)

#def rect_dist(self,bmu):
if 1 : # je tente l'équivalent de rect_dist
    bmu  = x; rows = nbl; cols = nbc
        
    #bmu should be an integer between 0 to no_nodes
    if 0<=bmu<=(rows*cols):
        c_bmu = int(bmu%cols); print(c_bmu);
        r_bmu = int(bmu/cols); print(r_bmu);
    #else: print('wrong bmu')  
      
    #calculating the grid distance
    if np.logical_and(rows>0 , cols>0):
        r,c = np.arange(0, rows, 1)[:,np.newaxis] , np.arange(0,cols, 1)
        dist2 = (r-r_bmu)**2 + (c-c_bmu)**2
        #return dist2.ravel()
    #else:...






    

