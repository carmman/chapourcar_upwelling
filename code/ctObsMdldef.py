# -*- coding: cp1252 -*-
import sys
import time as time
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
#
TRIEDPY = "C:/Users/Charles/Documents/FAD/FAD_Charles/WORKZONE/Python3"
sys.path.append(TRIEDPY);
from   triedpy import triedtools as tls; # triedpy rep should have been appended
from   localdef import *                 # this may be true for others rep 
#
#======================================================================
# Table(s) des Modèles
Tmodels_anyall_OUT = np.array([     # Mettre de coté certains modèles; on peut
        ["Observations",    ""],    # aussi les déplacer dans un repertoire sst_OUT/     
]);                                 # par exemple.
Tmodels_anyall = np.array([
        ["bcc-csm1-1",      ""],    #( 3)
        ["bcc-csm1-1-m",    ""],    #( 4)
        ["BNU-ESM",         ""],        # non fourni
        ["CanCM4",          ""],    #( 5)
        ["CanESM2",         ""],    #( 6)
        ["CMCC-CESM",       ""],    #(13)
        ["CMCC-CM",         ""],    #(14)
        ["CMCC-CMS",        ""],    #(15) 
        ["CNRM-CM5",        ""],    #(17)
        ["CNRM-CM5-2",      ""],    #(16)
        ["ACCESS1-0",       ""],    #(01)
        ["ACCESS1-3",       ""],    #( 2)       
        ["CSIRO-Mk3-6-0",   ""],    #(18)
        ["inmcm4",          ""],    #(35)
        ["IPSL-CM5A-LR",    ""],    #(36)
        ["IPSL-CM5A-MR",    ""],    #(37)
        ["IPSL-CM5B-LR",    ""],    #(38)
        ["FGOALS-g2",       ""],    #(20)
        ["FGOALS-s2",       ""],    # AAMMfin=200412 (360, 25, 36) -> manque une année 2005
        ["MIROC-ESM",       ""],    #(40)
        ["MIROC-ESM-CHEM",  ""],    #(22)
        ["MIROC5",          ""],    #(39)
        ["HadCM3",          ""],    #(31)
        ["HadGEM2-CC",      ""],    #(33)
        ["HadGEM2-ES",      ""],    #(34)
        ["MPI-ESM-LR",      ""],    #(41)
        ["MPI-ESM-MR",      ""],    #(42)
        ["MPI-ESM-P",       ""],    #(43)   
        ["MRI-CGCM3",       ""],    #(44)
        ["MRI-ESM1",        ""],    #(45)   
        ["GISS-E2-H",       ""],    #(28)
        ["GISS-E2-H-CC",    ""],    #(27)
        ["GISS-E2-R",       ""],    #(30)
        ["GISS-E2-R-CC",    ""],    #(29)
        ["CCSM4",           ""],    #(07)
        ["NorESM1-M",       ""],    #(46)
        ["NorESM1-ME",      ""],    #(47)
        ["HadGEM2-AO",      ""],    #(32) 
        ["GFDL-CM2p1",      ""],    #(23)
        ["GFDL-CM3",        ""],    #(24)
        ["GFDL-ESM2G",      ""],    #(25)
        ["GFDL-ESM2M",      ""],    #(26)
        ["CESM1-BGC",       ""],    #( 8)
        ["CESM1-CAM5",      ""],    #(10)  
        ["CESM1-CAM5-1-FV2",""],    #( 9)        
        ["CESM1-FASTCHEM",  ""],    #(11)
        ["CESM1-WACCM",     ""],    #(12)
        ["FIO-ESM",         ""],    #(??)
#       ["OBS",             ""],    #(??)
]);
#----------------------------------------------------------------------
def pentes(X) : # Courbes des pentes (b1) par pîxel
    N,L,C = np.shape(X);
    tps   = np.arange(N)+1;
    Tb1   = []
    plt.figure();
    for i in np.arange(C) :
        for j in np.arange(L) :
            y = X[:,j,i]
            b0,b1,s,R2,sigb0,sigb1= tls.linreg(tps,y);
            Tb1 = np.append(Tb1, b1)  #print(b0, b1)
    plt.plot(Tb1); plt.axis('tight');

def trendless(X) : # Suppression de la tendance
    N,L,C = np.shape(X);
    tps   = np.arange(N)+1;
    X_ = np.empty(np.shape(X))
    for i in np.arange(C) :
        for j in np.arange(L) :
            y = X[:,j,i]
            b0,b1,s,R2,sigb0,sigb1= tls.linreg(tps,y);
            ycor = y - b1*tps;
            X_[:,j,i] = ycor
    return X_

def anomalies(X) :
    N,L,C = np.shape(X);
    Npix  = L*C;
    X_    = np.reshape(X, (N,Npix)); 
    for i in np.arange(Npix) :
        for j in np.arange(0,N,12) :  # 0,   12,   24,   36, ...
            moypiaj = np.mean(X_[j:j+12, i]); # moyenne du pixel i année j
            X_[j:j+12, i] = X_[j:j+12, i] - moypiaj
    X_ = np.reshape(X_, (N,L,C));
    return X_

def Dpixmoymens(data,visu=None, climato=None) :
    global vvmin, vvmax
    #data may include nan
    #
    Ndata, Ldata, Cdata = np.shape(data)
    # On réorganise les données par pixel (Mise à plat des données)
    Npix = Ldata*Cdata; # 12x20
    Data = np.reshape(data, (Ndata,Npix));
    #
    if climato==None : # Calcul des moyennes mensuelles par pixels
        Data_mmoy = np.zeros((Npix,12));
        for m in np.arange(12) :            # Pour chaque mois m
            imois = np.arange(m,Ndata,12);  # les indices d'un mois m
            for i in np.arange(Npix) :      # Pour chaque pixel
                Data_mmoy[i,m] = np.mean(Data[imois,i]);    # on calcule la moyenne du mois
    #
    elif climato=="GRAD" : # Calcul des gradients mensuelles par pixels
        tm = np.arange(Ndata/12)+1;
        Data_mmoy = np.zeros((Npix,12));
        for m in np.arange(12) :            # Pour chaque mois m
            imois = np.arange(m,Ndata,12);  # les indices d'un mois m     
            for i in np.arange(Npix) :      # Pour chaque pixel
                y = Data[imois,i];
                b0,b1,s,R2,sigb0,sigb1= tls.linreg(tm,y);    # on calcule la régression
                Data_mmoy[i,m] = b1;
    else :
        print("Dpixmoymens : chosse a good climato");
        sys.exit(0)

    if visu is not None : # Une visu (2D) pour voir si ca ressemble à quelque chose.
        if climato=="GRAD" :
            vmin=np.nanmin(Data_mmoy);
            vmax=np.nanmax(Data_mmoy);
        else :
            vmin = vvmin; vmax = vvmax;
        showimgdata(Data_mmoy.T.reshape(12,1,Ldata,Cdata),n=12,fr=0,Labels=varnames,
                    cmap=cm.gist_ncar,interp='none',
                    figsize=(12, 9), wspace=0.0, hspace=0.0,
                    vmin=vmin,vmax=vmax);
                    #vmin=vvmin,vmax=vvmax);
                    #vmin=12.,vmax=32.);
                    #vmin=np.nanmin(Data),vmax=np.nanmax(Data));
        plt.suptitle("Dpixmoymens: visu to check : %s \nmin=%f, max=%f"
                    %(visu,np.nanmin(Data_mmoy),np.nanmax(Data_mmoy)));
                    #%(visu,np.nanmin(Data),np.nanmax(Data)));
    #
    # On vire les pixels qui sont nan (données manquantes)
    Data = np.reshape(Data_mmoy,(Npix*12)); 
    Inan = np.where(np.isnan(Data))[0];  # Is Not a Number
    Iisn = np.where(~np.isnan(Data))[0]; # IS a Number
    Data = Data[Iisn]
    Data = Data.reshape(int(len(Iisn)/12),12);
    return Data, Iisn, Inan

def aff2D(XD,L,C,isnum,isnan,varnames=None,wvmin=None,wvmax=None,fignum=None,figsize=(9,9),
          cbpos='vertical', wspace=0.01, hspace=0.01, top=0.93, bottom=0.10,
          left=0.05, right=0.98,x=0.5,y=0.96,noaxes=True,noticks=True,nolabels=True) :
    ND,p      = np.shape(XD);
    X_        = np.empty((L*C,p));   
    X_[isnum] = XD   
    X_[isnan] = np.nan
    showimgdata(X_.T.reshape(p,1,L,C),n=p,fr=0,Labels=varnames,interp='none',
                cmap=cm.gist_ncar,fignum=fignum,figsize=figsize,cbpos=cbpos,
                wspace=wspace, hspace=hspace, top=top, bottom=bottom,
                left=left, right=right,x=x,y=y,noaxes=noaxes,noticks=noticks,nolabels=nolabels,
                vmin=wvmin,vmax=wvmax);
    
def refbmusD(sm, bmus, Lig, Col, Iisn, Inan) :
    Ndata = len(bmus);
    #obs_bmus  = bmus2O[:,0];
    Dbmus    = sm.codebook[bmus,]; 
    X_       = np.empty(Lig*Col*12);  
    X_[Iisn] = Dbmus.reshape(Ndata*12);
    X_[Inan] = np.nan;
    X_       =  X_.reshape(Lig*Col,12); 
    showimgdata(X_.T.reshape(12,1,Lig,Col),n=12,fr=0,Labels=varnames,
                cmap=cm.gist_ncar,interp='none',
                figsize=(12, 9), wspace=0.0, hspace=0.0,
                vmin=wvmin,vmax=wvmax);
    return X_

def moybmusD(X_,Lig,Col): # Visu des moyennes des pixels
    moyX_ = np.nanmean(X_,axis=1)
    plt.figure()
    plt.imshow(moyX_.reshape(Lig,Col), cmap=cm.gist_ncar,interpolation='none', 
               vmin=wvmin,vmax=wvmax);

def dto2d(X1D,L,C,isnum,missval=np.nan) :
    # Pour les classes par exemple, passer de 1D (cad N pixels
    # valides en ligne à une image 2D (LxC) des classes)
    X = np.ones(L*C)*missval;
    X[isnum] = X1D;
    X = np.reshape(X,(L,C));
    return X

def indsc(X, C=20) :
    # Climatologie glissante par une année
    # X : N images-mensuelles de dimension py, px
    # C : Nombre d'années pour une Climatologie
    N, py, px = np.shape(X);
    P = py*px;
    X = X.reshape(N,py*px);  
    K = 1 + (N-C*12) / 12; # Nombre de climatologies
    K = int(K); # because python 2.7 et 3.4 c'est pas pareil
    IndSC = np.zeros((K,P)); # Init
    for k in np.arange(K) :
        MoyCp = np.zeros((12,P));   # Init Moyenne mensuelle sur les C années pour les pixels i
        for m in np.arange(12) :
            Im = np.arange(m,12*C,12);                      #(1)
            Im = Im + 12*k;         # print(Im)
            for i in np.arange(P) : # Pour chaque pixel
                MoyCp[m,i] = np.mean(X[Im,i])               #(2) SST20[x,y,12]
        #
        # Now, Calcul de IndSC    
        for i in np.arange(P) : # Pour chaque pixel
            IndSC[k,i] =  np.max(MoyCp[:,i]) - np.min(MoyCp[:,i])   
    IndSC = IndSC.reshape(K,py,px);
    return IndSC
#----------------------------------------------------------------------
def inan(X) :
    # On fait l'hypothèse que tous les nans d'une image à l'autre sont
    # à la même place
    L,C   = np.shape(X); #print("X.shape : ", N, L, C);
    X_    = np.reshape(X, L*C); 
    Inan  = np.where(np.isnan(X_))[0];
    return Inan
def imaxnan() :
    X_       = np.load("Datas/sst_obs_1854a2005_Y60X315.npy");
    Inan     = inan(X_[0])
    X_       = np.load("Datas/CM5A-LR_rm_1854a2005_25L36C.npy")
    Inan_mdl = inan(X_[0])
    Inan     = np.unique(np.concatenate((Inan,Inan_mdl)))
    X_       = np.load("Datas/CM5A-MR_rm_1854a2005_25L36C.npy")
    Inan_mdl = inan(X_[0])
    Inan     = np.unique(np.concatenate((Inan,Inan_mdl)))
    X_       = np.load("Datas/GISS-E2-R_rm_1854a2005_25L36C.npy")
    Inan_mdl = inan(X_[0])
    Inan     = np.unique(np.concatenate((Inan,Inan_mdl)))
    return Inan;
#----------------------------------------------------------------------
def nan2moy (X, Left=True, Above=True, Right=True, Below=True) :
    # On supose que quelle que soit l'image de X tous les nan son calé 
    # sur les memes pixels; on a choisi de s'appuyer sur l'image 0
    # (pompé de C:\...\DonneesUpwelling\Upwell2_predictions\sst_nc_X.py)
    Nim, Lim, Cim = np.shape(X);
    ctrnan = 0;
    for L in np.arange(Lim) :             #Lim=25:=> [0, 1, ..., 23, 24]
        for C in np.arange(Cim) :         #Cim=36:=> [0, 1, ..., 34, 35]
    #for C in np.arange(Cim) :             #Cim=36:=> [0, 1, ..., 34, 35]
    #    for L in np.arange(Lim) :         #Lim=25:=> [0, 1, ..., 23, 24]
    #for L in np.arange(Lim-1,-1,-1) :     #Lim=25:=> [24, 23, ..., 1, 0]
    #    for C in np.arange(Cim-1,-1,-1) : #Cim=36:=> [35, 34, ..., 1, 0]
    #for C in np.arange(Cim-1,-1,-1) :     #Cim=36:=> [35, 34, ..., 1, 0]
    #    for L in np.arange(Lim-1,-1,-1) : #Lim=25:=> [24, 23, ..., 1, 0]
    #for L in np.arange(Lim) :             #Lim=25:=> [0, 1, ..., 23, 24]
    #    for C in np.arange(Cim-1,-1,-1) : #Cim=36:=> [35, 34, ..., 1, 0]
    #for C in np.arange(Cim-1,-1,-1) :     #Cim=36:=> [35, 34, ..., 1, 0]
    #    for L in np.arange(Lim) :         #Lim=25:=> [0, 1, ..., 23, 24]
    #for L in np.arange(Lim-1,-1,-1) :     #Lim=25:=> [24, 23, ..., 1, 0]
    #    for C in np.arange(Cim) :         #Cim=36:=> [0, 1, ..., 34, 35]
    #for C in np.arange(Cim) :              #Cim=36:=> [0, 1, ..., 34, 35]
    #    for L in np.arange(Lim-1,-1,-1) :  #Lim=25:=> [24, 23, ..., 1, 0]
    #      
            if np.isnan(X[0,L,C]) :   # tous calé sur les memes pixels
                ctrnan = ctrnan+1;
                nok = 0; som=np.zeros(Nim)
                if Left and C > 0 :
                    if ~np.isnan(X[0,L,  C-1]) :  # carré de gauche
                        som = som + X[:,L,  C-1]; nok = nok + 1;
                if Above and L > 0 :
                    if ~np.isnan(X[0,L-1,C]) :    # carré du dessus
                        som = som + X[:,L-1,C  ]; nok = nok + 1;
                if Right and C+1 < Cim :
                    if ~np.isnan(X[0,L,  C+1]) :  # carré de droite
                        som = som + X[:,L,  C+1]; nok = nok + 1;
                if Below and L+1 < Lim :
                    if ~np.isnan(X[0,L+1,C]) :    # carré du dessous
                        som = som + X[:,L+1,C  ]; nok = nok + 1;
                if nok > 0 :
                    X[:,L,C] = som/nok;
    #print("nombre de nan de chaque image : %d" %(ctrnan));
    return X, ctrnan;
#======================================================================
#----------------------------------------------------------------------
def grid() :
    axes = plt.axis(); # (-0.5, 35.5, 24.5, -0.5)
    for i in np.arange(axes[0], axes[1], 1) :
        for j in np.arange(axes[3], axes[2], 1) :
            plt.plot([axes[0], axes[1]],[j, j],'k-',linewidth=0.5);
            plt.plot([i, i],[axes[2], axes[3]],'k-',linewidth=0.5);
    plt.axis('tight');

#----------------------------------------------------------------------
def transco_class(class_ref,codebook,crit='') :
    nb_class = max(class_ref)
    if isinstance(crit, str) :
        Tvalcrit = np.zeros(nb_class); 
        for c in np.arange(nb_class) :
            Ic = np.where(class_ref==c+1)[0];
            if crit=='GAP' :
                Tvalcrit[c] = np.max(codebook[Ic])-np.min(codebook[Ic]);
                #print(c+1, np.max(codebook[Ic]),np.min(codebook[Ic]),np.max(codebook[Ic])-np.min(codebook[Ic]))
            elif crit=='GAPNM' : # NormMax par curiosité pour voir
                Tvalcrit[c] = (np.max(codebook[Ic])-np.min(codebook[Ic])) / np.max(codebook[Ic]);
            elif crit=='STD' :
                Tvalcrit[c] = np.std(codebook[Ic]);
            elif crit=='MOY' : # does not ok for anomalie ?
                Tvalcrit[c] = np.mean(codebook[Ic]);
            elif crit=='MAX' :
                Tvalcrit[c] = np.max(codebook[Ic]);
            elif crit=='MIN' :
                Tvalcrit[c] = np.min(codebook[Ic]);
            elif crit=='GRAD' :
                tm = np.arange(12)+1;  #!!! 12 en dure
                sompente = 0.0
                for p in np.arange(len(Ic)) :
                    y = codebook[Ic[p]]
                    b0,b1,s,R2,sigb0,sigb1= tls.linreg(tm,y);
                    sompente = sompente + b1;
                Tvalcrit[c] = sompente / len(Ic); # la moyenne des pentes sur les CB de cette classe 
            else :
                print("transco_class : bad criterium, should be one of this : \
                       'GAP', 'GAPNM', 'STD', 'MOY', 'MAX', 'MIN', 'GRAD'; found %s"%crit);
                sys.exit(0);
        Icnew = np.argsort(Tvalcrit);
    else :
        Icnew = np.array(crit)-1;
    #    
    cref_new = np.zeros(len(class_ref)).astype(int);               
    cc = 1;
    for c in Icnew :
        Ic = np.where(class_ref==c+1)[0];
        cref_new[Ic] = cc;
        cc = cc+1; 
    return cref_new

def moymensclass (varN2D,isnum,classe_D,nb_class) :
    # Moyennes mensuelles par classe
    N,L,C   = np.shape(varN2D);
    #!!!X   = np.reshape(sst_obs,(N,L*C))
    X       = np.reshape(varN2D,(N,L*C))
    MoyMens = np.zeros((12,L*C))
    for m in np.arange(12) :
        Im = np.arange(m,N,12);
        Dm = X[Im];
        MoyMens[m] = np.mean(Dm, axis=0);
    MoyMens = MoyMens[:,isnum]
    #        
    Tmoymensclass = np.zeros((12,nb_class))
    for c in np.arange(nb_class) :
        Ic = np.where(classe_D==c+1)[0];
        Tmoymensclass[:,c] = np.mean(MoyMens[:,Ic], axis=1)
    return Tmoymensclass

def deltalargeM12(X,L,C,isnum) :
    # Différence de sst par rapport à la valeur au large. (UISST?)
    M, m = np.shape(X);    # (743, 12)
    Xm_  = np.zeros((L*C, m))
    Xm_[isnum,:] = X;
    Xm_  = Xm_.reshape(L,C, m);
    for l in np.arange(L) : 
        for c in np.arange(C-1,-1,-1) : # [35, 34, ..., 1, 0]
            Xm_[l,c,:] = Xm_[l,c,:] - Xm_[l,0,:];
    Xm_ = Xm_.reshape(L*C,m);
    return Xm_[isnum,:]

#======================================================================
def perfbyclass (classe_Dobs,classe_Dmdl,nb_class,globismean=False) :
    NDobs = len(classe_Dobs);
    perflist = []; # to check
    Tperf = [];
    classe_DD = np.ones(NDobs)*np.nan; 
    for c in np.arange(nb_class)+1 :      
        iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
        imdlc = np.where(classe_Dmdl==c)[0]; # Indices des classes c des mdl
        #igood= np.where(imdlc==iobsc)[0];   # np.where => même dim
        igood = np.intersect1d(imdlc,iobsc);
        classe_DD[igood] = c;
        
        Niobsc=len(iobsc); Nigood=len(igood);
        #perfc = Nigood/Niobsc; #
        if Niobsc>0 : # because avec red_class... on peut avoir écarté une classe
            perfc = Nigood/Niobsc; # erratum: perfc = Nigood/Nimdlc;
        else :
            perfc = 0.0; # ... à voir ... 
        Tperf.append(perfc)
        if globismean :
            perflist.append(perfc); # cummule la performance de la classe
        else:
            perflist.append(Niobsc * perfc); # cummule la performance ponderee par le nombre de pixels Obs dans la classe
    if globismean :
        Perfglob = np.mean(perflist);
    else:
        Perfglob = np.sum(perflist)/NDobs;
    return classe_DD, Tperf, Perfglob

def red_classgeo(X,isnum,classe_D,frl,tol,frc,toc) :
    # Doit retourner les classes sur une zone réduite en faisant
    # attention aux nans
    N,L,C = np.shape(X);  
    XC_   = np.ones(L*C)*np.nan;
    XC_[isnum] = classe_D;
    XC_   = XC_.reshape(L,C)
    XC_   = XC_[frl:tol,frc:toc];
    l,c   = np.shape(XC_); 
    oC_   = XC_.reshape(l*c)
    isnum_red   = np.where(~np.isnan(oC_))[0]
    oC_   = oC_[isnum_red]
    X_     = X[:,frl:tol,frc:toc];
    return X_, XC_, oC_, isnum_red;
#======================================================================
#
