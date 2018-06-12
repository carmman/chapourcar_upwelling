# -*- coding: cp1252 -*-
import sys
import time as time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
from   scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.io
#
from sklearn.metrics.cluster import adjusted_rand_score
#
from   triedpy import triedtools as tls
from   triedpy import triedsompy as SOM
from   triedpy import triedacp   as acp
#from  triedpy import triedctk   as ctk
import UW3_triedctk   as ctk
#
from   localdef    import *
from   ctObsMdldef import *
#
'''                NOTES
VERSION POUR CARLOS

Glossaire
   4CT ___ pour Carte Topologique
   
'''
#%=====================================================================
def afcnuage (CP,cpa,cpb,Xcol,K,xoomK=500,linewidths=1,indname=None,
              cmap=cm.jet,holdon=False) :
# pompé de WORKZONE ... TPA05
    if holdon == False :
        # j'ai un pb obs \ pas obs qui apparaissent dans la même couleur que le dernier cluster
        # quand bien même il ne participe pas à la clusterisation
        lenCP = len(CP); lenXcol = len(Xcol);
        if lenCP > lenXcol : # hyp : lenCP>lenXcol
            # Je considère que les (LE) surnuméraire de CP sont les obs (ou aut chose), je l'enlève,
            # et le met de coté
            CPobs = CP[lenXcol:lenCP,:];
            CP    = CP[0:lenXcol,:];
            # K et indname suivent CP
            if np.ndim(K) == 1 :
                K = K.reshape(len(K),1)
            Kobs  = K[lenXcol:lenCP,:];
            K     = K[0:lenXcol,:];
            obsname = indname[lenXcol:lenCP];
            indname = indname[0:lenXcol];
        #
        plt.figure(figsize=(16,12));
        my_norm = plt.Normalize()
        my_normed_data = my_norm(Xcol)
        ec_colors = cmap(my_normed_data) # a Nx4 array of rgba value
        #? if np.ndim(K) > 1 : # On distingue triangle à droite ou vers le haut selon l'axe
        n,p = np.shape(K);
        if p > 1 : # On distingue triangle à droite ou vers le haut selon l'axe 
            plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpa-1]*xoomK,
                            marker='>',edgecolors=ec_colors,facecolor='none',linewidths=linewidths)
            plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpb-1]*xoomK,
                            marker='^',edgecolors=ec_colors,facecolor='none',linewidths=linewidths)
            if lenCP > lenXcol : # cas des surnumeraire, en principe les obs
                plt.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs[:,cpa-1]*xoomK,
                                marker='>',edgecolors='k',facecolor='none',linewidths=linewidths)
                plt.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs[:,cpb-1]*xoomK,
                                marker='^',edgecolors='k',facecolor='none',linewidths=linewidths)            
        else :
            plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K*xoomK,
                            marker='s',edgecolors=ec_colors,facecolor='none',linewidths=linewidths);
            if lenCP > lenXcol : # ? cas des surnumeraire, en principe les obs
                plt.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs*xoomK,
                            marker='s',edgecolors='k',facecolor='none',linewidths=linewidths);
                
    else : #(c'est pour les colonnes)
        plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpa-1]*xoomK,
                        marker='o',facecolor='m')
        plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpb-1]*xoomK,
                        marker='o',facecolor='c',alpha=0.5)
    #plt.axis('tight')
    plt.xlabel('axe %d'%cpa); plt.ylabel('axe %d'%cpb)
    
    if 0 : # je me rapelle plus tres bien à quoi ca sert; do we need a colorbar here ? may be
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(Xcol), vmax=np.max(Xcol)))
        sm.set_array([])
        #if holdon == False :
        #    plt.colorbar(sm);

    # Labelisation des points, if not empty
    if indname is not None :
        N,p = np.shape(CP);
        for i in np.arange(N) :
            plt.text(CP[i,cpa-1],CP[i,cpb-1],indname[i])
    if holdon == False and lenCP > lenXcol :
        N,p = np.shape(CPobs);
        for i in np.arange(N) :
            plt.text(CPobs[i,cpa-1],CPobs[i,cpb-1],obsname[i])
            
    # Tracer les axes
    xlim = plt.xlim(); plt.xlim(xlim);
    plt.plot(xlim, np.zeros(2));
    ylim = plt.ylim(); plt.ylim(ylim);
    plt.plot(np.zeros(2),ylim);

    # Plot en noir des triangles de référence en bas à gauche
    if holdon == False :
        dx = xlim[1] - xlim[0];
        dy = ylim[1] - ylim[0];
        px = xlim[0] + dx/(xoomK) + dx/20; # à ajuster +|- en ...
        py = ylim[0] + dy/(xoomK) + dy/20; # ... fonction de xoomK
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK,     facecolor='none');
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK*0.5, facecolor='none');
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK*0.1, facecolor='none');
#
#----------------------------------------------------------------------
def Dmdlmoy4CT (TDmdl4CT,igroup,pond=None) :
    # Modèle Moyen d'un groupe\cluster des données 4CT
    # Si pond, il doit avoir la meme longueur que TDmdl4CT
    if pond is None : # Non pondéré
        CmdlMoy = np.mean(TDmdl4CT[igroup],axis=0); # Dmdl moyen d'un cluster
    else : # Modèle Moyen Pondéré
        pond       = pond[igroup]; # dans igroup y'a pas l'indice qui correspond aux Obs
        TDmdl4CTi_ = TDmdl4CT[igroup];        # (11,743,12)
        CmdlMoy    = TDmdl4CTi_[0] * pond[0]; # init du modele moyen
        for kk in np.arange(len(pond)-1)+1 :
            CmdlMoy = CmdlMoy + (TDmdl4CTi_[kk] * pond[kk])
        CmdlMoy    = CmdlMoy / np.sum(pond);
    return CmdlMoy; # Cluster modèle Moyen
#
def Dgeoclassif(sMap,Data,L,C,isnum) :
    bmus_   = ctk.mbmus (sMap,Data);
    classe_ = class_ref[bmus_].reshape(NDmdl);   
    X_Mgeo_ = dto2d(classe_,L,C,isnum); # Classification géographique
    #plt.figure(); géré par l'appelant car ce peut être une fig déjà définie
    #et en subplot ... ou pas ...
    plt.imshow(X_Mgeo_, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    #
    classe_DD_, Tperf_, Perfglob_ = perfbyclass(classe_Dobs,classe_, nb_class);
    Tperf_ = np.round([iperf*100 for iperf in Tperf_]).astype(int); #print(Tperf_)    
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(Tperf_);
    hcb.ax.tick_params(labelsize=8);
    plt.axis('off');
    #grid(); # for easier check
    return Perfglob_
#
def datacodification4CT(data) :
    # Codification des données to CT
    Ndata, Ldata, Cdata = np.shape(data);
    #
    #if INDSC : # Indicateur de Saisonalité Climatologique
    #
    if TRENDLESS : # Suppression de la tendance pixel par pixel
        data = trendless(data);
    #
    if WITHANO :   # Anomalies : supression moy mens annuelle par pixel
        if 1 :     #<><>
            data = anomalies(data)
        elif 0 : # Apres Trendless, ca revient quasi au meme de Centrer
            X_ = data.reshape(Ndata,Ldata*Cdata);
            X_ = tls.centree(X_)
            data = X_.reshape(Ndata,Ldata,Cdata)
            del X_;
    #
    #-----
    #if UISST == "before" :  data = deltalargeM12(data);
    #-----
    # Climatologie: Moyennes mensuelles par pixel
    Ddata, Iisn_data, Inan_data  = Dpixmoymens(data,climato=climato);
    #-----
    if UISST : # == "after" :
        Ddata = deltalargeM12(Ddata,Ldata,Cdata,isnumobs);
    #----
    NDdata = len(Ddata);
    #----
    # Transfo Après mise sous forme de pixels Moyens Mensuels
    if NORMMAX == True :
        Maxi = np.max(Ddata, axis=1);
        Ddata = (Ddata.T / Maxi).T;
    #if NORMMAX == 2 :
    if CENTRED :
        Ddata  = tls.centred(Ddata,biais=0); # mais en fait ...
    #----
    return data, Ddata, NDdata;
#%%----------------------------------------------------------------------
# Des trucs qui pourront servir
tpgm0 = time();
plt.ion()
varnames = np.array(["JAN","FEV","MAR","AVR","MAI","JUI",
                    "JUI","AOU","SEP","OCT","NOV","DEC"]);
#######################################################################
#
#
#######################################################################
# PARAMETRAGE (#1) DU CAS
from ParamCasTestTopol import *
#======================================================================
#
#
#######################################################################
# ACQUISITION DES DONNEES D'OBSERVATION (et application des codifications)
#======================================================================
#%% Lecture des Obs____________________________________
if DATAOBS == "raverage_1975_2005" :
    if 0 : # Ca c'était avant
        #sst_obs = np.load("Datas/sst_obs_1854a2005_25L36C.npy")
        #lat: 29.5 à 5.5 ; lon: -44.5 à -9.5
        sst_obs  = np.load("Datas/sst_obs_1854a2005_Y60X315.npy");
    else :
        import netCDF4
        #nc      = netCDF4.Dataset("./Datas/raverage_1975-2005/ersstv3b_1975-2005_extract_LON-315-351_LAT-30-5.nc");
        nc      = netCDF4.Dataset("./Datas/raverage_1975-2005/ersstv5_1975-2005_extract_LON-315-351_LAT-30-5.nc");
        liste_var = nc.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess
        sst_var   = liste_var['sst']    # 1960 - 1930 + 1 = 31 ; 31 * 12 = 372
        sst_obs   = sst_var[:];         # np.shape = (372, 1, 25, 36)
        Nobs,Ncan,Lobs,Cobs = np.shape(sst_obs);
        if 0 : # visu obs
            showimgdata(sst_obs,fr=0,n=Nobs);
            plt.suptitle("Obs raverage 1930 à 1960")
            plt.show(); sys.exit(0)
        sst_obs   = sst_obs.reshape(Nobs,Lobs,Cobs); # np.shape = (144, 25, 36)
        sst_obs   = sst_obs.filled(np.nan);
#
elif DATAOBS == "raverage_1930_1960" :
    import netCDF4
    nc      = netCDF4.Dataset("./Datas/raverage_1930-1960/ersstv3b_1930-1960_extract_LON-315-351_LAT-30-5.nc");
    liste_var = nc.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess
    sst_var   = liste_var['sst']    # 1960 - 1930 + 1 = 31 ; 31 * 12 = 372
    sst_obs   = sst_var[:];         # np.shape = (372, 1, 25, 36)
    Nobs,Ncan,Lobs,Cobs = np.shape(sst_obs);
    if 0 : # visu obs
        showimgdata(sst_obs,fr=0,n=Nobs);
        plt.suptitle("Obs raverage 1930 à 1960")
        plt.show(); sys.exit(0)
    sst_obs   = sst_obs.reshape(Nobs,Lobs,Cobs); # np.shape = (144, 25, 36)
    sst_obs   = sst_obs.filled(np.nan);
#    
elif DATAOBS == "rcp_2006_2017" :
    import netCDF4
    nc      = netCDF4.Dataset("./Datas/rcp_2006-2017/ersst.v3b._2006-2017_extrac-zone_LON-315-351_LAT-30-5.nc");
    liste_var = nc.variables;       # mois par mois de janvier 2006 à decembre 2017 I guess
    sst_var   = liste_var['sst']    # 2017 - 2006 + 1 = 12 ; 12 * 12 = 144
    sst_obs   = sst_var[:];         # np.shape = (144, 1, 25, 36)
    Nobs,Ncan,Lobs,Cobs = np.shape(sst_obs);
    if 0 : # visu obs
        showimgdata(sst_obs,fr=0,n=Nobs);
        plt.suptitle("Obs rcp 2006 à 2017")
        plt.show(); sys.exit(0)
    sst_obs   = sst_obs.reshape(Nobs,Lobs,Cobs); # np.shape = (144, 25, 36)
    sst_obs   = sst_obs.filled(np.nan); 
#    
lat      = np.arange(29.5, 4.5, -1);
lon      = np.arange(-44.5, -8.5, 1);
Nobs,Lobs,Cobs = np.shape(sst_obs); print("obs.shape : ", Nobs,Lobs,Cobs);
#
# Selection___________________________________________
if Nda > 0 : # Ne prendre que les Nda dernières années (rem ATTENTION, toutes les ne commencent
    sst_obs = sst_obs[Nobs-(Nda*12):Nobs,]; #  pas à l'année 1850 ou 1854 ni au mois 01 !!!!!!!
    if 0 :
        vmin=np.nanmin(sst_obs);    vmax=np.nanmax(sst_obs)
        vmoy=np.nanmean(sst_obs);   vstd=np.nanstd(sst_obs) 
        showimgdata(sst_obs.reshape(372, 1, 25, 36),fr=0,n=4,vmin=vmin, vmax=vmax);
        plt.suptitle("min=%.4f, max=%.4f moy=%.4f, std=%.4f"%(vmin,vmax,vmoy,vstd))
        plt.show(); sys.exit(0)
#
# Paramétrage (#2) : _________________________________
print("-- SIZE_REDUCTION == '{}'".format(SIZE_REDUCTION))
if SIZE_REDUCTION == 'sel' or SIZE_REDUCTION == 'RED':
    # Définir une zone plus petite
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #    Lon > -28 et Lon < -16 et
    #    Lat > 10 et Lat < 23
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    frl = int(np.where(lat==22.5)[0]);
    tol = int(np.where(lat==9.5)[0]); # pour avoir 10.5, faut mettre 9.5
    lat = lat[frl:tol];
    frc = int(np.where(lon==-27.5)[0]);
    toc = int(np.where(lon==-15.5)[0]); # pour avoir 16.5, faut mettre 15.5
    lon = lon[frc:toc];
    print("   New LAT limits [{}, {}]".format(np.min(lat),np.max(lat)))
    print("   New LON limits [{}, {}]".format(np.min(lon),np.max(lon)))
if SIZE_REDUCTION == 'sel' :
    # Prendre d'entrée de jeu une zone plus petite
    sst_obs = sst_obs[:,frl:tol,frc:toc];
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Nobs, Lobs, Cobs = np.shape(sst_obs); print("obs.shape : ", Nobs, Lobs, Cobs);
Npix = Lobs*Cobs; # C'est sensé etre la même chose pour tous les mdl
#
# Définir une fois pour toutes, les indices des nan et non nan pour UNE SEULE
# image (sachant qu'on fait l'hypothese que pour toutes les images, les nans
# sont aux memes endroits). En principe ici les modèles sont alignés sur les Obs
X_       = sst_obs[0].reshape(Lobs*Cobs);
isnanobs = np.where(np.isnan(X_))[0];
isnumobs = np.where(~np.isnan(X_))[0];
del X_;
#%%_________________________
# Codification des Obs 4CT 
sst_obs, Dobs, NDobs = datacodification4CT(sst_obs);
#-------------------------
#
if WITHANO :
    #wvmin=-3.9; wvmax = 4.9; # ok pour obs 1975-2005 : ANO 4CT: min=-3.8183; max=4.2445 (4.9 because ...)
    #wvmin=-4.3; wvmax = 4.9; # ok pour obs 2006-2017 : ANO 4CT: min=-4.2712; max=4.3706
    wvmin = -4.9; wvmax = 4.9; # pour mettre tout le monde d'accord ?
else : # On suppose qu'il s'agit du brute ...
    wvmin =16.0; wvmax =30.0; # ok pour obs 1975-2005 : SST 4CT: min=16.8666; max=29.029
#    
if 0 : # Visu (et sauvegarde éventuelle de la figure) des données telles
       # qu'elles vont etre utilisées par la Carte Topologique
    minDobs = np.min(Dobs);   maxDobs=np.max(Dobs);
    moyDobs = np.mean(Dobs);  stdDobs=np.std(Dobs);
    if climato != "GRAD" :
        aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,figsize=(12,9)); #...
    else :
        aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,wvmin=0.0,wvmax=0.042,figsize=(12,9)); #...
    plt.suptitle("%sSST%d-%d). Obs for CT\nmin=%f, max=%f, moy=%f, std=%f"
                 %(fcodage,andeb,anfin,minDobs,maxDobs,moyDobs,stdDobs));
    if 0 : #SAVEFIG : # sauvegarde de la figure
        plt.savefig("%sObs4CT"%(fshortcode))
    #X_ = np.mean(Dobs, axis=1); X_ = X_.reshape(743,1); #rem = 0.0 when anomalies
    plt.show(); sys.exit(0)
#%%
#######################################################################
#
#
#
#######################################################################
#                       Carte Topologique
#======================================================================
nbl      = 30;  nbc =  4;  # Taille de la carte
#nbl      = 36;  nbc =  6;  # Taille de la carte
#nbl      = 52;  nbc =  8;  # Taille de la carte
#Parm_app = ( 5, 5., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
Parm_app = ( 500, 5., 1.,  1000, 1., 0.1); # Température ini, fin, nb_it
epoch1,radini1,radfin1,epoch2,radini2,radfin2 = Parm_app
#----------------------------------------------------------------------
tseed = 0; #tseed = 9; #tseed = np.long(time());
print("tseed=",tseed); np.random.seed(tseed);
#----------------------------------------------------------------------
# Création de la structure de la carte_______________
norm_method = 'data'; # je n'utilise pas 'var' mais je fais centred à
                      # la place (ou pas) qui est équivalent, mais qui
                      # me permet de garder la maitrise du codage
sMapO = SOM.SOM('sMapObs', Dobs, mapsize=[nbl, nbc], norm_method=norm_method, \
              initmethod='random', varname=varnames)
#print("NDobs(sm.dlen)=%d, dim(Dapp)=%d\nCT : %dx%d=%dunits" \
#      %(sMapO.dlen,sMapO.dim,nbl,nbc,sMapO.nnodes));
#
# Apprentissage de la carte _________________________
etape1=[epoch1,radini1,radfin1];    etape2=[epoch2,radini2,radfin2];
qerr = sMapO.train(etape1=etape1,etape2=etape2, verbose='off',retqerrflg=True);
# + err topo maison
bmus2O = ctk.mbmus (sMapO, Data=None, narg=2);
etO    = ctk.errtopo(sMapO, bmus2O); # dans le cas 'rect' uniquement
print("Obs, cas: {}".format(qerr))
print("Obs, quantization error = {:.4f}".format(qerr))
print("Obs, topological error  = {:.4f}".format(etO))
#
# Visualisation______________________________________
if 0 : #==>> la U_matrix
    a=sMapO.view_U_matrix(distance2=2, row_normalized='No', show_data='Yes', \
                      contooor='Yes', blob='No', save='No', save_dir='');
    plt.suptitle("Obs, The U-MATRIX", fontsize=16);
if 0 : #==>> La carte
    ctk.showmap(sMapO,sztext=11,colbar=1,cmap=cm.rainbow,interp=None);
    plt.suptitle("Obs, Les Composantes de la carte", fontsize=16);
#
#%% Other stuffs ______________________________________
bmusO  = ctk.mbmus (sMapO, Data=Dobs); # déjà vu ? conditionnellement ?
minref = np.min(sMapO.codebook);
maxref = np.max(sMapO.codebook);
Z_          = linkage(sMapO.codebook, method_cah, dist_cah);
class_ref   = fcluster(Z_,nb_class,'maxclust'); # Classes des referents
del Z_
#
coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
bounds = np.arange(nb_class+1)+1; # pour bounds faut une frontière de plus [1, 2, 3, 4, 5, 6, 7]
sztitle = 10;
#
# Transcodage des indices des classes
if TRANSCOCLASSE is not '' :
    class_ref = transco_class(class_ref,sMapO.codebook,crit=TRANSCOCLASSE);
#
classe_Dobs = class_ref[bmusO].reshape(NDobs); #(sMapO.dlen)
XC_Ogeo     = dto2d(classe_Dobs,Lobs,Cobs,isnumobs); # Classification géographique

#>
# Nombre de pixels par classe (pour les obs)
Nobsc = np.zeros(nb_class)
for c in np.arange(nb_class)+1 :
    iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
    Nobsc[c-1] = len(iobsc);
#%%<
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Pour différencier la zone entiere de la zone REDuite, je conviens que le o
# de obs sera en majuscule pour la zone entière (selectionnée).
# du coup, je duplique.
sst_Obs  = np.copy(sst_obs); NObs=Nobs; LObs=Lobs; CObs=Cobs;
isnumObs = isnumobs; XC_ogeo = XC_Ogeo; classe_DObs = classe_Dobs;
#
if SIZE_REDUCTION == 'RED' :
    #sst_obs, XC_Ogeo, classe_DObs, isnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,frl,tol,frc,toc);
    sst_obs, XC_ogeo, classe_Dobs, isnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,frl,tol,frc,toc);
    # si on ne passe pas ici, les petits o et les grand O sont égaux
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#plt.figure(); plt.imshow(XC_ogeo, interpolation='none',vmin=1,vmax=nb_class)
Nobs, Lobs, Cobs = np.shape(sst_obs)
NDobs  = len(classe_Dobs)
fond_C = np.ones(NDobs)
fond_C = dto2d(fond_C,Lobs,Cobs,isnumobs,missval=0.5)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#
if 1 : # for Obs
    plt.figure(figsize=(6,6) );
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb    = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("obs, classe géographique Method %s"%(method_cah),fontsize=16); #,fontweigth='bold');
    if SIZE_REDUCTION == 'All' :
        lolast = 4
    else :
        lolast = 2
    if 1 :
        plt.xticks(np.arange(0,Cobs,lolast), lon[np.arange(0,Cobs,lolast)], rotation=45, fontsize=10)
        plt.yticks(np.arange(0,Lobs,lolast), lat[np.arange(0,Lobs,lolast)], fontsize=10)
    else :
        plt.xticks(np.arange(-0.5,Cobs,lolast), np.round(lon[np.arange(0,Cobs,lolast)]).astype(int), rotation=45, fontsize=10)
        plt.yticks(np.arange(0.5,Lobs,lolast), np.round(lat[np.arange(0,Lobs,lolast)]).astype(int), fontsize=10)
    #grid(); # for easier check
    #plt.show(); sys.exit(0)
#
if 0 : # for obs
    plt.figure(figsize=(12,6) );
    TmoymensclassObs = moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class)
    #plt.plot(TmoymensclassObs); plt.axis('tight');
    for i in np.arange(nb_class) :
            plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
    plt.axis('tight');
    plt.xlabel('mois');
    plt.legend(np.arange(nb_class)+1,loc=2,fontsize=8);
    plt.title("obs, Moy. Mens. par Classe Method %s"%(method_cah),fontsize=16);
    #plt.show(); sys.exit(0)
#
if 0 :
    fig = plt.figure(figsize=(6,10) );
    ctk.showprofils(sMapO, figure=fig, Data=Dobs,visu=3, scale=2,Clevel=class_ref-1,Gscale=0.5,
                ColorClass=pcmap);
    #plt.show(); sys.exit(0)
#
#######################################################################
#
#
#%%
import os
print("whole time code %s: %f" %(os.path.basename(sys.argv[0]), time()-tpgm0));

#======================================================================
