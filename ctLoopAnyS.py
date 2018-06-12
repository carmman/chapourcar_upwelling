# -*- coding: cp1252 -*-
import sys
import time as time
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
from ParamCas import *
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
tseed = 0; #tseed = 9; #tseed = np.long(time());
print("tseed=",tseed); np.random.seed(tseed);
#----------------------------------------------------------------------
# Création de la structure de la carte_______________
norm_method = 'data'; # je n'utilise pas 'var' mais je fais centred à
                      # la place (ou pas) qui est équivalent, mais qui
                      # me permet de garder la maitrise du codage
sMapO = SOM.SOM('sMapObs', Dobs, mapsize=[nbl, nbc], norm_method=norm_method, \
              initmethod='random', varname=varnames)
print("NDobs(sm.dlen)=%d, dim(Dapp)=%d\nCT : %dx%d=%dunits" \
      %(sMapO.dlen,sMapO.dim,nbl,nbc,sMapO.nnodes));
#
# Apprentissage de la carte _________________________
etape1=[epoch1,radini1,radfin1];    etape2=[epoch2,radini2,radfin2];
sMapO.train(etape1=etape1,etape2=etape2, verbose='on');
# + err topo maison
bmus2O = ctk.mbmus (sMapO, Data=None, narg=2);
etO    = ctk.errtopo(sMapO, bmus2O); # dans le cas 'rect' uniquement
print("Obs, erreur topologique = %.4f" %etO)
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
    if 0 :
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
#######################################################################
#                        MODELS STUFFS START HERE
#======================================================================
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#       INITILISATIONS EN AMONT de LA BOUCLE SUR LES MODELES
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# For (sub)plot by modele
nsub   = 49; # actuellement au plus 48 modèles + 1 pour les obs.      
#nsub  = 9;  # Pour Michel (8+1pour les obs)     
nbsubc = np.ceil(np.sqrt(nsub));
nbsubl = np.ceil(1.0*nsub/nbsubc);
isubplot=0;
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
min_moymensclass = 999999.999999; # sert pour avoir tous les ...
max_moymensclass = 000000.000000; # ... subplots à la même échelles
TypePerf         = ["AccuracyPrecision"]; #,"Index2Rand"];
Tperfglob        = np.zeros((Nmodels,len(TypePerf))); # Tableau des Perf globales des modèles
if NIJ==1 :
    TNIJ         = [];  
TTperf           = [];  
#
TDmdl4CT         = []; # Stockage des modèles 4CT pour AFC-CAH ...  
#
Tmdlok           = []; # Pour construire une table des modèles valides
Nmdlok           = 0;  # Pour si y'a cumul ainsi connaitre l'indice de modèle valide 
                       # courant, puis au final, le nombre de modèles valides
                       # quoique ca dependra aussi que SUMi(Ni.) soit > 0                   
Tperfglob4Sort   = []; #!!??
Tclasse_DMdl     = []; #!!??
Tmdlname         = []; #!!??
Tmoymensclass    = []; #!!??
#%%
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#           PREMIERE BOUCLE SUR LES MODELES START HERE
#           PREMIERE BOUCLE SUR LES MODELES START HERE
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
print("ooooooooooooooooooooooooooooo first loop ooooooooooooooooooooooooooooo")
for imodel in np.arange(Nmodels) :
    mdlname = Tmodels[imodel,0]; print(mdlname)
    anstart = Tmodels[imodel,1]; # (utile pour rmean seulement)
    #________________________________________________
    # Lecture des données
    if DATAMDL=="raverage_1975_2005" : # fichiers.mat générés par Carlos
        datalib = 'Datas/raverage_1975-2005/sst_'
        try :
            sst_mat = scipy.io.loadmat(datalib+mdlname+"_raverage_1975-2005.mat");
        except :
            continue;
        sst_mdl = sst_mat['SST'];       
    elif DATAMDL=="raverage_1930_1960" : # fichiers.mat générés par Carlos
        datalib = 'Datas/raverage_1930-1960/sst_'
        try :
            sst_mat = scipy.io.loadmat(datalib+mdlname+"_raverage_1930-1960.mat");
        except :
            continue;
        sst_mdl = sst_mat['SST'];      
    elif DATAMDL == "rcp_2006_2017": # fichiers.mat scénarios générés par Carlos.
        # dédiés à l'étude de la généralisation
        datalib = "Datas/rcp_2006-2017/%s/sst_"%scenar
        try :
            sst_mat = scipy.io.loadmat(datalib+mdlname+"_raverage_2006-2017.mat");
        except :
            continue;
        sst_mdl = sst_mat['SST'];
    Nmdl, Lmdl, Cmdl = np.shape(sst_mdl); # print("mdl.shape : ", Nmdl, Lmdl, Cmdl);
    #
    Nmdlok = Nmdlok + 1; # Pour si y'a cumul ainsi connaitre l'indice de modèle 
             # valide courant, puis au final, le nombre de modèles valides.
             # (mais ...)
    # Là je construis une table des modèles valides
    Tmdlok.append(Tmodels[imodel]);
    #
    if MDLCOMPLETION : # Complémentation des données modèles de sorte à ce que seul
        nnan=1;        # le mappage des nans d'obs soit utilisé
        while nnan > 0 :
            sst_mdl, nnan = nan2moy(sst_mdl, Left=1, Above=0, Right=0, Below=0)
    #________________________________________________
    # Selection
    if SIZE_REDUCTION=='sel' : # Prendre une zone plus petite
        sst_mdl = sst_mdl[:,frl:tol,frc:toc];
    Nmdl, Lmdl, Cmdl = np.shape(sst_mdl); # print("mdl.shape : ", Nmdl, Lmdl, Cmdl); 
    #
    #________________________________________________
    if 1 :
        # Pour etre plus correct prendre l'union des nans
        # Dans le cas de la complétion des modèles, ce sont les nans des obs qui s'appliquent
        X_O   = np.reshape(sst_Obs, NObs*LObs*CObs);
        X_M   = np.reshape(sst_mdl, Nmdl*Lmdl*Cmdl);
        IOnan = np.where(np.isnan(X_O))[0];
        IMnan = np.where(np.isnan(X_M))[0];
        Inan  = np.union1d(IOnan,IMnan)
        X_O[Inan] = np.nan;
        sst_Obs = np.reshape(X_O, (NObs,LObs,CObs));

        if mdlname == "FGOALS-s2" and DATAMDL == "raverage_1975_2005" :
            # Pour "FGOALS-s2", il ne faut pas prendre les nans qui correspondent aux 12
            # derniers mois car FGOALS-s2 termine en 2004 et non pas 2005
            # (pour DATARUN=="rmean", anstart="" on ne devrait donc pas passer ici
            Nnan = len(Inan)
            Nnan = int(Nnan - Nnan/12)
            Inan = Inan[0:Nnan];
            mdlname = "FGOALS-s2(2004)" # au passage
        X_M[Inan]= np.nan;
        sst_mdl = np.reshape(X_M, (Nmdl,Lmdl,Cmdl));
        del X_O, X_M
    #________________________________________________________
    # Codification du modèle (4CT)            
    sst_mdl, Dmdl, NDmdl = datacodification4CT(sst_mdl);
    #________________________________________________________
    if 0 : # Visu du modèle (moyen) #bok
        minDmdl = np.min(Dmdl);   maxDmdl=np.max(Dmdl);
        moyDmdl = np.mean(Dmdl);  stdDmdl=np.std(Dmdl);
        aff2D(Dmdl,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,figsize=(12,9));
        plt.suptitle("%s %s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
                     %(mdlname,fcodage,andeb,anfin,minDmdl,maxDmdl,moyDmdl,stdDmdl));
        #continue; #plt.show(); sys.exit(0);
        #X_ = np.mean(Dmdl, axis=1); X_=X_.reshape(743,1); #rem = 0.0 when anomalie
        #aff2D(X_,Lobs,Cobs,isnumobs,isnanobs,wvmin=-3.8?,wvmax=4.9?,figsize=(12,9));
    #________________________________________________________
    TDmdl4CT.append(Dmdl);  # stockage des modèles 4CT pour AFC-CAH ...
    Tmdlname.append(Tmodels[imodel,0])
    # Calcul de la perf glob du modèle et stockage pour tri
    bmusM       = ctk.mbmus (sMapO, Data=Dmdl);
    classe_DMdl = class_ref[bmusM].reshape(NDmdl);
    perfglob    = len(np.where(classe_DMdl==classe_Dobs)[0])/NDobs
    Tperfglob4Sort.append(perfglob)
    Tclasse_DMdl.append(classe_DMdl)
    #
    if OK106 : # Stockage (if required) pour la Courbes des moyennes mensuelles par classe
        Tmoymensclass.append(moymensclass(sst_mdl,isnumobs,classe_Dobs,nb_class));
#
# Fin de la PREMIERE boucle sur les modèles
#
#%%_____________________________________________________________________
######################################################################
# Reprise de la boucle (avec les modèles valides du coup).
# (question l'emplacement des modèles sur les figures ne devrait pas etre un problème ?)
#*****************************************
del Tmodels
del Tmdlok
#_________________________________________
# TRI et Reformatage des tableaux si besoin
Nmodels = len(TDmdl4CT);            ##!!??
if 1 : # Sort On Perf
    IS_ = np.argsort(Tperfglob4Sort)
    IS_= np.flipud(IS_)
else : # no sort
    IS_ = np.arange(Nmodels)
X1_ = np.copy(TDmdl4CT)
X2_ = np.copy(Tmdlname)
X3_ = np.copy(Tclasse_DMdl)
for i in np.arange(Nmodels) : # TDmdl4CT = TDmdl4CT[I_];
    TDmdl4CT[i]     = X1_[IS_[i]]
    Tmdlname[i]     = X2_[IS_[i]]  
    Tclasse_DMdl[i] = X3_[IS_[i]]
del X1_, X2_, X3_;
if OK106 :
    X1_ = np.copy(Tmoymensclass);
    for i in np.arange(Nmodels) :
        Tmoymensclass[i] = X1_[IS_[i]]
    del X1_   
    Tmoymensclass    = np.array(Tmoymensclass);
    min_moymensclass = np.nanmin(Tmoymensclass); ##!!??
    max_moymensclass = np.nanmax(Tmoymensclass); ##!!??
#*****************************************
MaxPerfglob_Qm  = 0.0; # Utilisé pour savoir les quels premiers modèles
IMaxPerfglob_Qm = 0;   # prendre dans la stratégie du "meilleur cumul moyen"
#*****************************************
# l'Init des figures à produire doit pouvoir etre placé ici ##!!?? (sauf la 106)
if OK104 : # Classification avec, "en transparance", les mals classés
           # par rapport aux obs
    plt.figure(104,figsize=(18,9),facecolor='w')
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle104="%sSST(%s)). %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah);
if OK105 : #Classification
    plt.figure(105,figsize=(18,9))
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle105="%sSST(%s)). %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah);
if OK106 : # Courbes des moyennes mensuelles par classe
    plt.figure(106,figsize=(18,9),facecolor='w'); # Moyennes mensuelles par classe
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle106="MoyMensClass(%sSST(%s)).) %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah);
if OK107 : # Variance (not 'RED' compatible)
    suptitle107="VARiance(%sSST(%s)).) Variance (by pixel) on Completed Models" \
                 %(fcodage,DATAMDL);
    Dmdl_TVar  = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance par pixel sur climatologie
                   # J'utiliserais ainsi showimgdata pour avoir une colorbar commune
#
if MCUM  : # # Moyenne des Models climatologiques CUmulés
    DMdl_Q  = np.zeros((NDobs,12));  # f(modèle climatologique Cumulé)
    DMdl_Qm = np.zeros((NDobs,12));  # f(modèle climatologique Cumulé moyen)
#
# Moyenne CUMulative
if OK108 : # Classification en Model Cumulé Moyen
    plt.figure(108,figsize=(18,9),facecolor='w'); # Moyennes mensuelles par classe
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle108="MCUM - %sSST(%s)). %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah);
#
# Variance CUMulative
if OK109 : # Variance sur les Models Cumulés Moyens (not 'RED' compatible)
    Dmdl_TVm = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance sur climatologie
               # cumulée, moyennée par pixel. J'utiliserais ainsi showimgdata pour avoir
               # une colorbar commune
    suptitle109="VCUM - %sSST(%s)). Variance sur la Moyenne Cumulée de Modeles complétés" \
                 %(fcodage,DATAMDL);
#%%
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#           DEUXIEME BOUCLE SUR LES MODELES START HERE
#           DEUXIEME BOUCLE SUR LES MODELES START HERE
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
isubplot = 0;
print("ooooooooooooooooooooooooooooo 2nd loop ooooooooooooooooooooooooooooo")
#for imodel in np.arange(8) :  # pour les figures de Michel
for imodel in np.arange(Nmodels) : ##!!??
    isubplot=isubplot+1;
    #
    Dmdl    = TDmdl4CT[imodel];
    mdlname = Tmdlname[imodel]; #print(mdlname)
    # 
    classe_DMdl = Tclasse_DMdl[imodel];
    XC_Mgeo     = dto2d(classe_DMdl,LObs,CObs,isnumObs); # Classification géographique
    #
    #>>>>>>>>>>> (à revoir)
    classe_Dmdl = classe_DMdl; # ... because RED ... du coup je duplique            pour avoir les memes
    XC_mgeo     = XC_Mgeo;     # ... because RED ... du coup je duplique            noms de variables.
    if 0 : # SIZE_REDUCTION == 'RED' :
        #sst_Mdl  = np.copy(sst_mdl); NObs=Nmdl; LObs=Lmdl; CObs=Cmdl;
        #sst_mdl, XC_mgeo, classe_Dmdl, isnum_red = red_classgeo(sst_Mdl,isnumObs,classe_DMdl,frl,tol,frc,toc);
        sst_mdl, XC_mgeo, classe_Dmdl, isnum_red = red_classgeo(sst_mdl,isnumObs,classe_DMdl,frl,tol,frc,toc);
    #<<<<<<<<<<<
    #
    classe_DD, Tperf, Perfglob = perfbyclass(classe_Dobs,classe_Dmdl,nb_class);
    #else : # Perf Indice de Rand (does not work yet) voir version précédante
    #
    Tperf = np.round([i*100 for i in Tperf]).astype(int); #print(Tperf)
    TTperf.append(Tperf);           # !!!rem AFC est faite avec ca    
    Tperfglob[imodel,0] = Perfglob;
    #
    Nmdl, Lmdl, Cmdl = np.shape(sst_mdl)    ##!!??
    NDmdl = len(classe_Dmdl); # needed for 108, ...? 
    #
    #%%%> l'AFC pourrait aussi etre faite avec ça
    if NIJ == 1 : # Nij = card|classes| ; Pourrait semblé inapproprié car les classes
                  # peuvent géographiquement être n'importe où, mais ...               
        Znij_ = [];  
        for c in np.arange(nb_class) :
            imdlc = np.where(classe_Dmdl==c+1)[0]; # Indices des classes c du model
            Znij_.append(len(imdlc));  
        TNIJ.append(Znij_);  
    #%%%<
    #:::>------------------------------------------------------------------
    '''FAUDRA VOIR QUEL INDICE ON MET SUR CHAQUE MODEL,
    ET SI IL FAUDRA FAIRE LA COLORBAR PAR CLASSE SELON CET INDICE !!!??? si c'est possible !!!???
    pour le moment faut faire attention car ce n'est pas maitrisé !!!!!!
    '''
    ip = 0; 
    if "Index2Rand" in TypePerf :
        ip = ip+1;
        Tperfglob[imodel,ip] = adjusted_rand_score(classe_Dobs, classe_Dmdl);
    #<:::------------------------------------------------------------------
    #
    if 0 : # print ponctuels de mise au point ?
        print("Modèle: %s ; Method: %s"%(mdlname, method_cah));
        print("Perf toutes classes = %.4f"%(Perfglob))
        print("Perf par classe : ");
        tls.tprin(np.arange(nb_class)+1, "  %d ");
        tls.tprin(Tperf, " %3d");
    #
    if OK104 : # Classification avec, "en transparance", les pixels mals
               # classés par rapport aux obs. (pour les modèles les Perf
               # par classe sont en colorbar)
        plt.figure(104); plt.subplot(nbsubl,nbsubc,isubplot);
        X_ = dto2d(classe_DD,Lobs,Cobs,isnumobs); #X_= classgeo(sst_obs, classe_DD);
        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        if FONDTRANS == "Obs" :
            plt.imshow(XC_ogeo, interpolation='none', cmap=ccmap, alpha=0.2,vmin=1,vmax=nb_class);
        elif FONDTRANS == "Mdl" :
            plt.imshow(XC_mgeo, interpolation='none', cmap=ccmap, alpha=0.2,vmin=1,vmax=nb_class);
        plt.imshow(X_, interpolation='none', cmap=ccmap,vmin=1,vmax=nb_class);
        del X_
        plt.axis('off'); #grid(); # for easier check
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf);
        hcb.ax.tick_params(labelsize=8)
        #
    if OK105 : # Classification (pour les modèles les Perf par classe sont en colorbar)
        plt.figure(105); plt.subplot(nbsubl,nbsubc,isubplot);
        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        plt.imshow(XC_mgeo, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf);
        hcb.ax.tick_params(labelsize=8);
        plt.axis('off');
        #grid(); # for easier check
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle);
        #
    if OK106 : # Courbes des moyennes mensuelles par classe
        plt.figure(106); plt.subplot(nbsubl,nbsubc,isubplot);
        for i in np.arange(nb_class) :
            plt.plot(Tmoymensclass[imodel,:,i],'.-',color=pcmap[i]);
        plt.axis([0, 11, min_moymensclass, max_moymensclass]); 
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle);
        #
    if OK107 : # Variance (not 'RED' compatible)
        Dmdl_TVar[imodel] = np.var(Dmdl, axis=1, ddof=0);
        #
    if MCUM : # Cumul et moyenne
        DMdl_Q  = DMdl_Q + Dmdl;       # Cumul Zone
        DMdl_Qm = DMdl_Q / (imodel+1); # Moyenne Zone
        #
    if OK108 : # Classification en Model Cumulé Moyen (Perf par classe en colorbar)
        plt.figure(108); plt.subplot(nbsubl,nbsubc,isubplot);
        bmusMdl_Qm = ctk.mbmus (sMapO, Data=DMdl_Qm);
        classe_DMdl_Qm= class_ref[bmusMdl_Qm].reshape(NDmdl);
                       # Ici classe_D* correspond à un résultats de classification
                       # (bon ou mauvais ; donc sans nan)
        XC_mgeo_Qm    = dto2d(classe_DMdl_Qm,Lobs,Cobs,isnumobs); # Classification géographique
                       # Mise sous forme 2D de classe_D*, en mettant nan pour les
                       # pixels mas classés
        classe_DD_Qm, Tperf_Qm, Perfglob_Qm = perfbyclass(classe_Dobs, classe_DMdl_Qm, nb_class);
                       # Ici pour classe_DD* : les pixels bien classés sont valorisés avec
                       # leur classe, et les mals classés ont nan
        Tperf_Qm = np.round([i*100 for i in Tperf_Qm]).astype(int); #print(Tperf)

        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        plt.imshow(XC_mgeo_Qm, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf_Qm);
        hcb.ax.tick_params(labelsize=8);
        plt.axis('off');
        #grid(); # for easier check
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob_Qm,'%'),fontsize=sztitle);
        #
        pgqm_ = np.round_(Perfglob_Qm*100)
        if pgqm_ >= MaxPerfglob_Qm :
            MaxPerfglob_Qm  = pgqm_;     # Utilisé pour savoir les quels premiers modèles
            IMaxPerfglob_Qm = imodel+1;  # prendre dans la stratégie du "meilleur cumul moyen"
            print("New best cumul perf for %dmodels : %d%c"%(imodel+1,pgqm_,'%'))
     #
    if OK109 : # Variance sur les Models Cumulés Moyens (not 'RED' compatible)
                          # Perf par classe en colorbar)
        Dmdl_TVm[imodel] = np.var(DMdl_Qm, axis=1, ddof=0);
#
# Fin de la DEUXIEME boucle sur les modèles
#%%__________________________________________
# Les Obs à la fin 
isubplot = 49; 
#isubplot = isubplot + 1; # Michel (ou pas ?)
if OK104 : # Obs for 104
    plt.figure(104); plt.subplot(nbsubl,nbsubc,isubplot);
    plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
    plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    #grid(); # for easier check
    plt.suptitle(suptitle104)
    if SAVEFIG :
        #plt.savefig("%s%s%dMdlvsObstrans"%(fshortcode,method_cah,nb_class))
        plt.savefig("%s%s_%s%s%dMdlvsObstrans"%(fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
if OK105 : # Obs for 105
    plt.figure(105); plt.subplot(nbsubl,nbsubc,isubplot);
    plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
    plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    #grid(); # for easier check
    plt.suptitle(suptitle105)
    if SAVEFIG :
        #plt.savefig("%s%s%dMdl"%(fshortcode,method_cah,nb_class))
        plt.savefig("%s%s_%s%s%dMdl"%(fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
if OK106 : # Obs for 106
    plt.figure(106); plt.subplot(nbsubl,nbsubc,isubplot);
    TmoymensclassObs = moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class); 
    for i in np.arange(nb_class) :
        plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
    plt.axis([0, 11, min_moymensclass, max_moymensclass]);
    plt.xlabel('mois');
    plt.xticks(np.arange(12), np.arange(12)+1, fontsize=8)
    plt.legend(np.arange(nb_class)+1,loc=2,fontsize=6,numpoints=1,bbox_to_anchor=(1.1, 1.0));
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    #
    # On repasse sur tous les supblots pour les mettre à la même echelle.
    print(min_moymensclass, max_moymensclass);
    plt.suptitle(suptitle106)
    if SAVEFIG :
        #plt.savefig("%s%s%dmoymensclass"%(fshortcode,method_cah,nb_class))
        plt.savefig("%s%s_%s%s%dmoymensclass"%(fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
#
if OK107 or OK109 : # Calcul de la variance des obs par pixel de la climatologie
    Tlabs = np.copy(Tmdlname);   
    Tlabs = np.append(Tlabs,'');                # Pour le subplot vide
    Tlabs = np.append(Tlabs,'Observations');    # Pour les Obs
    varobs= np.ones(Lobs*Cobs)*np.nan;          # Variances des ...
    varobs[isnumobs] = np.var(Dobs, axis=1, ddof=0); # ... Obs par pixel
#
if OK107 : # Variance par pixels des modèles
    X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
    X_[:,isnumobs] = Dmdl_TVar
    # Rajouter nan pour le subplot vide
    X_    = np.concatenate(( X_, np.ones((1,Lobs*Cobs))*np.nan))
    # Rajout de la variance des obs
    X_    = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
    #
    showimgdata(X_.reshape(Nmodels+2,1,Lobs,Cobs), Labels=Tlabs, n=Nmodels+2,fr=0,
                vmin=np.nanmin(Dmdl_TVar),vmax=np.nanmax(Dmdl_TVar),fignum=107);
    del X_
    plt.suptitle(suptitle107);
    if SAVEFIG :
        plt.savefig("%sVAR_%s_%sMdl"%(fprefixe,SIZE_REDUCTION,fshortcode))
#
if OK108 : # idem OK105, but ...
    plt.figure(108); plt.subplot(nbsubl,nbsubc,isubplot);
    plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
    plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    #grid(); # for easier check
    plt.suptitle(suptitle108);
    if SAVEFIG :
        plt.savefig("%sMCUM_%s_%s%s%dMdl"%(fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
#
if OK109 : # Variance par pixels des moyenne des modèles cumulés
    X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
    X_[:,isnumobs] = Dmdl_TVm
    # Rajouter nan pour le subplot vide
    X_ = np.concatenate(( X_, np.ones((1,Lobs*Cobs))*np.nan))
    # Rajout de la variance des obs
    X_ = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
    #
    showimgdata(X_.reshape(Nmodels+2,1,Lobs,Cobs), Labels=Tlabs, n=Nmodels+2,fr=0,
                vmin=np.nanmin(Dmdl_TVm),vmax=np.nanmax(Dmdl_TVm),fignum=109);
    del X_
    plt.suptitle(suptitle109);
    if SAVEFIG :
        plt.savefig("%sVCUM_%s_%sMdl"%(fprefixe,SIZE_REDUCTION,fshortcode))
#
##---------------------------------------------------------------------
# Redimensionnement de Tperfglob au nombre de modèles effectif
Tperfglob = Tperfglob[0:Nmodels]; 
#
# Edition des résultats
if 0 : # (print) Tableau des performances
    print("% de Perf global");
    Tperfglob = np.round(Tperfglob,2)*100
    Tperfglob = Tperfglob.astype(int);
    for i in np.arange(Nmodels) :
        print(Tperfglob[i])
#:::>
if 1 : # Tableau des performances en figure de courbes
    plt.figure(facecolor='w'); plt.plot(Tperfglob,'.-');
    plt.axis("tight"); plt.grid('on')
    plt.xticks(np.arange(Nmodels),Tmdlname, fontsize=8, rotation=45,
               horizontalalignment='right', verticalalignment='baseline');
    plt.legend(TypePerf,numpoints=1,loc=3)
    plt.title("%sSST(%s)) %s%d Indice(s) de classification of Completed Models (vs Obs)"\
                 %(fcodage,DATAMDL,method_cah,nb_class));
#<:::
#
#
# Mettre les Tableaux-Liste en tableau Numpy
Tmdlname = np.array(Tmdlname);
TTperf   = np.array(TTperf);
if NIJ==1 :
    TNIJ = np.array(TNIJ);
TDmdl4CT = np.array(TDmdl4CT);
#%%%>==================================================================
#%%%>------------------------------------------------------------------
if NIJ > 0 : # A.F.C
    #Ajout de l'indice dans le nom du modèle
    Tm_ = np.empty(len(Tmdlname),dtype='<U32');
    for i in np.arange(Nmdlok) : #a 
        Tm_[i] = str(i+1) + '-' +Tmdlname[i];
    #
    # Harmonaiser la variable servant de tableau de contingence (->Tp_), selon le cas
    if NIJ==1 : # Nij = card|classes| ; Pourrait semblé inapproprié car les classes
                # peuvent géographiquement être n'importe où, mais ...
        Tp_ = TNIJ; # TNIJ dans ce cas a été préparé en amont
        #
    elif NIJ==2 or NIJ==3 :
        Tp_ = TTperf; # Pourcentages des biens classés par classe (transformé ci après
                      # pour NIJ==3 en effectif après éventuel ajout des obs dans l'afc)
    # On supprime les lignes dont la somme est 0 car l'afc n'aime pas ca.
    # (en espérant que la suite du code reste cohérente !!!???)
    som_ = np.sum(Tp_, axis=1);
    Iok_ = np.where(som_>0)[0]
    Tp_  = Tp_[Iok_];
    Tm_  = Tm_[Iok_];
    Nmdlok = len(Iok_); # !!!Attention!!! redéfinition du nombre de modèles valides
    del som_, Iok_;
    #
    if AFCWITHOBS : # On ajoute Obs (si required)
        if NIJ==1 :
            Tp_ = np.concatenate((Tp_, Nobsc[np.newaxis,:]), axis=0).astype(int); 
        else :
            # Obs have 100% for any class par définition
            pobs_ = (np.ones(nb_class)*100).astype(int);
            Tp_   = np.concatenate((Tp_, pobs_[np.newaxis,:]), axis=0); # je mets les Obs A LA FIN
        lignames  = list(Tm_);
        lignames.append("Obs"); # Obs sera le dernier
    else :  
        Tp_      = Tp_[0:Nmodels-1];
        lignames = Tm_[0:Nmodels-1];
    #
    if NIJ == 3 : # On transforme les %tages en Nombre (i.e. en effectif)
        if 0 : # icicicicici
            Tp_ = Tp_ * Nobsc / 100;
        else : 
            Tp_ = np.round(Tp_ * Nobsc / 100).astype(int); ##$$
        #Tp_ = Nobsc - Tp_; # j'essaye ca ...
    # _________________________
    # Faire l'AFC proprment dit
    if AFCWITHOBS :
        VAPT, F1U, CAi, CRi, F2V, CAj, F1sU = afaco(Tp_);
        XoU = F1U[Nmdlok,:]; # coord des Obs #a #?
    else : # Les obs en supplémentaire
        VAPT, F1U, CAi, CRi, F2V, CAj, F1sU = afaco(Tp_, Xs=[Nobsc]);
        XoU = F1sU; # coord des Obs (ok j'aurais pu mettre directement en retour de fonction...)
    #
    #-----------------------------------------
    # MODELE MOYEN (pondéré ou pas) PAR CLUSTER D'UNE CAH
    if 1 : # CAH on afc Models's coordinates (without obs !!!???)
        metho_ = 'ward'; 
        dist_  = 'euclidean';
        coord2take = np.arange(NBCOORDAFC4CAH)
        if AFCWITHOBS :
            # (Mais ne pas prendre les obs dans la CAH (ne prendre que les modèles))
            Z_ = linkage(F1U[0:Nmdlok,coord2take], metho_, dist_);
        else :
            Z_ = linkage(F1U[:,coord2take], metho_, dist_);
        if 1 : # dendrogramme
            plt.figure(figsize=(16,12));
            R_ = dendrogram(Z_,Nmdlok,'lastp');
            L_ = np.array(lignames)
            plt.xticks((np.arange(len(Tmdlname))*10)+7,L_[R_['leaves']], fontsize=11,
                   rotation=45,horizontalalignment='right', verticalalignment='baseline')
            del R_, L_
            plt.title("AFC: Coord(%s), dendro. Métho=%s, dist=%s, nb_clust=%d"
                      %((coord2take+1).astype(str),metho_,dist_,nb_clust))
        #
        class_afc = fcluster(Z_,nb_clust,'maxclust');
        #
        figclustmoy = plt.figure(figsize=(16,12));
        for ii in np.arange(nb_clust) :
            iclust  = np.where(class_afc==ii+1)[0];
            #
            # Visu des Classif des modèles des cluster
            if  ii+1 in AFC_Visu_Classif_Mdl_Clust :
                plt.figure(figsize=(16,12));
                for jj in np.arange(len(iclust)) :
                    bmusj_   = ctk.mbmus (sMapO, Data=TDmdl4CT[iclust[jj]]);
                    classej_ = class_ref[bmusj_].reshape(NDmdl);
                    XCM_     = dto2d(classej_,LObs,CObs,isnumObs); # Classification géographique
                    plt.subplot(7,7,jj+1);
                    plt.imshow(XCM_, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
                    plt.axis('off');
                    plt.title(Tm_[iclust[jj]],fontsize=sztitle)
                plt.suptitle("Classification des modèles du cluster %d"%(ii+1));
            if 1 :    
                print("%d Modèles du cluster %d :\n"%(len(iclust),ii+1), Tmdlname[iclust]); ##!!??
            #
            # Modèle Moyen d'un cluster
            if 1 : # Non pondéré
                CmdlMoy  = Dmdlmoy4CT(TDmdl4CT,iclust,pond=None);
            elif 0 : # Pondéré                
                if 1 : # par les contributions relatives ligne (i.e. modèles) (CRi) du plan
                    pond = np.sum(CRi[:,[pa-1,po-1]],axis=1);
                elif 0 : # par les contributions absolues ligne (i.e. modèles) (CAi) du plan
                    pond = np.sum(CAi[:,[pa-1,po-1]],axis=1) 
                elif 0 : # Proportionnelle à la perf global de chaque modèle ?
                    pond = Tperfglob / sum(Tperfglob) 
                CmdlMoy  = Dmdlmoy4CT (TDmdl4CT,iclust,pond=pond); # dans iclust y'a pas l'indice qui correspond aux Obs
            #
            #if 1 : # Affichage Data cluster moyen for CT
            if  ii+1 in AFC_Visu_Clust_Mdl_Moy_4CT :
                aff2D(CmdlMoy,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,figsize=(12,9));
                plt.suptitle("MdlMoy clust%d %s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
                        %(ii+1,fcodage,andeb,anfin,np.min(CmdlMoy),
                          np.max(CmdlMoy),np.mean(CmdlMoy),np.std(CmdlMoy)))
            #
            # Classification du modèles moyen d'un cluster
#           plt.figure(figclustmoy.number); plt.subplot(3,3,ii+1);
            plt.figure(figclustmoy.number,figsize=(16,12)); plt.subplot(4,4,ii+1);
            Perfglob_ = Dgeoclassif(sMapO,CmdlMoy,LObs,CObs,isnumObs);
            plt.title("cluster %d, perf=%.0f%c"%(ii+1,100*Perfglob_,'%'),fontsize=sztitle); #,fontweigth='bold');
        # FIN de la boucle sur le nombre de cluster
        del metho_, dist_, Z_
    # FIN du if 1 : MODELE MOYEN (pondéré ou pas) PAR CLUSTER D'UNE CAH
    #-----------------------------------------
    #      
    if 1 : # Calcul de distance (d^2) des coordonnées des modèles à celles des Obs
        #!!! Attention : cette partie de code semble ne pas avoir été adaptée aux évolutions !!!
        D2 = (F1U - XoU)**2; # Avec les coordonnées du plan, triangle dédoublé
        if 0 : # Avec les coordonnées du plan sommé : forme carré
            D2 = np.sum(D2[:,[pa-1,po-1]], axis=1)
        elif 1 : # Avec toutes les coordonnées
            D2   = np.sum(D2,axis=1); # forme carrée
            #iord = np.argsort(D2);
            iord = np.argsort(D2[0:Nmdlok-1]); # pour ne pas prendre les Obs elles-mêmes (ou BMU?) #a
            if 0 : # triangles dédoublés
                D2 = np.matlib.repmat(D2,nb_class-1,1);
                D2 = D2.T
                iord = np.argsort(D2[:,0]);
            N2take = 10;
            print("\nd2u :");
            for i in np.arange(N2take) :
                print("\'"+Tmdlname[iord[i]],end="\',"); #?
            print();
    #
    # Nuage de l'AFC
    #K=D2; xoomK=500;    # Pour les distances au carré (D2) des modèles aux Obs
    K=CRi; xoomK=1000; # Pour les contrib Rel (CRi)
    if 1 : # ori afc avec tous les points
        afcnuage(F1U,cpa=pa,cpb=po,Xcol=class_afc,K=K,xoomK=xoomK,linewidths=2,indname=lignames); #,cmap=cm.jet);
        if NIJ==1 :
            plt.title("%sSST(%s)). %s%d AFC on classes of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah,nb_class));
        elif NIJ==3 :
            plt.title("%sSST(%s)). %s%d AFC on good classes of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah,nb_class));
    # Limiter les points aux contributions les plus fortes
    def afclim(K,xoomK) :
        if 0 : # Sur les axes considérés
            lim_ = 0.028; # 0.03
            Ia_ = np.where(CAi[:,pa-1]>=lim_)[0];
            Io_ = np.where(CAi[:,po-1]>=lim_)[0];
            Ix_ = np.union1d(Ia_,Io_); 
            afcnuage(F1U[Ix_,:],cpa=pa,cpb=po,Xcol=np.arange(len(Ix_)),K=K[Ix_],xoomK=xoomK,linewidths=2,indname=lignames[Ix_]);
            #plt.title("%s%s(SST%d-%d)). %s%d AFC on good classes of Completed Models (vs Obs) (CtrA lim = %.3f)" \
            #     %(fcodage,DATARUN,andeb,anfin,method_cah,nb_class, lim_));
            plt.title("%sSST(%s)). %s%d AFC on good classes of Completed Models (vs Obs) (CtrA lim = %.3f)" \
                 %(fcodage,DATAMDL,method_cah,nb_class, lim_));
            del lim_, Ia_, Io_, Ix_
        if 1 : # Sur tous les axes
            lim_ = 0.13;
            X_   = np.sum(CAi, axis=1)
            Ix_  = np.where(X_>=lim_)[0];
            afcnuage(F1U[Ix_,:],cpa=pa,cpb=po,Xcol=np.arange(len(Ix_)),K=K[Ix_],xoomK=xoomK,linewidths=2,indname=lignames[Ix_]);
            #plt.title("%s%s(SST%d-%d)). %s%d AFC on good classes of Completed Models (vs Obs) (CtrA lim = %.3f)" \
            #     %(fcodage,DATARUN,andeb,anfin,method_cah,nb_class, lim_));
            plt.title("%sSST(%s)). %s%d AFC on good classes of Completed Models (vs Obs) (CtrA lim = %.3f)" \
                 %(fcodage,DATAMDL,method_cah,nb_class, lim_));
            del lim_, X_, Ix_
    if 0 :
        afclim(K,xoomK);
    #
    if AFCWITHOBS  : # Mettre en évidence Obs
        plt.plot(F1U[Nmdlok,pa-1],F1U[Nmdlok,po-1], 'oc', markersize=20,        # marker for obs
                 markerfacecolor='none',markeredgecolor='m',markeredgewidth=2);    
    else : # Obs en supplémentaire
        plt.text(F1sU[0,0],F1sU[0,1], ".Obs")
        plt.plot(F1sU[0,0],F1sU[0,1], 'oc', markersize=20,
                     markerfacecolor='none',markeredgecolor='m',markeredgewidth=2);
    #        
    if 1 : # AJOUT ou pas des colonnes (i.e. des classes)
        colnames = (np.arange(nb_class)+1).astype(str)
        afcnuage(F2V,cpa=pa,cpb=po,Xcol=np.arange(len(F2V)),K=CAj,xoomK=xoomK,
                 linewidths=2,indname=colnames,holdon=True) #,cmap=cm.jet);
        plt.axis("tight"); #?
        #
        if 0 : # Calcul des distances (d^2) des coordonnées des modèles F1U (lignes)
               # à celles des Classes F2V (colonne)
            # PLM, ici je ne le fait qu'avec toutes les coordonnées
            # (c'est plus simple pour n'utiliser que les meilleurs)
            N2take = 5; # Nombre de modèles à prendre
            F1U_ = F1U[0:Nmdlok-1]; # -1 pour ne pas prendre Obs (ou BNU?) #a
            for c in np.arange(nb_class) :
                print("\nd2v : c=%d"%(c+1))
                D2V = (F1U_ - F2V[c,:])**2; 
                D2V = np.sum(D2V,axis=1);
                iord = np.argsort(D2V);
                moydc = np.mean(D2V[iord[0:N2take]]); #stddc = np.std(D2V[iord[0:N2take]])
                for i in np.arange(N2take) : 
                    plt.plot( [F2V[c,pa-1], F1U[iord[i],pa-1]],
                              [F2V[c,po-1], F1U[iord[i],po-1]],
                              '-b',linewidth=D2V[iord[i]]/moydc);
                    print('\''+Tmodels[iord[i],0],end='\',');
            del F1U_
    #
    if 1 : # Inertie
        inertie, icum = acp.phinertie(VAPT); #print("inertie=:"); tls.tprin(inertie," %6.3f ")
        if NIJ==1 :
            plt.title("%sSST(%s)). %s%d AFC on classes of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah,nb_class));
        elif NIJ==3 :
            plt.title("%sSST(%s)). %s%d AFC on good classes of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah,nb_class));
    #
    if 0 : # Contributions Absolues lignes (déjà calculé)
           # présentation en courbe (+ somme)
        CtrAi = np.concatenate( (CAi, np.sum(CAi,axis=1).reshape(len(CAi),1)),axis=1)
        leg = list((np.arange(nb_class-1)+1).astype(str)); leg.append('sum')
        plt.figure(); plt.plot(CtrAi); plt.legend(leg);
        plt.xticks(np.arange(Nmdlok),Tm_, fontsize=8, rotation=45,
                   horizontalalignment='right', verticalalignment='baseline');
        plt.title("AFC: Contributions Absolues lignes (i.e. modèle) pour chaque axe");
    #
    del Tp_, Tm_
    # Fin AFC
#
#%%%<------------------------------------------------------------------
#%%%<==================================================================
#**********************************************************************
# ........................... GENERALISATION ..........................
def mixtgeneralisation (TMixtMdl) :
    ''' Ici, j'ai : Tmdlok   : une table de NOMS de N modèles valides ;
        remplacé par 
        Ici, j'ai : Tmdlname : une table de NOMS de N modèles valides ; 
                    TDmdl4ct : la table correspondante des modèles 4CT (N, v,12)
        D'autre part, je vais définir d'une manière ou d'une autre les
        modèles devant participer à la définition du modèles moyen.
        Ici, il y a plusieurs possibilités :
        
        1) Une seule ligne de modèles à utiliser sans dicernement de classe
           par exemple :
        TMixtMdl = ['ACCESS1-3','NorESM1-ME','bcc-csm1-1','NorESM1-M','CESM1-BGC']
        
        2) Plusieurs lignes de modèle : une par classe, par exemple
        TMixtMdl = [['ACCESS1-3','NorESM1-ME','bcc-csm1-1','NorESM1-M','CESM1-BGC'],
             ['IPSL-CM5B-LR','ACCESS1-3','MPI-ESM-P','CMCC-CMS','GISS-E2-R-CC'],
             ['bcc-csm1-1-m','MIROC-ESM-CHEM','MIROC-ESM','CSIRO-Mk3-6-0','CanESM2'],
             ['MPI-ESM-MR','CMCC-CM','IPSL-CM5A-MR','FGOALS-g2','MPI-ESM-LR'],
             ['IPSL-CM5A-MR','CNRM-CM5-2','MPI-ESM-MR','MRI-ESM1','MRI-CGCM3'],
             ['FGOALS-s2','CNRM-CM5','CNRM-CM5-2','GFDL-CM3','CMCC-CM'],
             ['GFDL-CM3','GFDL-CM2p1','GFDL-ESM2G','CNRM-CM5','GFDL-ESM2M']];

        Dans les 2 cas, il faut :
        - Prendre les modèles de TMixtMdl à condition qu'ils soient aussi
          dans Tmdlname
        - Envisager ou pas une 2ème phase ... (pas PLM) 
    '''
    # Je commence par le plus simple : Une ligne de modèle sans classe en une phase
    # Je prend le cas : CAH effectuée sur les 6 coordonnées d’une AFC  nij=3 ... 
#    TMixtMdl = ['CMCC-CM',   'MRI-ESM1',    'HadGEM2-AO','MRI-CGCM3',   'HadGEM2-ES',
#                'HadGEM2-CC','FGOALS-g2',   'CMCC-CMS',  'GISS-E2-R-CC','IPSL-CM5B-LR',
#                'GISS-E2-R', 'IPSL-CM5A-LR','FGOALS-s2', 'bcc-csm1-1'];
    #
    # déterminer l'indice des modèles de TMixtMdl dans Tmdlname
    IMixtMdl = [];
    for mname in TMixtMdl :
        im = np.where(Tmdlname == mname)[0]; 
        if len(im) == 1 :
            IMixtMdl.append(im[0])
    #
    if len(IMixtMdl) == 0 :
        print("GENERALISATION IMPOSSIBLE : AUCUN MODELE DISPONIBLE (sur %d)"%(len(TMixtMdl)))
        return
    else :
        print("%d modèles disponibles (sur %d) pour la generalisation : %s"
              %(len(IMixtMdl),len(TMixtMdl),Tmdlname[IMixtMdl]));
    #
    # Modèle moyen
    MdlMoy = Dmdlmoy4CT(TDmdl4CT,IMixtMdl);
    if 1 : # Affichage du moyen for CT
        aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,figsize=(12,9));
        plt.suptitle("MdlMoy %s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
                    %(fcodage,andeb,anfin,np.min(MdlMoy),
                     np.max(MdlMoy),np.mean(MdlMoy),np.std(MdlMoy)))
    #
    # Classification du modèles moyen
    plt.figure();
    Perfglob_ = Dgeoclassif(sMapO,MdlMoy,LObs,CObs,isnumObs);
    ##!!?? plt.title("MdlMoy(%s), perf=%.0f%c"%(Tmdlok[IMixtMdl,0],100*Perfglob_,'%'),fontsize=sztitle); #,fontweigth='bold');
    plt.title("MdlMoy(%s), perf=%.0f%c"%(Tmdlname[IMixtMdl],100*Perfglob_,'%'),fontsize=sztitle); #,fontweigth='bold');
    #tls.klavier();
#%%-----------------------------------------------------------
if 1 :
    # Je commence par le plus simple : Une ligne de modèle sans classe en une phase
    # et une seule codification à la fois
    #
    # Sopt-1975-2005 : Les meilleurs modèles de la période "de référence" 1975-2005
    #
    # ANOMALIES
    if WITHANO and NIJ==1 :
        print("Cas ANOMALIE : CAH effectuée sur les 6 coordonnées d’une AFC  NIJ=1")
        
        TMixtMdl = ['CMCC-CM', 'MRI-ESM1', 'HadGEM2-AO', 'MRI-CGCM3', 'HadGEM2-ES',
                    'HadGEM2-CC', 'FGOALS-g2', 'IPSL-CM5B-LR', 'IPSL-CM5A-LR', 'FGOALS-s2'];
    elif WITHANO and NIJ==3 :
        print("Cas ANOMALIE : CAH effectuée sur les 6 coordonnées d’une AFC  NIJ=3")
        # Meilleur cluster :
        TMixtMdl = ['CMCC-CM',   'MRI-ESM1',    'HadGEM2-AO','MRI-CGCM3',   'HadGEM2-ES',
                    'HadGEM2-CC','FGOALS-g2',   'CMCC-CMS',  'GISS-E2-R-CC','IPSL-CM5B-LR',
                    'GISS-E2-R', 'IPSL-CM5A-LR','FGOALS-s2', 'bcc-csm1-1'];
    elif WITHANO : # Cas sans afc -> n premiers Best Cum
        print("Cas ANOMALIE : n premiers Best Cum")
        if IMaxPerfglob_Qm > 0 :
            print("Cum Best n=%d premiers :"%IMaxPerfglob_Qm, Tmdlname[0:IMaxPerfglob_Qm])
        TMixtMdl = ['CMCC-CM', 'MRI-ESM1', 'HadGEM2-AO', 'MRI-CGCM3', 'HadGEM2-ES',
                    'HadGEM2-CC', 'FGOALS-g2', 'CMCC-CMS'];
    # UISST
    elif UISST and NIJ==1 :
        print("Cas UISST : CAH effectuée sur les 6 coordonnées d’une AFC  NIJ=1")
        # Meilleur cluster :
        TMixtMdl = ['CMCC-CM', 'CMCC-CMS'];
    elif UISST and NIJ==3 :
        print("Cas UISST : CAH effectuée sur les 6 coordonnées d’une AFC  NIJ=3")
        # Meilleur cluster :
        TMixtMdl = ['CMCC-CM', 'HadGEM2-CC', 'CMCC-CMS', 'HadGEM2-ES', 'HadGEM2-AO'];
    elif UISST : # Cas sans afc -> n premiers Best Cum
        print("Cas UISST : n premiers Best Cum")
        if IMaxPerfglob_Qm > 0 :
            print("Cum Best n=%d premiers :"%IMaxPerfglob_Qm, Tmdlname[0:IMaxPerfglob_Qm])
        TMixtMdl = ['CMCC-CM', 'CNRM-CM5'];
    #
    # PENTE
    elif climato=="GRAD" and NIJ==1 :
        print("Cas GRAD : CAH effectuée sur les 6 coordonnées d’une AFC  NIJ=1")
        # Meilleur cluster :
        TMixtMdl = ['CMCC-CMS', 'IPSL-CM5A-LR', 'HadGEM2-CC', 'HadGEM2-ES', 'HadGEM2-AO',
                    'GFDL-CM2p1', 'GFDL-ESM2G', 'GFDL-ESM2M'];
    elif climato=="GRAD" and NIJ==3 :
        print("Cas GRAD : CAH effectuée sur les 6 coordonnées d’une AFC  NIJ=3");
        # Meilleur cluster :
        TMixtMdl = ['bcc-csm1-1', 'CanESM2', 'CMCC-CM', 'IPSL-CM5B-LR', 'FGOALS-s2',
                    'MIROC5','HadGEM2-CC', 'MPI-ESM-MR', 'MPI-ESM-P', 'GISS-E2-R-CC',
                    'NorESM1-ME'];
    elif climato=="GRAD": # Cas sans afc -> n premiers Best Cum
        print("Cas GRAD : n premiers Best Cum")
        if IMaxPerfglob_Qm > 0 :
            print("Cum Best n=%d premiers :"%IMaxPerfglob_Qm, Tmdlname[0:IMaxPerfglob_Qm])
        TMixtMdl = ['IPSL-CM5B-LR', 'GISS-E2-R-CC', 'CMCC-CMS'];
    #
    # (sinon ?) BRUTE 
    elif NIJ==1 :
        print("Cas SST BRUTE : CAH effectuée sur les 6 coordonnées d’une AFC NIJ=1");
        # Meilleur cluster :
        TMixtMdl = ['FGOALS-s2', 'CESM1-BGC', 'CESM1-FASTCHEM', 'CCSM4'];
    elif NIJ==3 :
        print("Cas SST BRUTE : CAH effectuée sur les 6 coordonnées d’une AFC NIJ=3")
        # Meilleur cluster :
        TMixtMdl = ['MPI-ESM-MR','FGOALS-s2', 'MPI-ESM-LR',    'MPI-ESM-P',
                    'FGOALS-g2', 'CESM1-BGC', 'CESM1-FASTCHEM','CCSM4'];
        #TMixtMdl= ['MPI-ESM-P', 'MPI-ESM-LR', 'FGOALS-g2', 'CMCC-CMS', 'MPI-ESM-MR', 
        #           'FGOALS-s2', 'CMCC-CM', 'inmcm4']; was for [0, 1]
    else : # ---
        print("Cas SST BRUTE : Tous les modèles jusqu’à la dernière meilleurs perf en cumulé");
        if IMaxPerfglob_Qm > 0 :
            print("Cum Best n=%d premiers :"%IMaxPerfglob_Qm, Tmdlname[0:IMaxPerfglob_Qm])
        TMixtMdl = ['MPI-ESM-MR','FGOALS-s2', 'MPI-ESM-LR'];
    #
    print("%d modele(s) de generalisation : %s "%(len(TMixtMdl),TMixtMdl))
    #
    mixtgeneralisation (TMixtMdl);
#**********************************************************************
#**********************************************************************
#___________
plt.show();
#___________
print("WITHANO,UISST,climato,NIJ :\n", WITHANO, UISST,climato,NIJ)
import os
print("whole time code %s: %f" %(os.path.basename(sys.argv[0]), time()-tpgm0));

#======================================================================











