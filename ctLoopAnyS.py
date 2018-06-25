# -*- coding: cp1252 -*-
import sys
import os
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
# pomp� de WORKZONE ... TPA05
    if holdon == False :
        # j'ai un pb obs \ pas obs qui apparaissent dans la m�me couleur que le dernier cluster
        # quand bien m�me il ne participe pas � la clusterisation
        lenCP = len(CP); lenXcol = len(Xcol);
        if lenCP > lenXcol : # hyp : lenCP>lenXcol
            # Je consid�re que les (LE) surnum�raire de CP sont les obs (ou aut chose), je l'enl�ve,
            # et le met de cot�
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
        fig = plt.figure(figsize=(16,12));
        fignum = fig.number
        my_norm = plt.Normalize()
        my_normed_data = my_norm(Xcol)
        ec_colors = cmap(my_normed_data) # a Nx4 array of rgba value
        #? if np.ndim(K) > 1 : # On distingue triangle � droite ou vers le haut selon l'axe
        n,p = np.shape(K);
        if p > 1 : # On distingue triangle � droite ou vers le haut selon l'axe 
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
    
    if 0 : # je me rapelle plus tres bien � quoi ca sert; do we need a colorbar here ? may be
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

    # Plot en noir des triangles de r�f�rence en bas � gauche
    if holdon == False :
        dx = xlim[1] - xlim[0];
        dy = ylim[1] - ylim[0];
        px = xlim[0] + dx/(xoomK) + dx/20; # � ajuster +|- en ...
        py = ylim[0] + dy/(xoomK) + dy/20; # ... fonction de xoomK
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK,     facecolor='none');
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK*0.5, facecolor='none');
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK*0.1, facecolor='none');
#
#----------------------------------------------------------------------
def Dmdlmoy4CT (TDmdl4CT,igroup,pond=None) :
    # Mod�le Moyen d'un groupe\cluster des donn�es 4CT
    # Si pond, il doit avoir la meme longueur que TDmdl4CT
    if pond is None : # Non pond�r�
        CmdlMoy = np.mean(TDmdl4CT[igroup],axis=0); # Dmdl moyen d'un cluster
    else : # Mod�le Moyen Pond�r�
        pond       = pond[igroup]; # dans igroup y'a pas l'indice qui correspond aux Obs
        TDmdl4CTi_ = TDmdl4CT[igroup];        # (11,743,12)
        CmdlMoy    = TDmdl4CTi_[0] * pond[0]; # init du modele moyen
        for kk in np.arange(len(pond)-1)+1 :
            CmdlMoy = CmdlMoy + (TDmdl4CTi_[kk] * pond[kk])
        CmdlMoy    = CmdlMoy / np.sum(pond);
    return CmdlMoy; # Cluster mod�le Moyen
#
def Dgeoclassif(sMap,Data,L,C,isnum) :
    bmus_   = ctk.mbmus (sMap,Data);
    classe_ = class_ref[bmus_].reshape(NDmdl);   
    X_Mgeo_ = dto2d(classe_,L,C,isnum); # Classification g�ographique
    #plt.figure(); g�r� par l'appelant car ce peut �tre une fig d�j� d�finie
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
    # Codification des donn�es to CT
    Ndata, Ldata, Cdata = np.shape(data);
    #
    #if INDSC : # Indicateur de Saisonalit� Climatologique
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
    # Transfo Apr�s mise sous forme de pixels Moyens Mensuels
    if NORMMAX == True :
        Maxi = np.max(Ddata, axis=1);
        Ddata = (Ddata.T / Maxi).T;
    if CENTRED :
        Ddata  = tls.centred(Ddata,biais=0); # mais en fait ...
    #----
    return data, Ddata, NDdata;
#
def plot_classes(sMapO,sst_obs,Dobs,NDobs,listofclasses):
    Nobs, Lobs, Cobs = np.shape(sst_obs)
    bmusO  = ctk.mbmus (sMapO, Data=Dobs); # d�j� vu ? conditionnellement ?
    minref = np.min(sMapO.codebook);
    maxref = np.max(sMapO.codebook);
    Z_  = linkage(sMapO.codebook, method_cah, dist_cah);
    #del Z_
    #
    nclasses = len(listofclasses)
    nclassesc = np.ceil(np.sqrt(nclasses));
    nclassesl = np.ceil(1.0*nclasses/nclassesc);

    for iclass in np.arange(nclasses):
        nb_class = listofclasses[iclass]
        class_ref   = fcluster(Z_,nb_class,'maxclust'); # Classes des referents
        coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
        ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        bounds = np.arange(nb_class+1)+1; # pour bounds faut une fronti�re de plus [1, 2, 3, 4, 5, 6, 7]
        sztitle = 10;
        #
        # Transcodage des indices des classes
        if TRANSCOCLASSE is not '' :
            class_ref = transco_class(class_ref,sMapO.codebook,crit=TRANSCOCLASSE);
        #
        classe_Dobs = class_ref[bmusO].reshape(NDobs); #(sMapO.dlen)
        XC_Ogeo     = dto2d(classe_Dobs,Lobs,Cobs,isnumobs); # Classification g�ographique
        
        #>
        # Nombre de pixels par classe (pour les obs)
        Nobsc = np.zeros(nb_class)
        for c in np.arange(nb_class)+1 :
            iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
            Nobsc[c-1] = len(iobsc);
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #plt.figure(); plt.imshow(XC_Ogeo, interpolation='none',vmin=1,vmax=nb_class)
        NclDobs  = len(classe_Dobs)
        fond_C = np.ones(NclDobs)
        fond_C = dto2d(fond_C,Lobs,Cobs,isnumobs,missval=0.5)
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #plt.figure(figsize=(8,6) );
        plt.subplot(nclassesl,nclassesc,iclass+1)
        #
        plt.imshow(XC_Ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
        hcb    = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(coches);
        hcb.ax.tick_params(labelsize=8)
        plt.title("{} classes".format(nb_class),fontsize=14)
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
    plt.suptitle("obs, classe g�og., Method %s [%s]"%(method_cah,case_label),fontsize=18); #,fontweigth='bold');
    #plt.show(); sys.exit(0)

#%% ###################################################################
# INITIALISATION
# Des trucs qui pourront servir
#======================================================================
plt.rcParams.update({'figure.max_open_warning': 0})
#
tpgm0 = time();
plt.ion()
#
#######################################################################
# eface toutes les fenetres de figures en cours
plt.close('all')
#######################################################################
# PARAMETRAGE (#1) DU CAS
from ParamCas import *

varnames = np.array(["JAN","FEV","MAR","AVR","MAI","JUI",
                    "JUI","AOU","SEP","OCT","NOV","DEC"]);

print("case label: {}\n".format(case_label))

if SAVEFIG :
    if not os.path.exists(FIGSDIR) :
        os.makedirs(FIGSDIR)
    case_figs_dir = os.path.join(FIGSDIR,case_label)
    if not os.path.exists(case_figs_dir) :
        os.makedirs(case_figs_dir)

print(os.path.exists("/home/el/myfile.txt"))

#======================================================================
#
#
#%% ####################################################################
# ACQUISITION DES DONNEES D'OBSERVATION (et application des codifications)
#======================================================================
# Lecture des Obs____________________________________
if DATAOBS == "raverage_1975_2005" :
    if 0 : # Ca c'�tait avant
        #sst_obs = np.load("Datas/sst_obs_1854a2005_25L36C.npy")
        #lat: 29.5 � 5.5 ; lon: -44.5 � -9.5
        sst_obs  = np.load("Datas/sst_obs_1854a2005_Y60X315.npy");
    else :
        import netCDF4
        #nc      = netCDF4.Dataset("./Datas/raverage_1975-2005/ersstv3b_1975-2005_extract_LON-315-351_LAT-30-5.nc");
        nc      = netCDF4.Dataset("./Datas/raverage_1975-2005/ersstv5_1975-2005_extract_LON-315-351_LAT-30-5.nc");
        liste_var = nc.variables;       # mois par mois de janvier 1930 � decembre 1960 I guess
        sst_var   = liste_var['sst']    # 1960 - 1930 + 1 = 31 ; 31 * 12 = 372
        sst_obs   = sst_var[:];         # np.shape = (372, 1, 25, 36)
        Nobs,Ncan,Lobs,Cobs = np.shape(sst_obs);
        if 0 : # visu obs
            showimgdata(sst_obs,fr=0,n=Nobs);
            plt.suptitle("Obs raverage 1930 � 1960")
            plt.show(); sys.exit(0)
        sst_obs   = sst_obs.reshape(Nobs,Lobs,Cobs); # np.shape = (144, 25, 36)
        sst_obs   = sst_obs.filled(np.nan);
#
elif DATAOBS == "raverage_1930_1960" :
    import netCDF4
    nc      = netCDF4.Dataset("./Datas/raverage_1930-1960/ersstv3b_1930-1960_extract_LON-315-351_LAT-30-5.nc");
    liste_var = nc.variables;       # mois par mois de janvier 1930 � decembre 1960 I guess
    sst_var   = liste_var['sst']    # 1960 - 1930 + 1 = 31 ; 31 * 12 = 372
    sst_obs   = sst_var[:];         # np.shape = (372, 1, 25, 36)
    Nobs,Ncan,Lobs,Cobs = np.shape(sst_obs);
    if 0 : # visu obs
        showimgdata(sst_obs,fr=0,n=Nobs);
        plt.suptitle("Obs raverage 1930 � 1960")
        plt.show(); sys.exit(0)
    sst_obs   = sst_obs.reshape(Nobs,Lobs,Cobs); # np.shape = (144, 25, 36)
    sst_obs   = sst_obs.filled(np.nan);
#    
elif DATAOBS == "rcp_2006_2017" :
    import netCDF4
    nc      = netCDF4.Dataset("./Datas/rcp_2006-2017/ersst.v3b._2006-2017_extrac-zone_LON-315-351_LAT-30-5.nc");
    liste_var = nc.variables;       # mois par mois de janvier 2006 � decembre 2017 I guess
    sst_var   = liste_var['sst']    # 2017 - 2006 + 1 = 12 ; 12 * 12 = 144
    sst_obs   = sst_var[:];         # np.shape = (144, 1, 25, 36)
    Nobs,Ncan,Lobs,Cobs = np.shape(sst_obs);
    if 0 : # visu obs
        showimgdata(sst_obs,fr=0,n=Nobs);
        plt.suptitle("Obs rcp 2006 � 2017")
        plt.show(); sys.exit(0)
    sst_obs   = sst_obs.reshape(Nobs,Lobs,Cobs); # np.shape = (144, 25, 36)
    sst_obs   = sst_obs.filled(np.nan); 
#    
lat      = np.arange(29.5, 4.5, -1);
lon      = np.arange(-44.5, -8.5, 1);
Nobs,Lobs,Cobs = np.shape(sst_obs); print("obs.shape : ", Nobs,Lobs,Cobs);
#
# Selection___________________________________________
if Nda > 0 : # Ne prendre que les Nda derni�res ann�es (rem ATTENTION, toutes les ne commencent
    sst_obs = sst_obs[Nobs-(Nda*12):Nobs,]; #  pas � l'ann�e 1850 ou 1854 ni au mois 01 !!!!!!!
    if 0 :
        vmin=np.nanmin(sst_obs);    vmax=np.nanmax(sst_obs)
        vmoy=np.nanmean(sst_obs);   vstd=np.nanstd(sst_obs) 
        showimgdata(sst_obs.reshape(372, 1, 25, 36),fr=0,n=4,vmin=vmin, vmax=vmax);
        plt.suptitle("min=%.4f, max=%.4f moy=%.4f, std=%.4f"%(vmin,vmax,vmoy,vstd))
        plt.show(); sys.exit(0)
#
# Param�trage (#2) : _________________________________
print("-- SIZE_REDUCTION == '{}'".format(SIZE_REDUCTION))
if SIZE_REDUCTION == 'sel' or SIZE_REDUCTION == 'RED':
    # D�finir une zone plus petite
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
    # Prendre d'entr�e de jeu une zone plus petite
    sst_obs = sst_obs[:,frl:tol,frc:toc];
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Nobs, Lobs, Cobs = np.shape(sst_obs); print("obs.shape : ", Nobs, Lobs, Cobs);
Npix = Lobs*Cobs; # C'est sens� etre la m�me chose pour tous les mdl
#
# D�finir une fois pour toutes, les indices des nan et non nan pour UNE SEULE
# image (sachant qu'on fait l'hypothese que pour toutes les images, les nans
# sont aux memes endroits). En principe ici les mod�les sont align�s sur les Obs
X_       = sst_obs[0].reshape(Lobs*Cobs);
isnanobs = np.where(np.isnan(X_))[0];
isnumobs = np.where(~np.isnan(X_))[0];
del X_;
#_________________________
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
if 0 : # Visu (et sauvegarde �ventuelle de la figure) des donn�es telles
       # qu'elles vont etre utilis�es par la Carte Topologique
    minDobs = np.min(Dobs);   maxDobs=np.max(Dobs);
    moyDobs = np.mean(Dobs);  stdDobs=np.std(Dobs);
    if climato != "GRAD" :
        aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,figsize=(12,9)); #...
    else :
        aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,wvmin=0.0,wvmax=0.042,figsize=(12,9)); #...
    plt.suptitle("%sSST%d-%d). Obs for CT\nmin=%f, max=%f, moy=%f, std=%f"
                 %(fcodage,andeb,anfin,minDobs,maxDobs,moyDobs,stdDobs));
    if 0 : #SAVEFIG : # sauvegarde de la figure
        fig = plt.gcf() # figure en cours ...
        fignum = fig.number # numero de figure en cours ...
        plt.savefig(case_figs_dir+os.sep+"F{:d}_{}Obs4CT".format(fignum,fshortcode))
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
# Cr�ation de la structure de la carte_______________
norm_method = 'data'; # je n'utilise pas 'var' mais je fais centred �
                      # la place (ou pas) qui est �quivalent, mais qui
                      # me permet de garder la maitrise du codage
sMapO = SOM.SOM('sMapObs', Dobs, mapsize=[nbl, nbc], norm_method=norm_method, \
              initmethod='random', varname=varnames)
print("NDobs(sm.dlen)=%d, dim(Dapp)=%d\nCT : %dx%d=%dunits" \
      %(sMapO.dlen,sMapO.dim,nbl,nbc,sMapO.nnodes));
#
# Apprentissage de la carte _________________________
etape1=[epoch1,radini1,radfin1];    etape2=[epoch2,radini2,radfin2];
#sMapO.train(etape1=etape1,etape2=etape2, verbose='on');
qerr = sMapO.train(etape1=etape1,etape2=etape2, verbose='on', retqerrflg=True);
# + err topo maison
bmus2O = ctk.mbmus (sMapO, Data=None, narg=2);
etO    = ctk.errtopo(sMapO, bmus2O); # dans le cas 'rect' uniquement
#print("Obs, erreur topologique = %.4f" %etO)
print("Obs,\n  case: {}\n  tseed={} ... qerr={:8.6f} ... terr={:.4f}".format(case_label,tseed,qerr,etO))
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
# Other stuffs ______________________________________
bmusO  = ctk.mbmus (sMapO, Data=Dobs); # d�j� vu ? conditionnellement ?
minref = np.min(sMapO.codebook);
maxref = np.max(sMapO.codebook);
Z_          = linkage(sMapO.codebook, method_cah, dist_cah);
class_ref   = fcluster(Z_,nb_class,'maxclust'); # Classes des referents
del Z_
#
coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
bounds = np.arange(nb_class+1)+1; # pour bounds faut une fronti�re de plus [1, 2, 3, 4, 5, 6, 7]
sztitle = 10;
#
# Transcodage des indices des classes
if TRANSCOCLASSE is not '' :
    class_ref = transco_class(class_ref,sMapO.codebook,crit=TRANSCOCLASSE);
#
classe_Dobs = class_ref[bmusO].reshape(NDobs); #(sMapO.dlen)
XC_Ogeo     = dto2d(classe_Dobs,Lobs,Cobs,isnumobs); # Classification g�ographique

#>
# Nombre de pixels par classe (pour les obs)
Nobsc = np.zeros(nb_class)
for c in np.arange(nb_class)+1 :
    iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
    Nobsc[c-1] = len(iobsc);
#
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Pour diff�rencier la zone entiere de la zone REDuite, je conviens que le o
# de obs sera en majuscule pour la zone enti�re (selectionn�e).
# du coup, je duplique.
sst_Obs  = np.copy(sst_obs); NObs=Nobs; LObs=Lobs; CObs=Cobs;
isnumObs = isnumobs; XC_ogeo = XC_Ogeo; classe_DObs = classe_Dobs;
#
if SIZE_REDUCTION == 'RED' :
    #sst_obs, XC_Ogeo, classe_DObs, isnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,frl,tol,frc,toc);
    sst_obs, XC_ogeo, classe_Dobs, isnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,frl,tol,frc,toc);
    # si on ne passe pas ici, les petits o et les grand O sont �gaux
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#plt.figure(); plt.imshow(XC_ogeo, interpolation='none',vmin=1,vmax=nb_class)
Nobs, Lobs, Cobs = np.shape(sst_obs)
NDobs  = len(classe_Dobs)
fond_C = np.ones(NDobs)
fond_C = dto2d(fond_C,Lobs,Cobs,isnumobs,missval=0.5)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#%% ###################################################################
#          Visualisations apres entrainement de la carte SOM
#======================================================================
if True :
    # Figure par clases de la zone geographique, pour plussieurs valeurs de nombre de classes
    list_of_classes_to_show = [4,5,6,7,8,9]
    fig = plt.figure(figsize=(12,7.5));
    fignum = fig.number
    plot_classes(sMapO,sst_obs,Dobs,NDobs,list_of_classes_to_show)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if SAVEFIG :
        #plt.savefig(case_figs_dir+os.sep+"%s%s%dMdlvsObstrans"%(fshortcode,method_cah,nb_class))
        plt.savefig(case_figs_dir+os.sep+"F{:d}_{}{}_{}_ObsShow-{:d}-{:d}-Classes".format(fignum,
                    fprefixe,SIZE_REDUCTION,fshortcode,
                    list_of_classes_to_show[0],list_of_classes_to_show[-1]))

if 1 : # for Obs
    # Figure par clases de la zone geographique, uniquement pour le nombre de classes choisie
    fig = plt.figure(figsize=(8,6));
    fignum = fig.number
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb    = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs {:d} Classes, classe g�og., Method {}\n[{}]".format(nb_class,
        method_cah,case_label),fontsize=16); #,fontweigth='bold');
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
    if SAVEFIG :
        #plt.savefig(case_figs_dir+os.sep+"%s%s%dMdlvsObstrans"%(fshortcode,method_cah,nb_class))
        plt.savefig(case_figs_dir+os.sep+"F{:d}_{}{}_{}_Obs-{:d}-Classes".format(fignum,
                    fprefixe,SIZE_REDUCTION,fshortcode,nb_class))
#
if 0 : # for obs
    # Figure du spectre moyen par clases
    fig = plt.figure(figsize=(12,6) );
    fignum = fig.number
    TmoymensclassObs = moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class)
    #plt.plot(TmoymensclassObs); plt.axis('tight');
    for i in np.arange(nb_class) :
        plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
    plt.axis('tight');
    plt.xlabel('mois');
    plt.legend(np.arange(nb_class)+1,loc=2,fontsize=8);
#    plt.title("Obs {} Classes, Moy. Mens. par Classe Method {} [{}]".format(nb_class,method_cah,case_label),fontsize=16);
    #plt.show(); sys.exit(0)
    if SAVEFIG :
        #plt.savefig(case_figs_dir+os.sep+"%s%s%dMdlvsObstrans"%(fshortcode,method_cah,nb_class))
        plt.savefig(case_figs_dir+os.sep+"F{:d}_{}{}_{}_ObsMoyMensByClass-{:d}-Classes".format(fignum,
                    fprefixe,SIZE_REDUCTION,fshortcode,nb_class))
#
if 0 :
    # Figure du spectre moyen par referent
    #fig = plt.figure(figsize=(6,10));
    ctk.showprofils(sMapO, figsize=(6,10), Data=Dobs,visu=3, scale=2,
                    Clevel=class_ref-1,Gscale=0.5,
                    ColorClass=pcmap,showcellid=False);
    fig = plt.gcf() # figure en cours ...
    fignum = fig.number # numero de figure en cours ...
    plt.suptitle("Obs {} Classes, Moy. Mens. par Classe Method {}\n[{}]".format(nb_class,
            method_cah,case_label),x=0.5,y=0.99,fontsize=14);
    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.show(); sys.exit(0)
    if SAVEFIG :
        #plt.savefig(case_figs_dir+os.sep+"%s%s%dMdlvsObstrans"%(fshortcode,method_cah,nb_class))
        plt.savefig(case_figs_dir+os.sep+"F{:d}_{}{}_{}_ObsMoyMensByClassByRef-{:d}-Classes".format(fignum,
                    fprefixe,SIZE_REDUCTION,fshortcode,nb_class))
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
nsub   = 49; # actuellement au plus 48 mod�les + 1 pour les obs.      
#nsub  = 9;  # Pour Michel (8+1pour les obs)     
nbsubc = np.ceil(np.sqrt(nsub));
nbsubl = np.ceil(1.0*nsub/nbsubc);
isubplot=0;
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
min_moymensclass = 999999.999999; # sert pour avoir tous les ...
max_moymensclass = 000000.000000; # ... subplots � la m�me �chelles
TypePerf         = ["AccuracyPrecision"]; #,"Index2Rand"];
Tperfglob        = np.zeros((Nmodels,len(TypePerf))); # Tableau des Perf globales des mod�les
if NIJ==1 :
    TNIJ         = [];  
TTperf           = [];  
#
TDmdl4CT         = []; # Stockage des mod�les 4CT pour AFC-CAH ...  
#
Tmdlok           = []; # Pour construire une table des mod�les valides
Nmdlok           = 0;  # Pour si y'a cumul ainsi connaitre l'indice de mod�le valide 
                       # courant, puis au final, le nombre de mod�les valides
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
    # Lecture des donn�es
    if DATAMDL=="raverage_1975_2005" : # fichiers.mat g�n�r�s par Carlos
        datalib = 'Datas/raverage_1975-2005/sst_'
        try :
            sst_mat = scipy.io.loadmat(datalib+mdlname+"_raverage_1975-2005.mat");
        except :
            continue;
        sst_mdl = sst_mat['SST'];       
    elif DATAMDL=="raverage_1930_1960" : # fichiers.mat g�n�r�s par Carlos
        datalib = 'Datas/raverage_1930-1960/sst_'
        try :
            sst_mat = scipy.io.loadmat(datalib+mdlname+"_raverage_1930-1960.mat");
        except :
            continue;
        sst_mdl = sst_mat['SST'];      
    elif DATAMDL == "rcp_2006_2017": # fichiers.mat sc�narios g�n�r�s par Carlos.
        # d�di�s � l'�tude de la g�n�ralisation
        datalib = "Datas/rcp_2006-2017/%s/sst_"%scenar
        try :
            sst_mat = scipy.io.loadmat(datalib+mdlname+"_raverage_2006-2017.mat");
        except :
            continue;
        sst_mdl = sst_mat['SST'];
    Nmdl, Lmdl, Cmdl = np.shape(sst_mdl); # print("mdl.shape : ", Nmdl, Lmdl, Cmdl);
    #
    Nmdlok = Nmdlok + 1; # Pour si y'a cumul ainsi connaitre l'indice de mod�le 
             # valide courant, puis au final, le nombre de mod�les valides.
             # (mais ...)
    # L� je construis une table des mod�les valides
    Tmdlok.append(Tmodels[imodel]);
    #
    if MDLCOMPLETION : # Compl�mentation des donn�es mod�les de sorte � ce que seul
        nnan=1;        # le mappage des nans d'obs soit utilis�
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
        # Dans le cas de la compl�tion des mod�les, ce sont les nans des obs qui s'appliquent
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
    # Codification du mod�le (4CT)            
    sst_mdl, Dmdl, NDmdl = datacodification4CT(sst_mdl);
    #________________________________________________________
    if 0 : # Visu du mod�le (moyen) #bok
        minDmdl = np.min(Dmdl);   maxDmdl=np.max(Dmdl);
        moyDmdl = np.mean(Dmdl);  stdDmdl=np.std(Dmdl);
        aff2D(Dmdl,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,figsize=(12,9));
        plt.suptitle("%s %s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
                     %(mdlname,fcodage,andeb,anfin,minDmdl,maxDmdl,moyDmdl,stdDmdl));
        #continue; #plt.show(); sys.exit(0);
        #X_ = np.mean(Dmdl, axis=1); X_=X_.reshape(743,1); #rem = 0.0 when anomalie
        #aff2D(X_,Lobs,Cobs,isnumobs,isnanobs,wvmin=-3.8?,wvmax=4.9?,figsize=(12,9));
    #________________________________________________________
    TDmdl4CT.append(Dmdl);  # stockage des mod�les 4CT pour AFC-CAH ...
    Tmdlname.append(Tmodels[imodel,0])
    # Calcul de la perf glob du mod�le et stockage pour tri
    bmusM       = ctk.mbmus (sMapO, Data=Dmdl);
    classe_DMdl = class_ref[bmusM].reshape(NDmdl);
    perfglob    = len(np.where(classe_DMdl==classe_Dobs)[0])/NDobs
    Tperfglob4Sort.append(perfglob)
    Tclasse_DMdl.append(classe_DMdl)
    #
    if OK106 : # Stockage (if required) pour la Courbes des moyennes mensuelles par classe
        Tmoymensclass.append(moymensclass(sst_mdl,isnumobs,classe_Dobs,nb_class));
#
# Fin de la PREMIERE boucle sur les mod�les
#
#%%_____________________________________________________________________
######################################################################
# Reprise de la boucle (avec les mod�les valides du coup).
# (question l'emplacement des mod�les sur les figures ne devrait pas etre un probl�me ?)
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
MaxPerfglob_Qm  = 0.0; # Utilis� pour savoir les quels premiers mod�les
IMaxPerfglob_Qm = 0;   # prendre dans la strat�gie du "meilleur cumul moyen"
#*****************************************
# l'Init des figures � produire doit pouvoir etre plac� ici ##!!?? (sauf la 106)
if OK104 : # Classification avec, "en transparance", les mals class�s
           # par rapport aux obs
    fig104 = plt.figure(104,figsize=(18,9),facecolor='w')
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle104="%sSST(%s)). [%s] - %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,case_label,method_cah);
if OK105 : #Classification
    fig105 = plt.figure(105,figsize=(18,9))
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle105="%sSST(%s)). [%s] - %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,case_label,method_cah);
if OK106 : # Courbes des moyennes mensuelles par classe
    fig106 = plt.figure(106,figsize=(18,9),facecolor='w'); # Moyennes mensuelles par classe
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle106="MoyMensClass(%sSST(%s)).) [%s] - %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,case_label,method_cah);
if OK107 : # Variance (not 'RED' compatible)
    fig107 = plt.figure(107,figsize=(18,9),facecolor='w'); # Moyennes mensuelles par classe
    suptitle107="VARiance(%sSST(%s)).) [%s] - Variance (by pixel) on Completed Models" \
                 %(fcodage,DATAMDL,case_label);
    Dmdl_TVar  = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance par pixel sur climatologie
                   # J'utiliserais ainsi showimgdata pour avoir une colorbar commune
#
if MCUM  : # # Moyenne des Models climatologiques CUmul�s
    DMdl_Q  = np.zeros((NDobs,12));  # f(mod�le climatologique Cumul�)
    DMdl_Qm = np.zeros((NDobs,12));  # f(mod�le climatologique Cumul� moyen)
#
# Moyenne CUMulative
if OK108 : # Classification en Model Cumul� Moyen
    fig108 = plt.figure(108,figsize=(18,9),facecolor='w'); # Moyennes mensuelles par classe
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle108="MCUM - %sSST(%s)) [%s] - %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,case_label,method_cah);
#
# Variance CUMulative
if OK109 : # Variance sur les Models Cumul�s Moyens (not 'RED' compatible)
    fig109 = plt.figure(109,figsize=(18,9),facecolor='w'); # Moyennes mensuelles par classe
    Dmdl_TVm = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance sur climatologie
               # cumul�e, moyenn�e par pixel. J'utiliserais ainsi showimgdata pour avoir
               # une colorbar commune
    suptitle109="VCUM - %sSST(%s)) [%s] -  Variance sur la Moyenne Cumul�e de Modeles compl�t�s" \
                 %(fcodage,DATAMDL,case_label);
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
    XC_Mgeo     = dto2d(classe_DMdl,LObs,CObs,isnumObs); # Classification g�ographique
    #
    #>>>>>>>>>>> (� revoir)
    classe_Dmdl = classe_DMdl; # ... because RED ... du coup je duplique            pour avoir les memes
    XC_mgeo     = XC_Mgeo;     # ... because RED ... du coup je duplique            noms de variables.
    if 0 : # SIZE_REDUCTION == 'RED' :
        #sst_Mdl  = np.copy(sst_mdl); NObs=Nmdl; LObs=Lmdl; CObs=Cmdl;
        #sst_mdl, XC_mgeo, classe_Dmdl, isnum_red = red_classgeo(sst_Mdl,isnumObs,classe_DMdl,frl,tol,frc,toc);
        sst_mdl, XC_mgeo, classe_Dmdl, isnum_red = red_classgeo(sst_mdl,isnumObs,classe_DMdl,frl,tol,frc,toc);
    #<<<<<<<<<<<
    #
    classe_DD, Tperf, Perfglob = perfbyclass(classe_Dobs,classe_Dmdl,nb_class);
    #else : # Perf Indice de Rand (does not work yet) voir version pr�c�dante
    #
    Tperf = np.round([i*100 for i in Tperf]).astype(int); #print(Tperf)
    TTperf.append(Tperf);           # !!!rem AFC est faite avec ca    
    Tperfglob[imodel,0] = Perfglob;
    #
    Nmdl, Lmdl, Cmdl = np.shape(sst_mdl)    ##!!??
    NDmdl = len(classe_Dmdl); # needed for 108, ...? 
    #
    #%> l'AFC pourrait aussi etre faite avec �a
    if NIJ == 1 : # Nij = card|classes| ; Pourrait sembl� inappropri� car les classes
                  # peuvent g�ographiquement �tre n'importe o�, mais ...               
        Znij_ = [];  
        for c in np.arange(nb_class) :
            imdlc = np.where(classe_Dmdl==c+1)[0]; # Indices des classes c du model
            Znij_.append(len(imdlc));  
        TNIJ.append(Znij_);  
    #%<
    #:::>------------------------------------------------------------------
    '''FAUDRA VOIR QUEL INDICE ON MET SUR CHAQUE MODEL,
    ET SI IL FAUDRA FAIRE LA COLORBAR PAR CLASSE SELON CET INDICE !!!??? si c'est possible !!!???
    pour le moment faut faire attention car ce n'est pas maitris� !!!!!!
    '''
    ip = 0; 
    if "Index2Rand" in TypePerf :
        ip = ip+1;
        Tperfglob[imodel,ip] = adjusted_rand_score(classe_Dobs, classe_Dmdl);
    #<:::------------------------------------------------------------------
    #
    if 0 : # print ponctuels de mise au point ?
        print("Mod�le: %s ; Method: %s"%(mdlname, method_cah));
        print("Perf toutes classes = %.4f"%(Perfglob))
        print("Perf par classe : ");
        tls.tprin(np.arange(nb_class)+1, "  %d ");
        tls.tprin(Tperf, " %3d");
    #
    if OK104 : # Classification avec, "en transparance", les pixels mals
               # class�s par rapport aux obs. (pour les mod�les les Perf
               # par classe sont en colorbar)
        plt.figure(num=fig104.number); plt.subplot(nbsubl,nbsubc,isubplot);
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
    if OK105 : # Classification (pour les mod�les les Perf par classe sont en colorbar)
        plt.figure(num=fig105.number); plt.subplot(nbsubl,nbsubc,isubplot);
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
        plt.figure(num=fig106.number); plt.subplot(nbsubl,nbsubc,isubplot);
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
    if OK108 : # Classification en Model Cumul� Moyen (Perf par classe en colorbar)
        plt.figure(num=fig108.number); plt.subplot(nbsubl,nbsubc,isubplot);
        bmusMdl_Qm = ctk.mbmus (sMapO, Data=DMdl_Qm);
        classe_DMdl_Qm= class_ref[bmusMdl_Qm].reshape(NDmdl);
                       # Ici classe_D* correspond � un r�sultats de classification
                       # (bon ou mauvais ; donc sans nan)
        XC_mgeo_Qm    = dto2d(classe_DMdl_Qm,Lobs,Cobs,isnumobs); # Classification g�ographique
                       # Mise sous forme 2D de classe_D*, en mettant nan pour les
                       # pixels mas class�s
        classe_DD_Qm, Tperf_Qm, Perfglob_Qm = perfbyclass(classe_Dobs, classe_DMdl_Qm, nb_class);
                       # Ici pour classe_DD* : les pixels bien class�s sont valoris�s avec
                       # leur classe, et les mals class�s ont nan
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
            MaxPerfglob_Qm  = pgqm_;     # Utilis� pour savoir les quels premiers mod�les
            IMaxPerfglob_Qm = imodel+1;  # prendre dans la strat�gie du "meilleur cumul moyen"
            print("New best cumul perf for %dmodels : %d%c"%(imodel+1,pgqm_,'%'))
     #
    if OK109 : # Variance sur les Models Cumul�s Moyens (not 'RED' compatible)
                          # Perf par classe en colorbar)
        Dmdl_TVm[imodel] = np.var(DMdl_Qm, axis=1, ddof=0);
#
# Fin de la DEUXIEME boucle sur les mod�les
#%%__________________________________________
# Les Obs � la fin 
isubplot = 49; 
#isubplot = isubplot + 1; # Michel (ou pas ?)
if OK104 : # Obs for 104
    fignum = fig104.number
    plt.figure(num=fignum); plt.subplot(nbsubl,nbsubc,isubplot);
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
        #plt.savefig(case_figs_dir+os.sep+"%s%s%dMdlvsObstrans"%(fshortcode,method_cah,nb_class))
        plt.savefig(case_figs_dir+os.sep+"F%d_%s%s_%s%s%dMdlvsObstrans"%(fignum,fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
if OK105 : # Obs for 105
    fignum = fig105.number
    plt.figure(num=fignum); plt.subplot(nbsubl,nbsubc,isubplot);
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
        #plt.savefig(case_figs_dir+os.sep+"%s%s%dMdl"%(fshortcode,method_cah,nb_class))
        plt.savefig(case_figs_dir+os.sep+"F%d_%s%s_%s%s%dMdl"%(fignum,fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
if OK106 : # Obs for 106
    fignum = fig106.number
    plt.figure(num=fignum); plt.subplot(nbsubl,nbsubc,isubplot);
    TmoymensclassObs = moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class); 
    for i in np.arange(nb_class) :
        plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
    plt.axis([0, 11, min_moymensclass, max_moymensclass]);
    plt.xlabel('mois');
    plt.xticks(np.arange(12), np.arange(12)+1, fontsize=8)
    plt.legend(np.arange(nb_class)+1,loc=2,fontsize=6,numpoints=1,bbox_to_anchor=(1.1, 1.0));
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    #
    # On repasse sur tous les supblots pour les mettre � la m�me echelle.
    print(min_moymensclass, max_moymensclass);
    plt.suptitle(suptitle106)
    if SAVEFIG :
        #plt.savefig(case_figs_dir+os.sep+"%s%s%dmoymensclass"%(fshortcode,method_cah,nb_class))
        plt.savefig(case_figs_dir+os.sep+"F%d_%s%s_%s%s%dmoymensclass"%(fignum,fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
#
if OK107 or OK109 : # Calcul de la variance des obs par pixel de la climatologie
    varobs= np.ones(Lobs*Cobs)*np.nan;          # Variances des ...
    varobs[isnumobs] = np.var(Dobs, axis=1, ddof=0); # ... Obs par pixel
#
if OK107 : # Variance par pixels des mod�les
    fignum = fig107.number
    plt.figure(num=fignum);
    # donnees
    X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
    X_[:,isnumobs] = Dmdl_TVar
    # Labels
    Tlabs = np.copy(Tmdlname);   
    nmodplus=Nmodels
    if Nmodels < 49 :
        for iblank in np.arange(Nmodels,49 - 1) :
            nmodplus += 1
            # Rajouter nan pour le subplot vide
            X_ = np.concatenate((X_, np.ones((1,Lobs*Cobs))*np.nan))
            Tlabs = np.append(Tlabs,'');                # Pour le subplot vide
    nmodplus += 1
    # Rajout de la variance des obs
    X_ = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
    Tlabs = np.append(Tlabs,'Observations');    # Pour les Obs
    #
    showimgdata(X_.reshape(nmodplus,1,Lobs,Cobs), Labels=Tlabs, n=nmodplus,fr=0,
                vmin=np.nanmin(Dmdl_TVar),vmax=np.nanmax(Dmdl_TVar),
                wspace=0.00, hspace=0.14, top=0.93, bottom=0.05, left=0.00, right=1.00,
                y=0.94, # position relative y des titres
                sztext=sztitle,cbpos='vertical',fignum=fignum);
    del X_
    plt.suptitle(suptitle107);
    if SAVEFIG :
        plt.savefig(case_figs_dir+os.sep+"F%d_%sVAR_%s_%sMdl"%(fignum,fprefixe,SIZE_REDUCTION,fshortcode))
#
if OK108 : # idem OK105, but ...
    fignum = fig108.number
    plt.figure(num=fignum); plt.subplot(nbsubl,nbsubc,isubplot);
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
        plt.savefig(case_figs_dir+os.sep+"F%d_%sMCUM_%s_%s%s%dMdl"%(fignum,fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
#
if OK109 : # Variance par pixels des moyenne des mod�les cumul�s
    fignum = fig109.number
    plt.figure(num=fignum);
    # donnees
    X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
    X_[:,isnumobs] = Dmdl_TVm
    # Labels
    Tlabs = np.copy(Tmdlname);   
    nmodplus=Nmodels
    if Nmodels < 49 :
        for iblank in np.arange(Nmodels,49 - 1) :
            nmodplus += 1
            # Rajouter nan pour le subplot vide
            X_ = np.concatenate((X_, np.ones((1,Lobs*Cobs))*np.nan))
            Tlabs = np.append(Tlabs,'');                # Pour le subplot vide
    nmodplus += 1
    # Rajout de la variance des obs
    X_ = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
    Tlabs = np.append(Tlabs,'Observations');    # Pour les Obs
    #
    showimgdata(X_.reshape(nmodplus,1,Lobs,Cobs), Labels=Tlabs, n=nmodplus,fr=0,
                vmin=np.nanmin(Dmdl_TVm),vmax=np.nanmax(Dmdl_TVm),
                wspace=0.00, hspace=0.14, top=0.93, bottom=0.05, left=0.00, right=1.00,
                y=0.94, # position relative y des titres
                sztext=sztitle,cbpos='vertical',fignum=fignum);
    del X_
    plt.suptitle(suptitle109);
    if SAVEFIG :
        plt.savefig(case_figs_dir+os.sep+"F%d_%sVCUM_%s_%sMdl"%(fignum,fprefixe,SIZE_REDUCTION,fshortcode))
#
#%% ---------------------------------------------------------------------
# Redimensionnement de Tperfglob au nombre de mod�les effectif
Tperfglob = Tperfglob[0:Nmodels]; 
#
# -----------------------------------------------------------------------
fig200=plt.figure(200,figsize=(2,1),facecolor='r');
# figure bidon pour forcer un numero de figure supperieur a 200 pour les prochaines figures
# -----------------------------------------------------------------------
# Edition des r�sultats
if 0 : # (print) Tableau des performances
    print("% de Perf global");
    Tperfglob = np.round(Tperfglob,2)*100
    Tperfglob = Tperfglob.astype(int);
    for i in np.arange(Nmodels) :
        print(Tperfglob[i])
#:::>
if 1 : # Tableau des performances en figure de courbes
    fig = plt.figure(figsize=(12,6),facecolor='w'); plt.plot(Tperfglob,'.-');
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.15, left=0.05, right=0.98)
    fignum = fig.number
    plt.axis("tight"); plt.grid('on')
    plt.xticks(np.arange(Nmodels),Tmdlname, fontsize=8, rotation=45,
               horizontalalignment='right', verticalalignment='baseline');
    #          horizontalalignment='right', verticalalignment='center');
    plt.legend(TypePerf,numpoints=1,loc=3)
    plt.title("%sSST(%s)) %s%d Indice(s) de classification of Completed Models (vs Obs)\n[%s]"\
                 %(fcodage,DATAMDL,method_cah,nb_class,case_label));
    if SAVEFIG :
        plt.savefig(case_figs_dir+os.sep+"F%d_%s_%s_%sIndexClassCompMod"%(fignum,fprefixe,SIZE_REDUCTION,fshortcode))
#<:::
#
#
#%%%>==================================================================
# Mettre les Tableaux-Liste en tableau Numpy
#----------------------------------------------------------------------
Tmdlname = np.array(Tmdlname);
TTperf   = np.array(TTperf);
if NIJ==1 :
    TNIJ = np.array(TNIJ);
TDmdl4CT = np.array(TDmdl4CT);
#
if NIJ > 0 : # A.F.C
    #Ajout de l'indice dans le nom du mod�le
    Tm_ = np.empty(len(Tmdlname),dtype='<U32');
    for i in np.arange(Nmdlok) : #a 
        Tm_[i] = str(i+1) + '-' +Tmdlname[i];
    #
    # Harmonaiser la variable servant de tableau de contingence (->Tp_), selon le cas
    if NIJ==1 : # Nij = card|classes| ; Pourrait sembl� inappropri� car les classes
                # peuvent g�ographiquement �tre n'importe o�, mais ...
        Tp_ = TNIJ; # TNIJ dans ce cas a �t� pr�par� en amont
        #
    elif NIJ==2 or NIJ==3 :
        Tp_ = TTperf; # Pourcentages des biens class�s par classe (transform� ci apr�s
                      # pour NIJ==3 en effectif apr�s �ventuel ajout des obs dans l'afc)
    # On supprime les lignes dont la somme est 0 car l'afc n'aime pas ca.
    # (en esp�rant que la suite du code reste coh�rente !!!???)
    som_ = np.sum(Tp_, axis=1);
    Iok_ = np.where(som_>0)[0]
    Tp_  = Tp_[Iok_];
    Tm_  = Tm_[Iok_];
    Nmdlok = len(Iok_); # !!!Attention!!! red�finition du nombre de mod�les valides
    del som_, Iok_;
    #
    if AFCWITHOBS : # On ajoute Obs (si required)
        if NIJ==1 :
            Tp_ = np.concatenate((Tp_, Nobsc[np.newaxis,:]), axis=0).astype(int); 
        else :
            # Obs have 100% for any class par d�finition
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
    else : # Les obs en suppl�mentaire
        VAPT, F1U, CAi, CRi, F2V, CAj, F1sU = afaco(Tp_, Xs=[Nobsc]);
        XoU = F1sU; # coord des Obs (ok j'aurais pu mettre directement en retour de fonction...)
    #
    #-----------------------------------------
    # MODELE MOYEN (pond�r� ou pas) PAR CLUSTER D'UNE CAH
    if 1 : # CAH on afc Models's coordinates (without obs !!!???)
        metho_ = 'ward'; 
        dist_  = 'euclidean';
        coord2take = np.arange(NBCOORDAFC4CAH)
        if AFCWITHOBS :
            # (Mais ne pas prendre les obs dans la CAH (ne prendre que les mod�les))
            Z_ = linkage(F1U[0:Nmdlok,coord2take], metho_, dist_);
            Z_ = linkage(F1U, metho_, dist_);
        else :
            Z_ = linkage(F1U[:,coord2take], metho_, dist_);
        if 1 : # dendrogramme
            fig = plt.figure(figsize=(18,11),facecolor='w');
            plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.15, left=0.05, right=0.98)
            fignum = fig.number
            fignum = fig.number
            if 0:
                R_ = dendrogram(Z_,Nmdlok,'lastp',orientation='top');
                L_ = np.array(lignames)
                plt.xticks((np.arange(len(Tmdlname))*10)+7,L_[R_['leaves']], fontsize=11,
                       rotation=45,horizontalalignment='right', verticalalignment='baseline')
            else:
                R_ = dendrogram(Z_,Nmdlok,'lastp',orientation='left');
                L_ = np.array(lignames)
                plt.yticks((np.arange(len(Tmdlname))*10)+7,L_[R_['leaves']], fontsize=11,
                       rotation=45,horizontalalignment='right', verticalalignment='baseline')

            del R_, L_
            plt.title("AFC: Coord(%s) dendrogram [%s]\nM�tho=%s, dist=%s, nb_clust=%d"
                      %((coord2take+1).astype(str),case_label,metho_,dist_,nb_clust), fontsize=18)
        #
        class_afc = fcluster(Z_,nb_clust,'maxclust');
        #
        figclustmoy = plt.figure(figsize=(16,12));
        figclustmoynum = figclustmoy.number
        nclustcol = np.round(np.sqrt(nb_clust)).astype(int)
        nclustlin = np.ceil(7/nclustcol).astype(int)
        for ii in np.arange(nb_clust) :
            iclust  = np.where(class_afc==ii+1)[0];
            #
            # Visu des Classif des mod�les des cluster
            if  ii+1 in AFC_Visu_Classif_Mdl_Clust :
                print("par ici {}".format(ii))
                fig = plt.figure(figsize=(16,12));
                fignum = fig.number
                for jj in np.arange(len(iclust)) :
                    bmusj_   = ctk.mbmus (sMapO, Data=TDmdl4CT[iclust[jj]]);
                    classej_ = class_ref[bmusj_].reshape(NDmdl);
                    XCM_     = dto2d(classej_,LObs,CObs,isnumObs); # Classification g�ographique
                    plt.subplot(7,7,jj+1);
                    plt.imshow(XCM_, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
                    plt.axis('off');
                    plt.title(Tm_[iclust[jj]],fontsize=sztitle)
                plt.suptitle("Classification des mod�les du cluster %d"%(ii+1));
            if 1 :    
                print("%d Mod�les du cluster %d :\n"%(len(iclust),ii+1), Tmdlname[iclust]); ##!!??
            #
            # Mod�le Moyen d'un cluster
            if 1 : # Non pond�r�
                CmdlMoy  = Dmdlmoy4CT(TDmdl4CT,iclust,pond=None);
            elif 0 : # Pond�r�                
                if 1 : # par les contributions relatives ligne (i.e. mod�les) (CRi) du plan
                    pond = np.sum(CRi[:,[pa-1,po-1]],axis=1);
                elif 0 : # par les contributions absolues ligne (i.e. mod�les) (CAi) du plan
                    pond = np.sum(CAi[:,[pa-1,po-1]],axis=1) 
                elif 0 : # Proportionnelle � la perf global de chaque mod�le ?
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
            # Classification du mod�les moyen d'un cluster
#           plt.figure(figclustmoy.number); plt.subplot(3,3,ii+1);
            plt.figure(figclustmoy.number,figsize=(16,12))
            plt.subplot(nclustcol,nclustlin,ii+1);
            Perfglob_ = Dgeoclassif(sMapO,CmdlMoy,LObs,CObs,isnumObs);
            plt.title("cluster %d, perf=%.0f%c"%(ii+1,100*Perfglob_,'%'),fontsize=sztitle); #,fontweigth='bold');
        plt.suptitle("AFC Clusters - [{}]".format(case_label))
        # FIN de la boucle sur le nombre de cluster
        del metho_, dist_, Z_
    # FIN du if 1 : MODELE MOYEN (pond�r� ou pas) PAR CLUSTER D'UNE CAH
    #-----------------------------------------
    #      
    if 1 : # Calcul de distance (d^2) des coordonn�es des mod�les � celles des Obs
        #!!! Attention : cette partie de code semble ne pas avoir �t� adapt�e aux �volutions !!!
        D2 = (F1U - XoU)**2; # Avec les coordonn�es du plan, triangle d�doubl�
        if 0 : # Avec les coordonn�es du plan somm� : forme carr�
            D2 = np.sum(D2[:,[pa-1,po-1]], axis=1)
        elif 1 : # Avec toutes les coordonn�es
            D2   = np.sum(D2,axis=1); # forme carr�e
            #iord = np.argsort(D2);
            iord = np.argsort(D2[0:Nmdlok-1]); # pour ne pas prendre les Obs elles-m�mes (ou BMU?) #a
            if 0 : # triangles d�doubl�s
                D2 = np.matlib.repmat(D2,nb_class-1,1);
                D2 = D2.T
                iord = np.argsort(D2[:,0]);
            N2take = 10;
            print("\nd2u :");
            for i in np.arange(N2take) :
                print('\''+Tmdlname[iord[i]],end='\','); #?
            print();
    #
    # Nuage de l'AFC
    #K=D2; xoomK=500;    # Pour les distances au carr� (D2) des mod�les aux Obs
    K=CRi; xoomK=1000; # Pour les contrib Rel (CRi)
    if 1 : # ori afc avec tous les points
        afcnuage(F1U,cpa=pa,cpb=po,Xcol=class_afc,K=K,xoomK=xoomK,linewidths=2,indname=lignames); #,cmap=cm.jet);
        if NIJ==1 :
            plt.title("%sSST(%s)) [%s]\n%s%d AFC on classes of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,case_label,method_cah,nb_class),fontsize=16);
        elif NIJ==3 :
            plt.title("%sSST(%s)) [%s]\n%s%d AFC on good classes of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,case_label,method_cah,nb_class),fontsize=16);
    # Limiter les points aux contributions les plus fortes
    def afclim(K,xoomK) :
        if 0 : # Sur les axes consid�r�s
            lim_ = 0.028; # 0.03
            Ia_ = np.where(CAi[:,pa-1]>=lim_)[0];
            Io_ = np.where(CAi[:,po-1]>=lim_)[0];
            Ix_ = np.union1d(Ia_,Io_); 
            afcnuage(F1U[Ix_,:],cpa=pa,cpb=po,Xcol=np.arange(len(Ix_)),K=K[Ix_],xoomK=xoomK,linewidths=2,indname=lignames[Ix_]);
            #plt.title("%s%s(SST%d-%d)). %s%d AFC on good classes of Completed Models (vs Obs) (CtrA lim = %.3f)" \
            #     %(fcodage,DATARUN,andeb,anfin,method_cah,nb_class, lim_));
            plt.title("[%s] %sSST(%s)) [%s]\n%s%d AFC on good classes of Completed Models (vs Obs) (CtrA lim = %.3f)" \
                 %(fcodage,DATAMDL,case_label,method_cah,nb_class, lim_));
            del lim_, Ia_, Io_, Ix_
        if 1 : # Sur tous les axes
            lim_ = 0.13;
            X_   = np.sum(CAi, axis=1)
            Ix_  = np.where(X_>=lim_)[0];
            afcnuage(F1U[Ix_,:],cpa=pa,cpb=po,Xcol=np.arange(len(Ix_)),K=K[Ix_],xoomK=xoomK,linewidths=2,indname=lignames[Ix_]);
            #plt.title("%s%s(SST%d-%d)). %s%d AFC on good classes of Completed Models (vs Obs) (CtrA lim = %.3f)" \
            #     %(fcodage,DATARUN,andeb,anfin,method_cah,nb_class, lim_));
            plt.title("%sSST(%s)) [%s]\n%s%d AFC on good classes of Completed Models (vs Obs) (CtrA lim = %.3f)" \
                 %(fcodage,DATAMDL,case_label,method_cah,nb_class, lim_));
            del lim_, X_, Ix_
    if 0 :
        afclim(K,xoomK);
    #
    if AFCWITHOBS  : # Mettre en �vidence Obs
        plt.plot(F1U[Nmdlok,pa-1],F1U[Nmdlok,po-1], 'oc', markersize=20,        # marker for obs
                 markerfacecolor='none',markeredgecolor='m',markeredgewidth=2);    
    else : # Obs en suppl�mentaire
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
        if 0 : # Calcul des distances (d^2) des coordonn�es des mod�les F1U (lignes)
               # � celles des Classes F2V (colonne)
            # PLM, ici je ne le fait qu'avec toutes les coordonn�es
            # (c'est plus simple pour n'utiliser que les meilleurs)
            N2take = 5; # Nombre de mod�les � prendre
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
            plt.title("%sSST(%s)) [%s]\n%s%d AFC on classes of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,case_label,method_cah,nb_class));
        elif NIJ==3 :
            plt.title("%sSST(%s)) [%s]\n%s%d AFC on good classes of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,case_label,method_cah,nb_class));
    #
    if 0 : # Contributions Absolues lignes (d�j� calcul�)
           # pr�sentation en courbe (+ somme)
        CtrAi = np.concatenate( (CAi, np.sum(CAi,axis=1).reshape(len(CAi),1)),axis=1)
        leg = list((np.arange(nb_class-1)+1).astype(str)); leg.append('sum')
        fig = plt.figure(); fignum = fig.number
        plt.plot(CtrAi); plt.legend(leg);
        plt.xticks(np.arange(Nmdlok),Tm_, fontsize=8, rotation=45,
                   horizontalalignment='right', verticalalignment='baseline');
        plt.title("AFC: Contributions Absolues lignes (i.e. mod�le) pour chaque axe");
    #
    del Tp_, Tm_
    # Fin AFC
#
#%%%<------------------------------------------------------------------
#%%%<==================================================================
#**********************************************************************
# ........................... GENERALISATION ..........................
def mixtgeneralisation (TMixtMdl) :
    ''' Ici, j'ai : Tmdlok   : une table de NOMS de N mod�les valides ;
        remplac� par 
        Ici, j'ai : Tmdlname : une table de NOMS de N mod�les valides ; 
                    TDmdl4ct : la table correspondante des mod�les 4CT (N, v,12)
        D'autre part, je vais d�finir d'une mani�re ou d'une autre les
        mod�les devant participer � la d�finition du mod�les moyen.
        Ici, il y a plusieurs possibilit�s :
        
        1) Une seule ligne de mod�les � utiliser sans dicernement de classe
           par exemple :
        TMixtMdl = ['ACCESS1-3','NorESM1-ME','bcc-csm1-1','NorESM1-M','CESM1-BGC']
        
        2) Plusieurs lignes de mod�le : une par classe, par exemple
        TMixtMdl = [['ACCESS1-3','NorESM1-ME','bcc-csm1-1','NorESM1-M','CESM1-BGC'],
             ['IPSL-CM5B-LR','ACCESS1-3','MPI-ESM-P','CMCC-CMS','GISS-E2-R-CC'],
             ['bcc-csm1-1-m','MIROC-ESM-CHEM','MIROC-ESM','CSIRO-Mk3-6-0','CanESM2'],
             ['MPI-ESM-MR','CMCC-CM','IPSL-CM5A-MR','FGOALS-g2','MPI-ESM-LR'],
             ['IPSL-CM5A-MR','CNRM-CM5-2','MPI-ESM-MR','MRI-ESM1','MRI-CGCM3'],
             ['FGOALS-s2','CNRM-CM5','CNRM-CM5-2','GFDL-CM3','CMCC-CM'],
             ['GFDL-CM3','GFDL-CM2p1','GFDL-ESM2G','CNRM-CM5','GFDL-ESM2M']];

        Dans les 2 cas, il faut :
        - Prendre les mod�les de TMixtMdl � condition qu'ils soient aussi
          dans Tmdlname
        - Envisager ou pas une 2�me phase ... (pas PLM) 
    '''
    # Je commence par le plus simple : Une ligne de mod�le sans classe en une phase
    # Je prend le cas : CAH effectu�e sur les 6 coordonn�es d�une AFC  nij=3 ... 
#    TMixtMdl = ['CMCC-CM',   'MRI-ESM1',    'HadGEM2-AO','MRI-CGCM3',   'HadGEM2-ES',
#                'HadGEM2-CC','FGOALS-g2',   'CMCC-CMS',  'GISS-E2-R-CC','IPSL-CM5B-LR',
#                'GISS-E2-R', 'IPSL-CM5A-LR','FGOALS-s2', 'bcc-csm1-1'];
    #
    # d�terminer l'indice des mod�les de TMixtMdl dans Tmdlname
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
        print("%d mod�les disponibles (sur %d) pour la generalisation : %s"
              %(len(IMixtMdl),len(TMixtMdl),Tmdlname[IMixtMdl]));
    #
    # Mod�le moyen
    MdlMoy = Dmdlmoy4CT(TDmdl4CT,IMixtMdl);
    if 1 : # Affichage du moyen for CT
        aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,figsize=(12,9));
        plt.suptitle("MdlMoy %s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
                    %(fcodage,andeb,anfin,np.min(MdlMoy),
                     np.max(MdlMoy),np.mean(MdlMoy),np.std(MdlMoy)))
    #
    # Classification du mod�les moyen
    fig = plt.figure();
    fignum = fig.number
    Perfglob_ = Dgeoclassif(sMapO,MdlMoy,LObs,CObs,isnumObs);
    ##!!?? plt.title("MdlMoy(%s), perf=%.0f%c"%(Tmdlok[IMixtMdl,0],100*Perfglob_,'%'),fontsize=sztitle); #,fontweigth='bold');
    plt.title("MdlMoy(%s), perf=%.0f%c"%(Tmdlname[IMixtMdl],100*Perfglob_,'%'),fontsize=sztitle); #,fontweigth='bold');
    #tls.klavier();
#%%-----------------------------------------------------------
if 1 :
    # Je commence par le plus simple : Une ligne de mod�le sans classe en une phase
    # et une seule codification � la fois
    #
    # Sopt-1975-2005 : Les meilleurs mod�les de la p�riode "de r�f�rence" 1975-2005
    #
    # ANOMALIES
    if WITHANO and NIJ==1 :
        print("Cas ANOMALIE : CAH effectu�e sur les 6 coordonn�es d�une AFC  NIJ=1")
        
        TMixtMdl = ['CMCC-CM', 'MRI-ESM1', 'HadGEM2-AO', 'MRI-CGCM3', 'HadGEM2-ES',
                    'HadGEM2-CC', 'FGOALS-g2', 'IPSL-CM5B-LR', 'IPSL-CM5A-LR', 'FGOALS-s2'];
    elif WITHANO and NIJ==3 :
        print("Cas ANOMALIE : CAH effectu�e sur les 6 coordonn�es d�une AFC  NIJ=3")
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
        print("Cas UISST : CAH effectu�e sur les 6 coordonn�es d�une AFC  NIJ=1")
        # Meilleur cluster :
        TMixtMdl = ['CMCC-CM', 'CMCC-CMS'];
    elif UISST and NIJ==3 :
        print("Cas UISST : CAH effectu�e sur les 6 coordonn�es d�une AFC  NIJ=3")
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
        print("Cas GRAD : CAH effectu�e sur les 6 coordonn�es d�une AFC  NIJ=1")
        # Meilleur cluster :
        TMixtMdl = ['CMCC-CMS', 'IPSL-CM5A-LR', 'HadGEM2-CC', 'HadGEM2-ES', 'HadGEM2-AO',
                    'GFDL-CM2p1', 'GFDL-ESM2G', 'GFDL-ESM2M'];
    elif climato=="GRAD" and NIJ==3 :
        print("Cas GRAD : CAH effectu�e sur les 6 coordonn�es d�une AFC  NIJ=3");
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
        print("Cas SST BRUTE : CAH effectu�e sur les 6 coordonn�es d�une AFC NIJ=1");
        # Meilleur cluster :
        TMixtMdl = ['FGOALS-s2', 'CESM1-BGC', 'CESM1-FASTCHEM', 'CCSM4'];
    elif NIJ==3 :
        print("Cas SST BRUTE : CAH effectu�e sur les 6 coordonn�es d�une AFC NIJ=3")
        # Meilleur cluster :
        TMixtMdl = ['MPI-ESM-MR','FGOALS-s2', 'MPI-ESM-LR',    'MPI-ESM-P',
                    'FGOALS-g2', 'CESM1-BGC', 'CESM1-FASTCHEM','CCSM4'];
        #TMixtMdl= ['MPI-ESM-P', 'MPI-ESM-LR', 'FGOALS-g2', 'CMCC-CMS', 'MPI-ESM-MR', 
        #           'FGOALS-s2', 'CMCC-CM', 'inmcm4']; was for [0, 1]
    else : # ---
        print("Cas SST BRUTE : Tous les mod�les jusqu�� la derni�re meilleurs perf en cumul�");
        if IMaxPerfglob_Qm > 0 :
            print("Cum Best n=%d premiers :"%IMaxPerfglob_Qm, Tmdlname[0:IMaxPerfglob_Qm])
        TMixtMdl = ['MPI-ESM-MR','FGOALS-s2', 'MPI-ESM-LR'];
    #
    print("%d modele(s) de generalisation : %s "%(len(TMixtMdl),TMixtMdl))
    #
    mixtgeneralisation(TMixtMdl);
#**********************************************************************
#**********************************************************************
#___________
plt.show();
#___________
print("WITHANO,UISST,climato,NIJ :\n", WITHANO, UISST,climato,NIJ)
print("whole time code %s: %f" %(os.path.basename(sys.argv[0]), time()-tpgm0));
print("\n<end '{}'>".format(case_label))

#======================================================================

#%%









