from   ctObsMdldef import *
#######################################################################
# FLAGS DE COMPORTEMENT
#======================================================================
#SAVEFIG    = False;
SAVEFIG    = True;
#SAVEMAP    = False;
SAVEMAP    = True;
# -----------------------------------------------------------------------------
#REWRITEMAP = True;
REWRITEMAP = False;
# -----------------------------------------------------------------------------
RELOADMAP = True;
#RELOADMAP = False;
# -----------------------------------------------------------------------------
#######################################################################
# PARAMETRAGE (#1) DU CAS
#======================================================================

# Choix du jeu de données
# On doit distinguer les données d'OBS (qui servent à la CT
# des données modèles à classer en fonction de ces Obs).

DATAOBS = "raverage_1975_2005";
#DATAOBS = "rcp_2006_2017";
#DATAOBS = "raverage_1930_1960";
#DATAOBS = "raverage_1944_1974";

DATAMDL = "raverage_1975_2005";
#DATAMDL = "rcp_2006_2017";
#DATAMDL = "raverage_1930_1960";
#DATAMDL = "raverage_1944_1974";
#
if DATAMDL=="rcp_2006_2017" :      # on précise le scénario
    scenar = "rcp85";              # rcp26 rcp45 rcp85
#
## Tableau des modèles (cf dans ctObsMdldef.py)
#Tmodels = Tmodels_anyall;
##Tmodels= Tmodels[0:5]; # Pour limiter le nombre de modèles en phase de mise au point
#Nmodels = len(Tmodels); # print(Nmodels); sys.exit(0)
#______________________________
# For the Carte Topo (see also ctObsMdl)
#
#
# -----------------------------------------------------------------------------
# Conditions d'execution:
#    | --------------- CARACTERISTIQUES ----------------- | -- VARIABLES ---- |
#  - Architecture de la carte SOM ......................... nbl, nbc
#  - Parametres d'entrainement en deux phases ............. Parm_app
#  - Zone geographique consideree (toute, reduite, ...) ... SIZE_REDUCTION
#  - Nombre de classes .................................... nb_class
#  - Nombre de clusters et de coordonnees pour l'AFC ...... nb_clust, NBCOORDAFC4CAH
#  - Critere pour evaluation des performances pour l'AFC .. NIJ
# -----------------------------------------------------------------------------
# Prendre une zone plus petite (concerne aussi l'entrainement)
    # 'All' : Pas de réduction d'aucune sorte
    # 'sel' : On sélectionne, d'entrée de jeu une zone plus petite,
    #         c'est à dire à la lecture des données. Zone sur laquelle
    #         tous les traitement seront effectues (codification, CT, Classif ...)
    #         ('sel' à la place de 'mini' dans les version précédantes);
    # 'RED' : Seule la classification est faite sur la zone REDuite
    # rem   : PLM 'sel' et 'RED' ne sont pas compatibles; voir ci-après pour
    #         la définition de la zone qui sera concernée
    # AFC
# -----------------------------------------------------------------------------
# NIJ = 0 : Pas d'AFC
#     = 1 : nombre d'elt par classe
#     = 2 : perf par classe
#     = 3 : nombre d'elt bien classés par classe
#           (le seul qui devrait survivre à mon sens)
#
if 1 : # conditions Code Charles: GRANDE ZONE
    # A - Grande zone de l’upwelling (25x36) : Longitude : -44 à -9.5 ; Latitude : 29.5 à 5.5
    #   * Carte topologique et CAH : 30x4 (5, 5, 1, - 16, 1, 0.1) : TE=0.6824 ; QE=0.153757
    #   Nb_classe = 7
    nbl            = 30;  nbc =  4;  # Taille de la carte
    Parm_app       = ( 5, 5., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
    #Parm_app       = ( 50, 5., 1.,  160, 1., 0.1); # Température ini, fin, nb_it
    SIZE_REDUCTION = 'All';
    nb_class       = 7; #6, 7, 8  # Nombre de classes retenu
    # et CAH for cluster with AFC
    NIJ            = 2;
    #PerfGlobIsMean = True;
    PerfGlobIsMean = False;
    nb_clust       = 4; # Nombre de cluster
    NBCOORDAFC4CAH = nb_class - 1; # n premières coordonnées de l'afc à
    #NBCOORDAFC4CAH = nb_class; # n premières coordonnées de l'afc à
                    # utiliser pour faire la CAH (limité à nb_class-1).
elif 1 : # conditions Code Charles: PETITE ZONE
    # B - Sous-zone ciblant l’upwelling (13x12) :    LON: 16W à 28W LAT : 10N à 23N
    #   * Carte topologique et CAH : 17x6 (4, 4, 1, - 16, 1, 0.1) : TE=0.6067 ; QE=0.082044
    #   Nb_classe = 4
    nbl            = 17;  nbc =  6;  # Taille de la carte
    Parm_app       = ( 4, 4., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
    SIZE_REDUCTION = 'sel';
    nb_class       = 4; #6, 7, 8  # Nombre de classes retenu
    # et CAH for cluster with AFC
    NIJ            = 2;
    #PerfGlobIsMean = True;
    PerfGlobIsMean = False;
    nb_clust       = 5; # Nombre de cluster
    NBCOORDAFC4CAH = nb_class - 1; # n premières coordonnées de l'afc à
else : # valeurs par defaut
    #nbl      = 6;  nbc =  6;  # Taille de la carte
    #nbl      = 30;  nbc =  4;  # Taille de la carte
    nbl       = 36;  nbc =  6;  # Taille de la carte
    #nbl      = 52;  nbc =  8;  # Taille de la carte
    # -------------------------------------------------------------------------
    #Parm_app = ( 5, 5., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
    Parm_app = ( 50, 5., 1.,  100, 1., 0.1); # Température ini, fin, nb_it
    #Parm_app = ( 500, 5., 1.,  1000, 1., 0.1); # Température ini, fin, nb_it
    #Parm_app = ( 2000, 5., 1.,  5000, 1., 0.1); # Température ini, fin, nb_it
    # -------------------------------------------------------------------------
    #SIZE_REDUCTION = 'All';
    SIZE_REDUCTION = 'sel'; # selectionne une zone reduite  
    #SIZE_REDUCTION = 'RED'; # Ne pas utiliser
    # -------------------------------------------------------------------------
    nb_class   = 7; #6, 7, 8  # Nombre de classes retenu
    # -------------------------------------------------------------------------
    NIJ        = 2; # cas de
    # -------------------------------------------------------------------------
    PerfGlobIsMean = True;
    #PerfGlobIsMean = False;
    # -------------------------------------------------------------------------
    nb_clust   = 7; # Nombre de cluster
    NBCOORDAFC4CAH = nb_class - 1; # n premières coordonnées de l'afc à
# -----------------------------------------------------------------------------
#
epoch1,radini1,radfin1,epoch2,radini2,radfin2 = Parm_app
case_label_base = "M{}x{}_Ep1-{}_Ep2-{}".format(nbl, nbc, epoch1, epoch2)
#______________________________
# Complémentation des nan pour les modèles
MDLCOMPLETION = True; # True=>Cas1
#
# Other stuff ...
TRANSCOCLASSE = 'STD'; # Permet le transcodage des classes de façon à ce
    # que les (indices de) classes correspondent à l'un des critères :
    # 'GAP' 'GAPNM' 'STD' 'MOY' 'MAX' 'MIN' 'GRAD'. Attention ce critère
    # est appliqué sur les référents ...
    # Avec la valeur '' le transcodage n'est pas requis.
#
# -----------------------------------------------------------------------------
FONDTRANS  = "Obs"; # "Obs"
#
# -----------------------------------------------------------------------------
FIGSDIR    = 'figs'
# -----------------------------------------------------------------------------
MAPSDIR    = 'maps'
# -----------------------------------------------------------------------------
mapfileext = ".pkl" # exten,sion du fichier des MAP
# -----------------------------------------------------------------------------
#
if SIZE_REDUCTION == 'All' :
    fprefixe  = 'Z_'
elif SIZE_REDUCTION == 'sel' :
    fprefixe  = 'Zsel_'
elif SIZE_REDUCTION == 'RED' :
    fprefixe  = 'ZR_'
#______________________________
# Choix des figures à produire
# fig Pour chaque modèle et par pixel :
# 104 : Classification avec, "en transparance", les mals classés
#       par rapport aux obs (*1)
# 105 : Classification (*1)
# 106 : Courbes des moyennes mensuelles par classe
# 107 : Variance (not 'RED' compatible)
# 108 : Classification en Model Cumulé Moyen (*1)
# 109 : Variance sur les Models Cumulés Moyens (not 'RED' compatible)
# (*1): (Pour les modèles les Perf par classe sont en colorbar)
# rem : Les classes peuvent être transcodée de sorte qu'elles correspondent
#       à un critère (cf paramètre TRANSCOCLASSE)
# PLM, le figures 107 et 109 n'ont pas été adaptées
# pour 'RED'. Je ne sais pas si c'est vraiment intéressant de le faire,
# attendre que le besoin émerge.
#True;
#OK104=OK105=OK106=False;
OK104=OK105=OK106=True;
OK108=True;
if SIZE_REDUCTION == 'RED' :
    OK107=OK109=False;
else :
    OK107=OK109=False;
    #OK107=OK109=True;
#OK104=OK105=OK106=OK107=OK108=OK109=True;
#
if OK108 or OK109 :
    MCUM = True; # Moyenne des Models climatologiques CUmulés
#______________________________
if DATAMDL=="raverage_1975_2005" :  # Les runs moyens des modèles calculés par Carlos
    Nda  = 31; #!!! Prendre que les Nda dernières années (All Mdls Compatibles)
    anfin=2005; andeb = anfin-Nda+1; # avec .MAT andeb est inclus
elif DATAMDL=="raverage_1930_1960" :  # Les runs moyens des modèles calculés par Carlos
    Nda  = 31; #!!! Prendre que les Nda dernières années (All Mdls Compatibles)
    anfin=1960; andeb = anfin-Nda+1; # avec .MAT andeb est inclus
elif DATAMDL=="rcp_2006_2017" :  # Les sénarios rcp26, rcp45 et rcp85 de 2006 à 2017
    Nda  = 12; # on prend tout
    anfin=2017; andeb = anfin-Nda+1; # avec .MAT andeb est inclus
#______________________________
INDSC     = False;  # IndSC : Indicateur de Saisonalité Climatologique 
#
# Transfo des données-séries brutes, dans l'ordre :
TRENDLESS = False;  # Suppression de la tendance
WITHANO   = True;   # False,  True
#
climato   = None;   # None  : climato "normale" : moyenne mensuelle par pixel et par mois
                    # "GRAD": pente b1 par pixel et par mois
#
# b) Transfo des moyennes mensuelles par pixel dans l'ordre :
UISST     = False;  # "after", "before" (Som(Diff) = Diff(Som))
NORMMAX   = False;  # Dobs =  Dobs / Max(Dobs)
CENTRED   = False ;
fcodage=""; fshortcode="";
if climato=="GRAD" :
    fcodage=fcodage+"GRAD(";       fshortcode=fshortcode+"Grad"
if INDSC :
    fcodage=fcodage+"INDSC(";       fshortcode=fshortcode+"Indsc"
if TRENDLESS :
    fcodage=fcodage+"TRENDLESS(";   fshortcode=fshortcode+"Tless"
if WITHANO :
    fcodage=fcodage+"ANOMALIE(";    fshortcode=fshortcode+"Ano"
#if CENTREE : fcodage=fcodage+"CENTREE(";
#-> Climatologie (Moyenne mensuelle par pixel)
if UISST :
    fcodage=fcodage+"UI(";          fshortcode=fshortcode+"Ui"
if NORMMAX :
    fcodage=fcodage+"NORMMAX(";     fshortcode=fshortcode+"Nmax"
if CENTRED :
    fcodage=fcodage+"CENTRED(";     fshortcode=fshortcode+"Ctred"
#print(fcodage); sys.exit(0);
#______________________________
# for CAH for classif with CT (see ctObsMdl for more)
method_cah = 'ward';      # 'average', 'ward', 'complete','weighted'
dist_cah   = 'euclidean'; #
#nb_class   = 7; #6, 7, 8  # Nombre de classes retenu
ccmap      = cm.jet;      # Accent, Set1, Set1_r, gist_ncar; jet, ... : map de couleur pour les classes
# pour avoir des couleurs à peu près equivalente pour les plots
#pcmap     = ccmap(np.arange(1,256,round(256/nb_class)));ko 512ko, 384ko
pcmap      = ccmap(np.arange(0,320,round(320/nb_class))); #ok?
#
#______________________________
AFCWITHOBS = True; #False True : afc avec ou sans les Obs ? dans le tableau de contingence 
pa=1; po=2; # Choix du plan factoriel (axes)
#pa=3; po=4; # Choix du plan factoriel
#
#
#Flag visu classif des modèles des cluster
AFC_Visu_Classif_Mdl_Clust  = []; # liste des cluster a afficher (à partir de 1)
#AFC_Visu_Classif_Mdl_Clust = [1,2,3,4,5,6,7]; 
#Flag visu Modèles Moyen 4CT des cluster
AFC_Visu_Clust_Mdl_Moy_4CT  = []; # liste des cluster a afficher (à partir de 1)
#AFC_Visu_Clust_Mdl_Moy_4CT = [1,2,3,4,5,6,7];
#######################################################################
case_label_base="Case_{}{}_NIJ{:d}".format(fprefixe,case_label_base,NIJ)
if PerfGlobIsMean :
    case_label_base += "_PGIM"
#######################################################################
