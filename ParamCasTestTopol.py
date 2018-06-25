from   ctObsMdldef import *
#######################################################################
# PARAMETRAGE (#1) DU CAS - Version de TestTopol
# Comme ParamCas.py mais -sans la partie AFC
# et avec quelques definitions commentes:
#   nbl
#   nbc
#   Parm_app
#   epoch1,radini1,radfin1,epoch2,radini2,radfin2
#   nb_class
#   pcmap
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
# Tableau des modèles (cf dans ctObsMdldef.py)
Tmodels = Tmodels_anyall;
#Tmodels= Tmodels[0:5]; # Pour limiter le nombre de modèles en phase de mise au point
Nmodels = len(Tmodels); # print(Nmodels); sys.exit(0)
#______________________________
# For the Carte Topo (see also ctObsMdl)
#nbl      = 30;  nbc =  4;  # Taille de la carte
#nbl      = 36;  nbc =  6;  # Taille de la carte
#nbl      = 52;  nbc =  8;  # Taille de la carte
#Parm_app = ( 5, 5., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
#Parm_app = ( 50, 5., 1.,  100, 1., 0.1); # Température ini, fin, nb_it
#epoch1,radini1,radfin1,epoch2,radini2,radfin2 = Parm_app
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
FONDTRANS = "Obs"; # "Obs"
#
SAVEFIG   = False; # True; False;
fprefixe  = 'Z_'
#
# Prendre une zone plus petite (concerne aussi l'entrainement)
#SIZE_REDUCTION = 'All';
SIZE_REDUCTION = 'sel'; # selectionne une zone reduite  
#SIZE_REDUCTION = 'RED'; # Ne pas utiliser
    # 'All' : Pas de réduction d'aucune sorte
    # 'sel' : On sélectionne, d'entrée de jeu une zone plus petite,
    #         c'est à dire à la lecture des données. Zone sur laquelle
    #         tous les traitement seront effectues (codification, CT, Classif ...)
    #         ('sel' à la place de 'mini' dans les version précédantes);
    # 'RED' : Seule la classification est faite sur la zone REDuite
    # rem   : PLM 'sel' et 'RED' ne sont pas compatibles; voir ci-après pour
    #         la définition de la zone qui sera concernée
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
OK104=OK105=OK106=False;
OK108=True;
if SIZE_REDUCTION == 'RED' :
    OK107=OK109=False;
else :
    OK107=OK109=False; #True;
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
#nb_class   = 6; #6, 7, 8  # Nombre de classes retenu
ccmap      = cm.jet;      # Accent, Set1, Set1_r, gist_ncar; jet, ... : map de couleur pour les classes
# pour avoir des couleurs à peu près equivalente pour les plots
#pcmap     = ccmap(np.arange(1,256,round(256/nb_class)));ko 512ko, 384ko
#pcmap      = ccmap(np.arange(0,320,round(320/nb_class))); #ok?
#
#######################################################################
