# -*- coding: cp1252 -*-
from __future__ import print_function
import sys
from   time  import time
import numpy as     np
import matplotlib.pyplot as plt
from   matplotlib import cm
#
def errors3(Yref,Yest,errtype=["rms"]) : # pris de upwell2/localdef.py
    ''' [Terr] = .errors(Yref,Yest,errtype)
    | Calcule de différents type d'erreur entre un vecteur de reference (Yref)
    | et un vecteur estimé (Yest). Ces types sont à passer dans le paramètre
    | errtype sous forme de chaine de caractères. Les types (et donc les erreurs)
    | possibles sont :
    | "errq"     : Erreur Quadratique 
    | "errqm"    : Erreur Quadratique Moyenne
    | "rms"      : RMS (Root Mean Squared Error)
    | "biais"    : Bias = moyenne des erreurs 
    | "errqrel"  : Erreur Quadratique relative
    | "errqrelm" : Erreur quadratique relative moyenne
    | "rmsrel"   : La RMS relative   
    | "biaisrel" : Le biais relatif   
    | "errrel"   : L'Erreur relative
    | exemple d'appel :
    |         rms, errqrel = .errors(Yref, Yest,["rms","errqrel"])
    '''
    nberr  = np.size(errtype);
    # As Yref & Yest may have différentes shapes with the same numbers
    # of terme I reshape them.
    Yref   = Yref.reshape(np.prod(np.shape(Yref)))
    Yest   = Yest.reshape(np.prod(np.shape(Yest)))
    Nall   = np.size(Yref);
    # and I check they have the same size
    if np.size(Yest) != Nall :
        print("errors : Yref and Yest must have the same number of element");
        sys.exit(0);
    #
    Err    = Yref - Yest;    # Vecteurs des erreurs
    if ("errqrel" in errtype or "errqrelm" in errtype or "rmsrel" in errtype
        or "biaisrel" in errtype or "errrel" in errtype) :
        Errrel = Err / Yref; # Vecteurs des erreurs relatives  
    
    TERR = np.zeros(nberr);
    for i in np.arange(nberr) :
        if errtype[i] == "errq" :           # Erreur Quadratique 
            TERR[i] = np.sum(Err**2);            
        elif errtype[i] == "errqm" :        # Erreur quadratique moyenne
            errq = np.sum(Err**2);
            TERR[i] = errq/Nall;            
        elif errtype[i] == "rms" :          # La RMS
            errq  = np.sum(Err**2);
            errqm = errq/Nall;
            TERR[i] = np.sqrt(errqm);           
        elif errtype[i] == "biais" :        # Bias = moyenne des erreurs
            TERR[i] = np.sum(Err)/Nall;            
        elif errtype[i] == "errqrel" :      # Erreur Quadratique relative
            TERR[i] = np.sum(Errrel**2);               
        elif errtype[i] == "errqrelm" :     # Erreur quadratique relative moyenne
            TERR[i] = np.sum(Errrel**2)/Nall;           
        elif errtype[i] == "rmsrel" :       # La RMS relative
            TERR[i] = np.sqrt(np.sum(Errrel**2)/Nall);            
        elif errtype[i] == "biaisrel" :     # Biais relatif
            TERR[i] = np.sum(Errrel)/Nall;            
        elif errtype[i] == "errrel" :       # L'Erreur relative
            TERR[i] = np.sum(abs(Errrel))/Nall; 
    #print("TERR=",TERR)
    return TERR
#----------------------------------------------------------------------
def rms(Xref, Xest) :
    '''Calcule et retourne la RMS entre Xref et Xest
    '''
    if len(Xest) != len(Xref) :
        print("rms: Xref and Xest must have the same length");
        sys.exit(0);
    return np.sqrt(np.sum((Xref-Xest)**2) / np.prod(np.shape(Xref)));
#----------------------------------------------------------------------
def nsublc(n,nsubl=0) :
    if nsubl > 0 :
        nbsubl = nsubl;
        nbsubc = np.ceil(1.0*n/nbsubl);
    elif nsubl < 0 :
        nbsubc = -nsubl;
        nbsubl = np.ceil(1.0*n/nbsubc);
    else :
        nbsubc = np.ceil(np.sqrt(n));
        nbsubl = np.ceil(1.0*n/nbsubc);   
            
    nbsubl=int(nbsubl); #nbsubl.astype(int)
    nbsubc=int(nbsubc); #nbsubc.astype(int)
    if nbsubl==1 and nbsubc==1 :
        nbsubc = 2; # Je force à au moins 2 subplots sinon ca plante si y'en a qu'1
    return nbsubl, nbsubc
#----------------------------------------------------------------------
def val2val (X, valfr=0.0,valto=np.nan) :
    X_shape  = np.shape(X);
    X        = np.reshape(X,np.prod(X_shape))
    if ~np.isnan(valfr) :
        Ifr  = np.where(X==valfr)[0];
    else :
        Ifr  = np.where(np.isnan(X))[0];
    X[Ifr]   = valto;
    X        = np.reshape(X, X_shape);
    return X
#----------------------------------------------------------------------
def showimgdata(X, Labels=None, n=1, fr=0, interp=None, cmap=cm.jet, nsubl=0,
                vmin=None, vmax=None, facecolor='w', vnorm=None, sztext=11,
                figsize=(12,16), wspace=0.1, hspace=0.3, top=0.93, bottom=0.01,
                left=0.05, right=0.90,x=0.5,y=0.96,noaxes=True,noticks=True,nolabels=True,
                cbpos='horizontal', fignum=None) :
    import matplotlib.colorbar as cb
    from matplotlib.colors import LogNorm
    nbsubl, nbsubc = nsublc(n,nsubl)
#    if nsubl > 0 :
#        nbsubl = nsubl;
#        nbsubc = np.ceil(1.0*n/nbsubl);
#    elif nsubl < 0 :
#        nbsubc = -nsubl;
#        nbsubl = np.ceil(1.0*n/nbsubc);
#    else :
#        nbsubc = np.ceil(np.sqrt(n));
#        nbsubl = np.ceil(1.0*n/nbsubc);   
#    nbsubl=int(nbsubl); #nbsubl.astype(int)
#    nbsubc=int(nbsubc); #nbsubc.astype(int)
#    if nbsubl==1 and nbsubc==1 :
#        nbsubc = 2; # Je force à au moins 2 subplots sinon ca plante si y'en a qu'1

    # Transformation pour l'affichage
    if vnorm=="truc01" : # avec truc01 
        X,pipo,pipo,pipo,pipo,pipo = truc01(X);
    elif vnorm=="log10normaleB2" : #avec the verboten one
        X,pipo,pipo,pipo = LoiLog10NormaleB2(X);
    elif vnorm=="log" :
        X = np.log(X);
    #
    # VISU
    if vmin is None :
        vmin = np.min(X);
    if vmax is None :
        vmax = np.max(X);
    print(" -- subplots({}x{}) ...".format(nbsubl,nbsubc))
    M, P, Q = np.shape(X[0]);
    fig, axes = plt.subplots(nrows=nbsubl, ncols=nbsubc, num=fignum,
                        sharex=True, sharey=True, figsize=figsize,facecolor=facecolor)
    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    ifig = 0;
    for ax in axes.flat:
        if ifig < n : 
            img  = X[ifig+fr];               
            if M == 1 :
                img  = img.reshape(P,Q)
            elif M != 3 :
                print("showimgdata: Invalid data dimensions image : must be : 1xPxQ or 3xPxQ")
                sys.exit(0);
            else : #=> = 3
                img  = img.transpose(1,2,0);
            if vnorm=="LogNorm" :
                #ims = ax.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation=interp, cmap=cmap);
                #ValueError: Data has no positive values, and therefore can not be log-scaled.
                ims = ax.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation=interp, cmap=cmap);
            else : 
                ims = ax.imshow(img, interpolation=interp, cmap=cmap, vmin=vmin, vmax=vmax);           
            if Labels is not None :
                #ax.set_title(Labels[ifig+fr],fontsize=6);
                print(n,fr,ifig,sztext,x,y)
                ax.set_title(Labels[ifig+fr],fontsize=sztext,x=x,y=y);
            ifig += 1;
            ax.axis("image"); #ax.axis("off");
        if noaxes :
            ax.axis('off')
        elif noticks :
            ax.set_xticks([]); ax.set_yticks([])
        elif nolabels :
            ax.set_xticklabels([]); ax.set_yticklabels([])
    if cbpos == 'horizontal':
        cbar_ax,kw = cb.make_axes([ax for ax in axes.flat],orientation="horizontal",
                                 fraction=0.04,pad=0.02,aspect=40)
    else :
        cbar_ax,kw = cb.make_axes([ax for ax in axes.flat],orientation="vertical",
                                 fraction=0.05,pad=0.02,aspect=30)
    fig.colorbar(ims, cax=cbar_ax, **kw);

#----------------------------------------------------------------------    
def fit01(X,fullret=False,gap01=0.0) :
    ''' Ramène les valeurs de X dans l'intervalle [0, 1] + gap01
    Si fullret == False, seul le résultat Y de cette opération est
    retourné, sinon on retourne également les valeurs (d=max(X)-min(X)
    et min(X/d) qui permettront à la fonction fit01 (ci-dessous) de
    réaliser l'opération inverse.
    '''
    minx = np.min(X);
    maxx = np.max(X)
    d = maxx-minx; 
    Y = (X-minx) / d;
    Y = Y + gap01;
    if fullret :
        return Y, minx, d
    else :
        return Y
def inv_fit01(Y,minx,d,gap01=0.0) :
    ''' Réalise l'opération inverse de la fonction fit01 ci-dessus.
    minx et d sont les paramètres qui ont été retournés par fit01 à
    condition d'avoir positionné le paramètre fullret = True, et le
    paramètre gap01 doit aussi avoir gardé la même valeur.
    '''
    X = (Y-gap01)*d + minx
    return X
#
#----------------------------------------------------------------------    
def codage(CHLA_brute,CODAGE,gap01=0.0) :
    # Codification \ normalisation
    if CODAGE=="log" :
        CHLA = np.log(CHLA_brute);
        coparm = ()
    elif CODAGE=="log01" :
        CHLA = np.log(CHLA_brute);
        CHLA,minx,delta01 = fit01(CHLA,True,gap01);
        coparm = (minx,delta01,gap01)
    elif CODAGE=="log10" :
        CHLA = np.log10(CHLA_brute);
        coparm = ()
    elif CODAGE=="log1001" :
        CHLA,minx,delta01 = fit01(np.log10(CHLA_brute),True);
        coparm = (minx,delta01)
    elif CODAGE=="fit01" :
        CHLA,minx,delta01 = fit01(CHLA_brute,True,gap01);
        coparm = (minx,delta01,gap01)
    else : 
        print("codage : unknown CODAGE (%s)" %CODAGE);
        sys.exit(0);

    return CHLA, coparm
#----------------------------------------------------------------------
def decodage(X,CODAGE,coparm=None) :
    if CODAGE=="log01" :
        minx,delta01,gap01 = coparm;
        Xb     = inv_fit01(X,minx,delta01,gap01)
        Xb     = np.exp(Xb);
    elif CODAGE=="truc01" :
        sigma,minx,deltax,minx2,deltax2 = coparm;
        Xb = inv_truc01(X,sigma,minx,deltax,minx2,deltax2)
    elif CODAGE=="fit01_old1" :
        miny,delta01 = coparm;
        Xb = inv_fit01_old1(X,miny,delta01);
    elif CODAGE=="fit01" :
        minx,delta01,gap01 = coparm;
        Xb = inv_fit01(X,minx,delta01,gap01);
    else :
        print("decodage : unknown CODAGE (%s) ; use 'log01', 'truc01' or 'fit01'"%CODAGE);
        sys.exit(0);
    return Xb
#----------------------------------------------------------------------
# Analyse Factorielle de Correspondances !!!!! (voir document de Sylvie)
def afaco (X, dual=True, Xs=None) :
    F2V=CAj=F1sU=None;
    
    m,p = np.shape(X);
    N   = np.sum(X);
    Fij = X / N;
    fip = np.sum(Fij,axis=1); #print(fip)
    fpj = np.sum(Fij,axis=0);
    F1a = Fij.T / fip;
    F1a = F1a.T;
    F1  = F1a / np.sqrt(fpj);
    
    sqrtfipfpj = np.sqrt(np.outer(fip,fpj)); 
    M = Fij / sqrtfipfpj;  
    T = np.dot(M.T,M); 
    VAPT, VEPT = np.linalg.eig(T); 
    # Ordonner selon les plus grandes valeurs propres
    idx  = sorted(range(len(VAPT)), key=lambda k: VAPT[k], reverse=True); 
    VAPT = VAPT[idx]; 
    VEPT = VEPT[:,idx];
    U    = VEPT[:,1:p];  
    F1U  = np.dot(F1,U);
    VAPT = VAPT[1:p]
    if dual :
        VBPTU = np.sqrt(VAPT) * U;
        F2V   = VBPTU.T / np.sqrt(fpj);
        F2V   = F2V.T
    #
    #Contribution Absolue ligne
    #A    = (F1U**2).T*fip
    F1U2T = (F1U**2).T
    A     = F1U2T*fip
    CAi   = A.T / VAPT;    #print(); tls.tprin(CAi, " %6.3f");
    if dual :
        #Contribution Absolue colonne
        A   = (F2V**2).T*fpj
        CAj = A.T / VAPT; #print(); tls.tprin(CAj, " %6.3f");
    #
    #Contribution Relative ligne
    d2pIG = np.sum((F1 - np.sqrt(fpj))**2, axis=1)
    CRi   = (F1U2T / d2pIG).T; #print(); tls.tprin(CRi, " %6.3f");
    #    
    # Individus supplémentaires
    if Xs is not None : 
        Fijs = Xs / N;
        fips = np.sum(Fijs,axis=1);
        F1as = Fijs.T / fips;
        F1as = F1as.T;
        F1s  = F1as / np.sqrt(fpj);
        F1sU = np.dot(F1s,U);
    #
    #return VAPT, F1U, CAi, F2V, CAj, F1sU
    return VAPT, F1U, CAi, CRi, F2V, CAj, F1sU

#----------------------------------------------------------------------
