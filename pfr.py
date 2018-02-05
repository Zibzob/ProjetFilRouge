#!/usr/bin/env python3
# coding: utf-8

import sys
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# =============================================================================
# Functions
# =============================================================================

# IMPORT 
# =============================================================================
def import_ct(n_sample=1000, random=1, shuffle=True):
    """Import 'cover types' database and prepare it."""
    # 'n_sample' can be either a float < 1 (proportion of observations) or an
    # int > 1 (number of observations), random is the seed (or False)
    df = datasets.fetch_covtype(random_state=random)
    idx_max = int(n_sample*len(df.target)) if n_sample <= 1 else n_sample
    X = df.data[0:idx_max, :]
    Y = df.target[0:idx_max]
    print(df.DESCR)
    print('####################################################')
    print('Array des variables explicatives ({:.3f}% du dataset original) : {}'
            .format(idx_max/len(df.target)*100, X.shape))
    print('Array des labels ({:.3f}% du dataset original) : {}'
            .format(idx_max/len(df.target)*100, Y.shape))

    return X, Y


def to_df(X, Y, qualitatif=False):
    """Transform X and Y to a pandas dataframe."""
    df = pd.DataFrame(np.c_[X, Y])
    df.columns = ['Elevation', 'Aspect', 'Slope',
                  'Horizontal_Distance_To_Hydrology',
                  'Vertical_Distance_To_Hydrology',
                  'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                  'Hillshade_Noon', 'Hillshade_3pm',
                  'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
                  'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                  'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
                  'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
                  'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
                  'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
                  'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
                  'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
                  'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
                  'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
                  'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
                  'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',
                  'Cover_Type'] 
    if qualitatif:
        # A POURSUIVRE : peut être avec grep plutot que les index, et pour les wilderness aussi
        # All soil columns in one
        df['Soil'] = None
        idx_soil_debut = 14
        idx_soil_fin = 54
        for i in range(idx_soil_debut, idx_soil_fin):
            l_idx_bool = df.iloc[:, i].astype(bool)
            if True in l_idx_bool:
                df.loc[l_idx_bool, 'Soil'] = df.columns[i]
        df.drop(df.columns[idx_soil_debut:idx_soil_fin], axis=1, inplace=True)

    return df


# PRE PROCESSING : split, reduction etc
# =============================================================================
def split(X, Y, test_ratio=0.2, random=1):
    """Split the DB in training set and test set, given the ratio."""
    Xa, Xt, Ya ,Yt = train_test_split(X, Y, shuffle=True, test_size=test_ratio,
                                      stratify=Y, random_state=random)
    return Xa, Xt, Ya, Yt


def reduction(Xa, Xt, param_mean=True, param_std=True):
    """Centrage et normalisation"""
    sc = StandardScaler(with_mean=param_mean, with_std=param_std)
    sc = sc.fit(Xa)
    Xa = sc.transform(Xa)
    Xt = sc.transform(Xt)
    return Xa, Xt


# MODELES ML
# =============================================================================
def lda(Xa, Xt, Ya, Yt):
    Xa, Xt = reduction(Xa, Xt)
    model_lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
    model_lda.fit(Xa, Ya)
    Y_lda = model_lda.predict(Xa)
    err_lda = sum(Y_lda != Ya)/Ya.size
    print('LDA : taux d''erreur = {}%'.format(100*err_lda))
    return err_lda

def qda(Xa, Xt, Ya, Yt):
    Xa, Xt = reduction(Xa, Xt)
    model_qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    model_qda.fit(Xa, Ya)
    # print(model_qda.means_)
    Y_qda = model_qda.predict(Xa)
    err_qda = sum(Y_qda!= Ya)/Ya.size
    print('QDA : taux d''erreur = {}%'.format(100*err_qda))
    return err_qda


def reg_log(Xa, Xt, Ya, Yt, size_vec_C=10, p_cla='multinomial', p_sol='lbfgs'):
    """Regression logistique."""
    # liblinear (label pour le un contre tous)

    # Optimisation param
    Xaa, Xav, Yaa, Yav = split(Xa, Ya, test_ratio=0.2)
    Xaa, Xav = reduction(Xaa, Xav)
    model_reglog = linear_model.LogisticRegression(tol=1e-5,
                                                   multi_class=p_cla,
                                                   solver=p_sol) 
    vectC_log = np.logspace(-3, 2, size_vec_C)
    e_aa_log = np.empty(vectC_log.shape[0])
    print(e_aa_log)
    e_av_log = np.empty(vectC_log.shape[0])
    for ind_C, C in enumerate(vectC_log):
        model_reglog.C = C
        model_reglog.fit(Xaa, Yaa)
        e_aa_log[ind_C] = 1 - accuracy_score(Yaa, model_reglog.predict(Xaa))
        e_av_log[ind_C] = 1 - accuracy_score(Yav, model_reglog.predict(Xav))
    # If True, plot the training and validation errors
    if False:
        plt.figure()
        plt.semilogx(vectC_log,
                     e_aa_log,
                     color="blue",
                     linestyle="--",
                     marker="s",
                     markersize=5,
                     label="Reg Log - App")
        plt.semilogx(vectC_log,
                     e_av_log,
                     color="blue",
                     linestyle="-",
                     marker="s",
                     markersize=5,
                     label="Reg Log - Val")
        plt.xlabel("Parametre C")
        plt.ylabel("Erreur Classification")
        plt.legend(loc="best")
        plt.show()
    # Choix du meilleur C
    err_min_val, ind_min = e_av_log.min(), e_av_log.argmin()
    Copt = vectC_log[ind_min]

    # Entrainement du modele sur toutes les donnees d'apprentissage et
    # évaluation des perf sur jeu de test
    model_reglog.C = Copt
    Xa, Xt = reduction(Xa, Xt)
    model_reglog.fit(Xa, Ya)
    err_app = 1 - accuracy_score(Ya, model_reglog.predict(Xa))
    print("\nRegression logistique optimal : erreur apprentissage = {}".format( err_app))
    err_test = 1 - accuracy_score(Yt, model_reglog.predict(Xt))
    print("Regression logistique optimal : erreur test = {}".format(err_test))
    # Confusion matrix
    df_conf = pd.DataFrame(confusion_matrix(Yt, model_reglog.predict(Xt)))
    df_conf_norm = df_conf.astype('float').divide(df_conf.sum(axis=1), axis=0)
    plt.figure()
    sn.heatmap(df_conf, annot=True, fmt='d')
    plt.figure()
    sn.heatmap(df_conf_norm, annot=True, fmt='.2f')
    plt.show()




# Main
if __name__ == '__main__':
    # All DB imported, to keep the label proportions of the dataset
    X, Y = import_ct(n_sample=1, random=2)
    df = to_df(X, Y, qualitatif=True)
    df.info()
    X = X[:, 0:2]
    X_jete, X, Y_jete, Y = split(X, Y, test_ratio=0.01)

    Xa, Xt, Ya, Yt = split(X, Y, test_ratio=0.2)
    # lda(Xa, Xt, Ya, Yt)
    # qda(Xa, Xt, Ya, Yt)
    reg_log(Xa, Xt, Ya, Yt, size_vec_C=15, p_cla='multinomial', p_sol='lbfgs')
