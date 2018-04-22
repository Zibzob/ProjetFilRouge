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
from sklearn.decomposition import PCA


# =============================================================================
# FUNCTIONS
# =============================================================================

# IMPORT
# =============================================================================
def import_ct(n_sample, random=1, shuffle=True):
    """import 'cover types' database and prepare it."""
    # 'n_sample' can be either a float < 1 (proportion of observations) or an
    # int > 1 (number of observations), random is the seed (or False)
    df = datasets.fetch_covtype(random_state=random)
    idX_max = int(n_sample*len(df.target)) if n_sample <= 1 else n_sample
    X = df.data[0:idX_max, :]
    Y = df.target[0:idX_max]
    print(df.DESCR)
    print('####################################################')
    print('array des variables explicatives ({:.2f}% du dataset original) : {}'
            .format(idX_max/len(df.target)*100, X.shape))
    print('array des labels ({:.2f}% du dataset original) : {}'
            .format(idX_max/len(df.target)*100, Y.shape))

    return X, Y


def to_df(X, Y, qualitatif=False):
    """transform X and Y to a pandas dataframe."""
    df = pd.dataframe(np.c_[X, Y])
    df.columns = ['elevation', 'aspect', 'slope',
                  'horizontal_distance_to_hydrology',
                  'vertical_distance_to_hydrology',
                  'horizontal_distance_to_roadways', 'hillshade_9am',
                  'hillshade_noon', 'hillshade_3pm',
                  'horizontal_distance_to_fire_points', 'wilderness_area1',
                  'wilderness_area2', 'wilderness_area3', 'wilderness_area4',
                  'soil_type1', 'soil_type2', 'soil_type3', 'soil_type4',
                  'soil_type5', 'soil_type6', 'soil_type7', 'soil_type8',
                  'soil_type9', 'soil_type10', 'soil_type11', 'soil_type12',
                  'soil_type13', 'soil_type14', 'soil_type15', 'soil_type16',
                  'soil_type17', 'soil_type18', 'soil_type19', 'soil_type20',
                  'soil_type21', 'soil_type22', 'soil_type23', 'soil_type24',
                  'soil_type25', 'soil_type26', 'soil_type27', 'soil_type28',
                  'soil_type29', 'soil_type30', 'soil_type31', 'soil_type32',
                  'soil_type33', 'soil_type34', 'soil_type35', 'soil_type36',
                  'soil_type37', 'soil_type38', 'soil_type39', 'soil_type40',
                  'cover_type']
    if qualitatif:
        # a poursuivre : peut être avec grep plutot que les index, et pour les wilderness aussi
        # all soil columns in one
        df['soil'] = None
        idX_soil_debut = 14
        idX_soil_fin = 54
        for i in range(idX_soil_debut, idx_soil_fin):
            l_idX_bool = df.iloc[:, i].astype(bool)
            if True in l_idX_bool:
                df.loc[l_idX_bool, 'soil'] = df.columns[i]
        df.drop(df.columns[idX_soil_debut:idx_soil_fin], axis=1, inplace=True)

    return df


# PRE PROCESSING : SPLIT, NORMALISATION, REDUCTION ETC
# =============================================================================
def split(X, Y, test_ratio=0.2, random=1, strat=True):
    """split the db in training set and test set, given the ratio."""
    if test_ratio != 1:
        Xa, Xt, Ya, Yt = train_test_split(X, Y, shuffle=True, test_size=test_ratio,
                                          stratify=Y if strat else None,
                                          random_state=random)
    else:
        Xa, Xt, Ya, Yt = (0, X, 0, Y)
    return Xa, Xt, Ya, Yt


def norma(Xa, Xt, param_mean=True, param_std=True):
    """centrage et normalisation"""
    sc = StandardScaler(with_mean=param_mean, with_std=param_std)
    sc = sc.fit(Xa)
    Xa = sc.transform(Xa)
    Xt = sc.transform(Xt)
    return Xa, Xt


# réduction de dimension
def pca(Xa, Xt, nb_var=3, alamano=False):
    pca = pca(n_components=nb_var)
    pca.fit(Xa)
    for i in range(nb_var):
        print("valeur propre n°{:d} : {:.1f}% de variance expliquée (lambda = {:.2f})"
              .format(i,
                      100*pca.explained_variance_ratio_[i],
                      pca.singular_values_[i]))
    if alamano:
        print()

    return pca.transform(Xa), pca.transform(Xt)


# MODELES ML
# =============================================================================
# classification
def lda(Xa, Xt, Ya, Yt):
    Xa, Xt = norma(Xa, Xt)
    model_lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
    model_lda.fit(Xa, Ya)
    Y_lda = model_lda.predict(Xa)
    acc_lda = sum(Y_lda == Ya)/Ya.size * 100
    # print('lda : taux d''accuracy = {}%'.format(acc_lda))
    return acc_lda

def qda(Xa, Xt, Ya, Yt):
    Xa, Xt = norma(Xa, Xt)
    model_qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    model_qda.fit(Xa, Ya)
    # print(model_qda.means_)
    Y_qda = model_qda.predict(Xa)
    acc_qda = sum(Y_qda == Ya)/Ya.size * 100
    # print('qda : taux d''accuracy = {}%'.format(acc_qda))
    return acc_qda


def reg_log(Xa, Xt, Ya, Yt, size_vec_c=10, p_cla='multinomial', p_sol='lbfgs',
            fig=False):
    """regression logistique."""
    # liblinear (label pour le un contre tous)

    # optimisation param
    xaa, xav, yaa, yav = split(Xa, Ya, test_ratio=0.2)
    xaa, xav = norma(xaa, xav)
    model_reglog = linear_model.LogisticRegression(tol=1e-5,
                                                   multi_class=p_cla,
                                                   solver=p_sol)
    vectc_log = np.logspace(-3, 2, size_vec_c)
    e_aa_log = np.empty(vectc_log.shape[0])
    e_av_log = np.empty(vectc_log.shape[0])
    for ind_c, c in enumerate(vectc_log):
        model_reglog.c = c
        model_reglog.fit(xaa, yaa)
        e_aa_log[ind_c] = 1 - accuracy_score(yaa, model_reglog.predict(xaa))
        e_av_log[ind_c] = 1 - accuracy_score(yav, model_reglog.predict(xav))
    # if True, plot the training and validation errors
    if fig:
        plt.figure()
        plt.semilogx(vectc_log,
                     e_aa_log,
                     color="blue",
                     linestyle="--",
                     marker="s",
                     markersize=5,
                     label="reg log - app")
        plt.semilogx(vectc_log,
                     e_av_log,
                     color="blue",
                     linestyle="-",
                     marker="s",
                     markersize=5,
                     label="reg log - val")
        plt.xlabel("parametre c")
        plt.ylabel("erreur classification")
        plt.legend(loc="best")
        plt.show()
    # choix du meilleur c
    err_min_val, ind_min = e_av_log.min(), e_av_log.argmin()
    copt = vectc_log[ind_min]

    # entrainement du modele sur toutes les donnees d'apprentissage et
    # évaluation des perf sur jeu de test
    model_reglog.c = copt
    Xa, Xt = norma(Xa, Xt)
    model_reglog.fit(Xa, Ya)
    acc_train = accuracy_score(Ya, model_reglog.predict(Xa)) * 100
    # print("\nregression logistique optimal : accuracy train = {}%".format(acc_train))
    acc_test = accuracy_score(Yt, model_reglog.predict(Xt)) * 100
    # print("regression logistique optimal : accuracy test = {}%".format(acc_test))
    # confusion matrix
    if fig:
        df_conf = pd.dataframe(confusion_matrix(Yt, model_reglog.predict(Xt)))
        df_conf_norm = df_conf.astype('float').divide(df_conf.sum(axis=1), axis=0)
        plt.figure()
        sn.heatmap(df_conf, annot=True, fmt='d')
        plt.figure()
        sn.heatmap(df_conf_norm, annot=True, fmt='.2f')
        plt.show()

    return acc_test


# regression
def reg_lin(Xa, Xt, Ya, Yt):
    """regression linéaire."""
    model_reg_lin = linear_model.linearregression(fit_intercept=True,
                                                  normalize=False,
                                                  copy_x=True,
                                                  n_jobs=1)
    model_reg_lin.fit(Xa, Ya)
    print(np.c_[Yt, model_reg_lin.predict(Xt)])
    print(model_reg_lin.score(Xt, Yt))


def reg_ridge(Xa, Xt, Ya, Yt):
    model_reg_ridge = linear_model.ridge(alpha=1.0,
                                         fit_intercept=True,
                                         normalize=False,
                                         copy_x=True,
                                         maX_iter=None,
                                         tol=0.001,
                                         solver='auto',
                                         random_state=None)
    model_reg_ridge.fit(Xa, Ya)
    print(np.c_[Yt, model_reg_ridge.predict(Xt)])
    print(model_reg_ridge.score(Xt, Yt))


def reg_lasso(Xa, Xt, Ya, Yt):
    model_reg_lasso = linear_model.lasso(alpha=0.1)
    model_reg_lasso.fit(Xa, Ya)
    print(model_reg_lasso.coef_)
    print(model_reg_lasso.intercept_)


# AUTRES
# =============================================================================
def repartition_classes(Y):
    """affiche les proportions de chaque classe du vecteur Y donné en entrée"""
    rep = np.unique(Y, return_counts=True)
    for i in range(len(rep[0])):
        print("classe {:d} : {:d} unités ({:.1f}%)"
              .format(rep[0][i],
                      rep[1][i],
                      rep[1][i] / len(Y) * 100))


def balanced_subsample(X, Y, subsample_size=1.0):
    """ne conserve que des classes équilibrées (réduit le nombre
    d'observation)"""
    class_xs = []
    min_elems = None

    for yi in np.unique(Y):
        elems = X[(Y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        X_ = this_xs[:use_elems]
        Y_ = np.empty(use_elems)
        Y_.fill(ci)

        xs.append(X_)
        ys.append(Y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys).astype('int')

    return xs, ys


def input_gen():
    pass


def input_prep(X, Y, conditions, verb=False):
    """pré-processe les données en fonction de conditions.
    0 : [0, 1, autre] pour 'quanti/quali/tout'
    1 : [0, 1] pour 'equilibré/brut'
    2 : [0:1] pour proportion de la db brute conservée
    3 : [0:1] pour proportion du jeu de test
    4 : [0, 1] pour 'False/True' si on veut normaliser
    5 : [0, 1, ..., n] pour 'False, nb de directions conservées si PCA'"""
    # condition 0
    Y_ = Y
    if conditions[0] == 0:
        X_ = X[:, 0:10]
    elif conditions[0] == 1:
        X_ = X[:, 11:]
    else:
        X_ = X
    # condition 1
    if conditions[1]:
        X_, Y_ = balanced_subsample(X_, Y)
    # condition 2
    X_jete, X_, Y_jete, Y_ = split(X_, Y_, test_ratio=conditions[2])
    del(X_jete, Y_jete)
    # condition 3
    Xa, Xt, Ya, Yt = split(X_, Y_, test_ratio=conditions[3])
    if verb:
        repartition_classes(Y_)
    # condition 4
    if conditions[4]:
        Xa, Xt = norma(Xa, Xt)
    # condition 5
    if conditions[5]:
        Xa, Xt = pca(Xa, Xt, conditions[5])

    return Xa, Xt, Ya, Yt


def ml_algos(Xa, Xt, Ya, Yt):
    """résultats sur les différents modèles utilisés."""
    print("Taille Xa : {}, taille Xt : {}".format(Xa.shape, Xt.shape))
    res = []
    res.append(lda(Xa, Xt, Ya, Yt))
    res.append(qda(Xa, Xt, Ya, Yt))
    res.append(reg_log(Xa, Xt, Ya, Yt,
                       size_vec_c=15,
                       p_cla='multinomial',
                       p_sol='lbfgs',
                       fig=False))
    return res


def broad_test_campaign():
    data = []
    X, Y = import_ct(n_sample=1, random=2)
    conditions = [[0, 0, 0.03, 0.2, 0, 0],
                  [0, 0, 0.03, 0.2, 1, 0],
                  [0, 1, 1, 0.2, 0, 0],
                  [0, 1, 1, 0.2, 1, 0]]
    for cond in conditions:
        Xa, Xt, Ya, Yt = input_prep(X, Y, cond, verb=True)
        data.append(cond + ml_algos(Xa, Xt, Ya, Yt))

    return data


def broad_test_campaign_old():
    """fait varier des paramètres en entrée et récupère les résultats
    (accuracy) provenant de différents modèles de classification.
    dans l'odre, voici les paramètres en jeu :
        - variables explicatives (toutes, juste qualitatives/quantitatives)
        - classes équilibrées (True/False)
        - proportion du dataset global utilisée (]0 1])
        - proportion du jeu de test (]0 1[)
        - normalisation (True/False)
        - pca (False/nombre de composantes conservées)"""
    table_param_res = {'cas 1':({'typevar':'quali',
                                 'balanced':True,
                                 'proportion':0.1,
                                 'testratio':0.2,
                                 'normalisation':0,
                                 'pca':0},
                                []),
                       'cas 2':({'typevar':'quant',
                                 'balanced':True,
                                 'proportion':0.1,
                                 'testratio':0.2,
                                 'normalisation':0,
                                 'pca':0},
                                []),
                       'cas 3':({'typevar':'toute',
                                 'balanced':True,
                                 'proportion':0.1,
                                 'testratio':0.2,
                                 'normalisation':0,
                                 'pca':0},
                                [])
                       }

    # import des données
    X, Y = import_ct(n_sample=1, random=2)
    # conditions / paramètres
    for i in table_param_res:
        cond = table_param_res[i][0]
        # Condition 0 sur les variables explicatives conservées
        if cond['typeVar'] == 'quant':
            X_ = X[:, 0:10]
        elif cond['typeVar'] == 'quali':
            X_ = X[:, 11:]
        else:
            X_ = X
        # Condition 1 sur l'équilibre des proportions des classes
        if cond['balanced']:
            X_, Y_ = balanced_subsample(X_, Y)
        # Condition 2 sur la proportion du dataset initial conservée
        X_jete, X_, Y_jete, Y_ = split(X_, Y_, test_ratio=cond['proportion'])
        del(X_jete, Y_jete)
        # Condition 3 sur la proportion du jeu de test
        Xa, Xt, Ya, Yt = split(X_, Y_, test_ratio=cond['testRatio'])
        repartition_classes(Y_)
        # Condition 4 sur la normalisation
        if cond['normalisation']:
            Xa, Xt = norma(Xa, Xt)
        # Condition 5 sur la PCA
        if cond['pca']:
            Xa, Xt = pca(Xa, Xt, cond[5])

        # Résultats sur les différents modèles utilisés
        table_param_res[i][1].append(lda(Xa, Xt, Ya, Yt))
        table_param_res[i][1].append(qda(Xa, Xt, Ya, Yt))
        table_param_res[i][1].append(reg_log(Xa, Xt, Ya, Yt,
                                             size_vec_C=15,
                                             p_cla='multinomial',
                                             p_sol='lbfgs',
                                             fig=False))
    res = [table_param_res[key][1] for key in table_param_res]
    res = pd.DataFrame(res,
                       columns=["LDA", "QDA", "Regression logistique"],
                       index=['Cas 1', 'Cas 2', 'Cas 3'])

    return res


# MAIN
# =============================================================================
if __name__ == '__main__':
    res = broad_test_campaign()
    #X, Y = import_ct(n_sample=1, random=2)
    #conditions = [0, 1, 1, 0.2, 1, 0]
    #Xa, Xt, Ya, Yt = input_prep(X, Y, conditions, verb=True)
    #res = ml_algos(Xa, Xt, Ya, Yt)
    #print(res)

# Regular main
    if 0:
        # !!!! ALL DB imported, to keep the label proportions of the dataset
        X, Y = import_ct(n_sample=1, random=2)
        X, Y = balanced_subsample(X, Y)

        #df = to_df(X, Y, qualitatif=True)
        # X = X[:, 0:10]
        X_jete, X, Y_jete, Y = split(X, Y, test_ratio=0.1)
        del(X_jete, Y_jete)
        repartition_classes(Y)
        print(pd.DataFrame(X).describe().round(decimals=2)) # stats basiques sur X
        Xa, Xt, Ya, Yt = split(X, Y, test_ratio=0.3)
        Xa, Xt = norma(Xa, Xt)

# Test des selections
    if 0:
        Xa, Xt = pca(Xa, Xt, 20)

# Test des classifications
    if 0:
        lda(Xa, Xt, Ya, Yt)
        qda(Xa, Xt, Ya, Yt)
        reg_log(Xa, Xt, Ya, Yt, size_vec_C=15, p_cla='multinomial',
                p_sol='lbfgs', fig=True)

# Test des regressions
    if 0:
        X_reg = np.c_[X[:, 1:10], Y]
        Y_reg = X[:, 0]
        Xa, Xt, Ya, Yt = split(X_reg, Y_reg, test_ratio=0.2, strat=False)
        reg_lin(Xa, Xt, Ya, Yt)
        reg_ridge(Xa, Xt, Ya, Yt)
        reg_lasso(Xa, Xt, Ya, Yt)
