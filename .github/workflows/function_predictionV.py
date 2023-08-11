# The code in which is implemented the machine learning model used to get the prediction
# This code is run from list_torun.py

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
import numpy as np
import joblib
from xgboost import XGBClassifier, plot_importance

plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 30})
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics

from sklearn.svm import SVC

import time
import sys
import warnings
import argparse

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from numpy import mean
from numpy import std

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold


def plot_feature_importance(importance,names,model_type,namefilex):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df.to_csv(namefilex)
    #Define size of bar plot
    plt.figure()
    #Plot Searborn bar chart
    #sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    if fi_df.shape[0]>20 :
        nfeatures = 20
    else :
        nfeatures = fi_df.shape[0]

    sns.barplot(x=fi_df.iloc[0:nfeatures,1], y=fi_df.iloc[0:nfeatures,0])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def tic():
    global _start_time
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))
    return (str('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec)))

def run_prediction(path_parent, csv_file_name,output_folder,n_repetitions,fold_type_out,fold_type_inn,n_folds_out,n_folds_in):

    foldername_figures = csv_file_name
    file_results=output_folder + '/' + csv_file_name + '.txt'

    file_input = path_parent + '/' + csv_file_name
    print("Input csv file: " + file_input)
    folder_figure = output_folder + '/' + foldername_figures

    if not os.path.exists(folder_figure):
        os.makedirs(folder_figure)

    code_id = 'ID'
    code_outcome = 'class'
    seed = 45

# ===========  process ===========

    text_file_output = open(file_results, "w+")

    _start_time = time.time()
    tic()

    data = pd.read_csv(file_input)
    print(data.shape)
    X = data.drop([code_id, code_outcome], axis=1)
    y = data[code_outcome]
    Z = data[code_id]

    accuracy_rf = list()
    f1_rf = list()
    precision_rf = list()
    recall_rf = list()
    auc_rf = list()

    accuracy_rf_std = list()
    f1_rf_std = list()
    precision_rf_std = list()
    recall_rf_std = list()
    auc_rf_std = list()

    accuracy_xgb = list()
    f1_xgb = list()
    precision_xgb = list()
    recall_xgb = list()
    auc_xgb = list()

    accuracy_xgb_std = list()
    f1_xgb_std = list()
    precision_xgb_std = list()
    recall_xgb_std = list()
    auc_xgb_std = list()

    # PARAMETERS
    params_rf = {'bootstrap': [True, False], 'class_weight': ['balanced'], 'max_depth': [9, 5, 3, 2, None],
                 'n_estimators': [100, 500], 'min_samples_split': [2, 3, 5, 7, 10]}

    params_xgb = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 'min_child_weight': [1, 5, 10],
                  'gamma': [0.5, 1, 5], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0],
                  'max_depth': [3, 4, 5]}

# start undersampling
    for k in range(n_repetitions):

        outer_results_accuracy_m5 = list()
        outer_results_f1_m5 = list()
        outer_results_precision_m5 = list()
        outer_results_recall_m5 = list()
        outer_results_auc_m5 = list()

        outer_results_accuracy_xgb = list()
        outer_results_f1_xgb = list()
        outer_results_precision_xgb = list()
        outer_results_recall_xgb = list()
        outer_results_auc_xgb = list()

        # save outcome distribution
        figure_data_balanced = folder_figure + '/'+ 'balancedData{0}.png'.format(str(k))
        bar_y = [(y == 1).sum(), (y == 0).sum()]
        bar_x = ["1", "0"]
        plt.figure()
        splot = sns.barplot(bar_x, bar_y)
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')
        plt.xlabel("Disease")
        plt.ylabel("Number of subjects")
        plt.savefig(figure_data_balanced, bbox_inches='tight')
        plt.clf()

        j = 1
        if fold_type_out == 1:
            cv_outer = KFold(n_splits=n_folds_out, shuffle=True, random_state=seed)
        else:
            cv_outer = StratifiedKFold(n_splits=n_folds_out, shuffle=True, random_state=seed)

        # Repeated k-folds
        for train_idx, test_idx in cv_outer.split(X, y):
            print(('{} of KFold {}'.format(j, cv_outer.n_splits)))
            text_file_output.write('{} of KFold {} \n'.format(j, cv_outer.n_splits))
            train_data = X.iloc[train_idx, :]
            train_target = y.iloc[train_idx]
            val_data = X.iloc[test_idx, :]
            val_target = y.iloc[test_idx]
            participants_ID = Z.iloc[test_idx]

            # save outcome distribution
            namefilef = "balancedTrainingkfold{0}_{1}.png".format(str(k),str(j))
            figure_data_balanced = folder_figure+'/' + namefilef
            bar_y = [(train_target == 1).sum(), (train_target == 0).sum()]
            bar_x = ["1", "0"]
            plt.figure()
            splot = sns.barplot(bar_x, bar_y)
            for p in splot.patches:
                splot.annotate(format(p.get_height(), '.1f'),
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center',
                               xytext=(0, 9),
                               textcoords='offset points')
            plt.xlabel("Disease")
            plt.ylabel("Number of subjects")
            plt.savefig(figure_data_balanced, bbox_inches='tight')
            plt.clf()

            namefilef = "balancedTestingkfold{0}_{1}.png".format(str(k),str(j))
            figure_data_balanced = folder_figure+'/' + namefilef
            bar_y = [(val_target == 1).sum(), (val_target == 0).sum()]
            bar_x = ["1", "0"]
            plt.figure()
            splot = sns.barplot(bar_x, bar_y)

            for p in splot.patches:
                splot.annotate(format(p.get_height(), '.1f'),
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center',
                               xytext=(0, 9),
                               textcoords='offset points')
            plt.xlabel("Disease")
            plt.ylabel("Number of subjects")
            plt.savefig(figure_data_balanced, bbox_inches='tight')
            plt.clf()

            if fold_type_inn == 1:
                cv_inner = KFold(n_splits=n_folds_in, shuffle=True, random_state=seed)
            if fold_type_inn == 2:
                cv_inner = StratifiedKFold(n_splits=n_folds_in, shuffle=True, random_state=seed)

            # # RandomForest
            # model = RandomForestClassifier(n_estimators=80, class_weight="balanced")
            # gd_search = GridSearchCV(model, params_rf, scoring='f1_macro', n_jobs=-1, cv=cv_inner).fit(train_data,
            #                                                                                            train_target)
            # best_model = gd_search.best_estimator_
            # classifier = best_model.fit(train_data, train_target)
            # y_pred_prob = classifier.predict_proba(val_data)[:, 1]
            # auc = metrics.roc_auc_score(val_target, y_pred_prob)
            # print("Val Acc:", auc, "Best GS Acc:", gd_search.best_score_, "Best Params:", gd_search.best_params_)
            # text_file_output.write("Val Acc: %f Best GS Acc: %f \n"  %(auc, gd_search.best_score_))
            # y_hat = classifier.predict(val_data)
            #
            # outer_results_accuracy_m5.append(metrics.accuracy_score(val_target, y_hat))
            # outer_results_f1_m5.append(metrics.f1_score(val_target, y_hat))
            # outer_results_auc_m5.append(metrics.roc_auc_score(val_target, y_hat))
            # outer_results_precision_m5.append(metrics.precision_score(val_target, y_hat))
            # outer_results_recall_m5.append(metrics.recall_score(val_target, y_hat))

            # # XGboost
            #
            model_xgb= XGBClassifier()
            gd_search_xgb = GridSearchCV(model_xgb, params_xgb, scoring='f1_macro', n_jobs=-1, cv=cv_inner).fit(train_data,
                                                                                                             train_target)
            best_model_xgb = gd_search_xgb.best_estimator_
            classifier_xgb = best_model_xgb.fit(train_data, train_target)
            y_hat_xgb = classifier_xgb.predict(val_data)
            y_pro_xgb = classifier_xgb.predict_proba(val_data)[:,1]
            y_pro_xgb2 = classifier_xgb.predict_proba(val_data)
            print(y_pro_xgb2)
            print(y_pro_xgb)


            ## save model

            nameModel = "Modelkfold{0}_{1}.joblib".format(str(k), str(j))
            fileModel = folder_figure + '/' + nameModel
            joblib.dump(classifier_xgb,fileModel)

            ## end save model
            outer_results_accuracy_xgb.append(metrics.accuracy_score(val_target, y_hat_xgb))
            outer_results_f1_xgb.append(metrics.f1_score(val_target, y_hat_xgb))
            outer_results_auc_xgb.append(metrics.roc_auc_score(val_target, y_hat_xgb))
            outer_results_precision_xgb.append(metrics.precision_score(val_target, y_hat_xgb))
            outer_results_recall_xgb.append(metrics.recall_score(val_target, y_hat_xgb))

            # namefilefexcel = "FeatImport_RF{0}_{1}.xlsx".format(str(k), str(j))
            # feature_importance_FILE_RF = folder_figure + '/' + namefilefexcel
            namefilefexcel = "FeatImport_XG{0}_{1}.xlsx".format(str(k), str(j))
            feature_importance_FILE_XG = folder_figure + '/' + namefilefexcel

            namefileOutput_RFXGB = "Output_RFXGB{0}_{1}.xlsx".format(str(k), str(j))
            output_FILE_RFXGB = folder_figure + '/' + namefileOutput_RFXGB

            print(output_FILE_RFXGB)

            # plt.clf()
            # plt.rcParams.update({'font.size': 10})
            # plt.figure()
            # plot_feature_importance(classifier.feature_importances_, train_data.columns, 'RANDOM FOREST',feature_importance_FILE_RF)
            # namefilef = "FeatImport_RF{0}_{1}.png".format(str(k),str(j))
            # feature_importance_RF = folder_figure + '/' + namefilef
            # plt.savefig(feature_importance_RF, bbox_inches='tight')
            # plt.clf()

            plt.clf()
            plt.rcParams.update({'font.size': 10})
            plt.figure()
            plot_feature_importance(classifier_xgb.feature_importances_, train_data.columns, 'XGBOOST',feature_importance_FILE_XG)
            namefilef = "FeatImport_XG{0}_{1}.png".format(str(k),str(j))
            feature_importance_XG = folder_figure + '/' + namefilef
            plt.savefig(feature_importance_XG, bbox_inches='tight')
            plt.clf()

            # Create arrays from feature importance and feature names
            # feature_groundtrue = np.array(val_target)
            # feature_prediction_xgb = np.array(y_hat)
            # feature_prediction_rf = np.array(y_hat)

            #
            # # Create a DataFrame using a Dictionary
            #datacl = {'feature_gt': feature_groundtrue, 'feature_xgb': feature_prediction_xgb, 'feature_rf': feature_prediction_rf}
            datacl = {'feature_gt': val_target, 'feature_xgb': y_hat_xgb, 'participants_id': participants_ID, 'probability': y_pro_xgb}
            fi_dfc  = pd.DataFrame(datacl)
            #print(fi_dfc)
            fi_dfc.to_csv(output_FILE_RFXGB, index=False)

           # fi_tg = pd.DataFrame(feature_groundtrue)
          #  fi_tg.to_csv(feature_importance_FILE_GT, index=False, header=None)

            j = j + 1

        auc_xgb.append(mean(outer_results_auc_xgb))
        f1_xgb.append(mean(outer_results_f1_xgb))
        precision_xgb.append(mean(outer_results_precision_xgb))
        recall_xgb.append(mean(outer_results_recall_xgb))
        accuracy_xgb.append(mean(outer_results_accuracy_xgb))

        auc_xgb_std.append(std(outer_results_auc_xgb))
        f1_xgb_std.append(std(outer_results_f1_xgb))
        precision_xgb_std.append(std(outer_results_precision_xgb))
        recall_xgb_std.append(std(outer_results_recall_xgb))
        accuracy_xgb_std.append(std(outer_results_accuracy_xgb))

        # auc_rf.append(mean(outer_results_auc_m5))
        # f1_rf.append(mean(outer_results_f1_m5))
        # precision_rf.append(mean(outer_results_precision_m5))
        # recall_rf.append(mean(outer_results_recall_m5))
        # accuracy_rf.append(mean(outer_results_accuracy_m5))
        #
        # auc_rf_std.append(std(outer_results_auc_m5))
        # f1_rf_std.append(std(outer_results_f1_m5))
        # precision_rf_std.append(std(outer_results_precision_m5))
        # recall_rf_std.append(std(outer_results_recall_m5))
        # accuracy_rf_std.append(std(outer_results_accuracy_m5))

        text_file_output.write("====> XGBOOST \n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_xgb), std(auc_xgb), mean(auc_xgb_std)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_xgb), std(f1_xgb), mean(f1_xgb_std)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_xgb), std(precision_xgb), mean(precision_xgb_std)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_xgb), std(recall_xgb), mean(recall_xgb_std)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_xgb), std(accuracy_xgb), mean(accuracy_xgb_std)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_xgb)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_xgb)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_xgb)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_xgb)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_xgb)))

        print("====> XGBOOST")
        print('AUC: %.2f (%.2f)' % (mean(auc_xgb), std(auc_xgb)))
        print('F1: %.2f (%.2f)' % (mean(f1_xgb), std(f1_xgb)))
        print('precision: %.2f (%.2f)' % (mean(precision_xgb), std(precision_xgb)))
        print('recall: %.2f (%.2f)' % (mean(recall_xgb), std(recall_xgb)))
        print('accuracy: %.2f (%.2f)' % (mean(accuracy_xgb), std(accuracy_xgb)))
        print("====> ")

        # text_file_output.write("====> RANDOM FOREST \n")
        # text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_rf), std(auc_rf), mean(auc_rf_std)))
        # text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_xgb), std(f1_xgb), mean(f1_rf_std)))
        # text_file_output.write(
        #     'precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_rf), std(precision_rf), mean(precision_rf_std)))
        # text_file_output.write(
        #     'recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_rf), std(recall_rf), mean(recall_rf_std)))
        # text_file_output.write(
        #     'accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_rf), std(accuracy_rf), mean(accuracy_rf_std)))
        # text_file_output.write("====> \n")
        # text_file_output.write('max AUC: %.2f \n' % (max(auc_rf)))
        # text_file_output.write('max F1: %.2f \n' % (max(f1_rf)))
        # text_file_output.write('max precision: %.2f \n' % (max(precision_rf)))
        # text_file_output.write('max recall: %.2f \n' % (max(recall_rf)))
        # text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_rf)))
    text_file_output.write(tac())
    text_file_output.close()
