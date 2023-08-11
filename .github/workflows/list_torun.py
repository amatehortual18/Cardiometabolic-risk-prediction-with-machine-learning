# This file learns the XGBoost model 


import function_predictionV


# The list of files with features that will be used to learn the model. The code will return a model for each file.
# The .csv file has to contain headers: ID (the participant identification) and class (the labels per class, 1 for disease, and 0 for control) 

names_files=[
            "DIABETESMellitus_Allinformation_train_combo1.csv",
            "DIABETESMellitus_Allinformation_val_combo1.csv"
        ]


path_parent="/Users/angelicaatehortua/Documents/posdoctorado/UKBIOBANK-TEST/Paper2/scripts/"

# The folder in which the results will be saved
output_folder= "/Users/angelicaatehortua/Documents/posdoctorado/UKBIOBANK-TEST/Paper2/scripts/ANGELICA/out_cvd_allfeatures/"

# information to perform the nested cross-validation strategy, the number of inner (n_folds_in) and outer (n_folds_out) folds. The process can be repeated several times (n_repetitions)
# fold_type 2 corresponds to a StratifiedKFold. 


n_repetitions=1
fold_type_out=2
fold_type_inn=2
n_folds_out=5
n_folds_in=3

for k in range(0,len(names_files)):
    print(names_files[k])
    function_predictionV.run_prediction(path_parent, names_files[k], output_folder, n_repetitions, fold_type_out, fold_type_inn, n_folds_out,n_folds_in)
