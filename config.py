NONCOVID_XGB_TRAIN_DATA_LOC = ""
NONCOVID_XGB_TEST_DATA_LOC = ""
COVID_XGB_DATA_LOC = ""

NONCOVID_RNN_TRAIN_DATA_LOC = ""
NONCOVID_RNN_TEST_DATA_LOC = ""
COVID_RNN_DATA_LOC = ""

NONCOVID_CV_SPLITS_LOC = ""
COVID_CV_SPLITS_LOC = ""

CV_N_FOLDS = 5
CV_N_REPEATS = 5
EPS = 6e-3
N_CI_RUNS = 100
SCALE_POS_WEIGHT = 1.4444444

RNN_OPTIMIZER = "adam"
RNN_LOSS = "mse"
RNN_N_FEATURES = 11
RNN_N_TIMESTEPS = 1440
RNN_LSTM_SIZE = 16
RNN_DENSE_SIZE = 4
RNN_EPOCHS_MAIN = 15
RNN_EPOCHS_FINETUNE = 5
RNN_BATCH_SIZE = 16

FEATURES_FORCE_SELECT_APACHE = ["|Vital| Temp mon *min", "|Vital| Temp mon *max", "|Vital| Temp man *min", "|Vital| Temp man *max", "|Vital| Art BP mit *min", "|Vital| Art BP mit *max", "|Vital| HF *min", "|Vital| HF *max", "|Vital| AF mon *min", "|Vital| AF mon *max", "|Respiratory| AF ges. /min *min", "|Respiratory| AF ges. /min *max", "|BGA| (venös) PaO² *min", "|BGA| (arteriell) PaO² *min", "|BGA| (venös) PaCO² *min", "|BGA| (arteriell) PaCO² *min", "|Respiratory| FiO² %% *min", "|Respiratory| FiO² %% *max", "|BGA| (arteriell) PH *min", "|BGA| (arteriell) PH *max", "|BGA| (venös) HC03std *min", "|BGA| (venös) HC03std *max", "|Labor| (Klin Chem) Natrium (Serum) *min", "|Labor| (Klin Chem) Natrium (Serum) *max", "|Labor| (Klin Chem) Kalium (Serum) *min", "|Labor| (Klin Chem) Kalium (Serum) *max", "|Labor| (Klin Chem) Kreatinin (Serum) *min", "|Labor| (Klin Chem) Kreatinin (Serum) *max", "|Labor| (Kleinem Bb) Hämatokrit *min", "|Labor| (Kleinem Bb) Hämatokrit *max", "|Labor| (Kleinem Bb) Leukozyten (kl. Bb) *min", "|Labor| (Kleinem Bb) Leukozyten (kl. Bb) *max", "|Metadaten| ALTER"]
FEATURES_FORCE_SELECT_SOFA = ["|BGA| (venös) PaO² *min", "|BGA| (arteriell) PaO² *min", "|BGA| (venös) PaCO² *min", "|BGA| (arteriell) PaCO² *min", "|Respiratory| FiO² %% *min", "|Respiratory| FiO² %% *max", "|Vital| Art BP mit *min", "|Vital| Art BP mit *max", "|Labor| (Klin Chem) Bilirubin (direkt) *min", "|Labor| (Klin Chem) Bilirubin (direkt) *max", "|Labor| (Kleinem Bb) Thrombozyten *min", "|Labor| (Kleinem Bb) Thrombozyten *max", "|Labor| (Klin Chem) Kreatinin (Serum) *min", "|Labor| (Klin Chem) Kreatinin (Serum) *max", "|Diurese| Urine / weight *sum", "|Vital| NiBP sys *min", "|Vital| NiBP sys *max", "|Vital| Art BP sys *min", "|Vital| Art BP sys *max", "|Respiratory| AF ges. /min *min", "|Respiratory| AF ges. /min *max"]
FEATURES_FORCE_SELECT_ALL = list(set(FEATURES_FORCE_SELECT_APACHE + FEATURES_FORCE_SELECT_SOFA))
FEATURES_SELECTED_NONCOVID = []

DEFAULT_XGB_PARAMS = {
    "eval_metric": "auc",
    "objective": "binary:logistic",
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "silent": 0,
    "nthread": -1,
    "learning_rate": 0.1
}

INITIAL_GRID_PARAMS = {
    "n_estimators": [60, 80, 100],

    "max_depth": [2, 4, 6, 8],
    "min_child_weight": [2, 4, 6, 8],
    "gamma": [0, 1, 2],
    "subsample": [0.9, 1],
    "colsample_bytree": [0.9, 1],

    "objective": ["binary:logistic"],
    "metric": ["auc"],
    "silent": [0],
    "nthread": [-1],

    "learning_rate": [0.1],
    "scale_pos_weight": [SCALE_POS_WEIGHT],
}
