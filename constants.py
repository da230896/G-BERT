BERT_IN_OUT = 300
BERT_HIDDEN = 300
BERT_NUM_LAYERS = 2
BERT_NUM_HEAD_EACH_LAYER = 4

PRETRAINING_FF_HIDDEN = 300
PRETRAINING_MAX_ATC_LEN = 101
PRETRAINING_MAX_ICD_LEN = 40
PRETRAINING_BATCH_SIZE = 100
PRETRAINING_EPOCH = 5
PRETRAINING_MODEL = "pretrain.pt"
PRE_TRAINING_LR = 0.001
PRE_TRAINING_THRESHOLD = 0.5

TRAINING_ITERATION = 15
TRAINING_EPOCH = 5
TRAINING_MODEL = "train.pt"
TRAINING_MAX_ATC_LEN = 93
TRAINING_MAX_ICD_LEN = 40
TRAINING_LR = 0.001
TRAINING_THRESHOLD = 0.5

GAT_CONV_IN_CHANNEL = 100
GAT_CONV_OUT_CHANNEL = 20
GAT_CONV_HEADS = 5
GAT_CONV_NEGATIVE_SLOP = 0.1
GAT_CONV_DROPOUT = 0.1

GLOBAL_DATA_PATH = "./Data"
GLOBAL_MODELS_PATH = "./Models"
ABLATIONS_PATH = "Ablations"
ABLATION_1 = "1"
ABLATION_2 = "2"


# MIMIC Files:
PRESCRIPTIONS = "PRESCRIPTIONS.csv"
ADMISSIONS = "ADMISSIONS.csv"
DIAGNOSIS_ICD="DIAGNOSES_ICD.csv"

# MIMIC Mapping Files
NDC_2_RXCUI_MAPPING = "ndc2rxnorm_mapping.txt"
RXCUI_2_ATC = "rxnorm2atc_level4.csv"
NDC_ATC_R_SCRIPT_MAPPING = "ndc_map_2022_04_05.csv"
NDC_ATC_MAPPING = "ndc_map.csv"

# Other File Names
UNIQUE_ATC_CSV= "unique-atc4.csv"
UNIQUE_ICD_CSV = "unique-icd.csv"
SINGLE_VISIT_PKL = "single_visit.pkl"
MULTI_VISIT_PKL = "multi_visit.pkl"
MULTI_VISIT_TEMPORAL_PKL = "multi_visit_temporal.pkl"