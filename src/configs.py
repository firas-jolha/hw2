from os.path import join as path_join

# Folders
MODELS_PATH = "models"
DATA_PATH = "data"
CODE_PATH = "src"
DOCS_PATH = "documentation"

# Files
NCF_MODEL_PATH = path_join(MODELS_PATH, "acf.pth")
NCF_MODEL_ONE_HOT_PATH = path_join(MODELS_PATH, "acf_oh.pth")
MOVIE_IDS_PATH = path_join(MODELS_PATH, "all_movies_indices.npy")
USER_IDS_PATH = path_join(MODELS_PATH, "all_users_indices.npy")
CONFIGS_PATH = path_join(MODELS_PATH, "configs.pkl")
P_ARRAY_PATH = path_join(MODELS_PATH, "P_ARRAY_CF.npy")
Q_ARRAY_PATH = path_join(MODELS_PATH, "Q_ARRAY_CF.npy")
R_TRAIN_MATRIX_PATH = path_join(MODELS_PATH, "R.npz")
R_TEST_MATRIX_PATH = path_join(MODELS_PATH, "R2.npz")
TRAIN_DATA_PATH = path_join(DATA_PATH, "train.csv")
TEST_DATA_PATH = path_join(DATA_PATH, "test.csv")


# For documentation Run
# pdoc --html src/*.py 
