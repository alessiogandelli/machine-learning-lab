#%% import
import numpy as np
import pandas as pd
import seaborn as sns
import statistics
import math


#functions
def distance(flower1, flower2):
    dist_sl = (flower1.sepal_length - flower2.sepal_length) ** 2
    dist_sw = (flower1.sepal_width - flower2.sepal_width) ** 2
    dist_pl = (flower1.petal_length - flower2.petal_length) ** 2
    dist_pw = (flower1.petal_width - flower2.petal_width) ** 2
    return math.sqrt(sum([dist_sl, dist_sw, dist_pl, dist_pw]))


def compute_knn(k, training_set, test_flower):
    distances = []
    for flo in training_set.iterrows():
        distances.append((distance(flo[1], test_flower), flo[1].species))
    
    
    return sorted(distances, key=lambda t: t[0])[:k]


# %% import dataset 

iris = sns.load_dataset("iris")
iris.to_csv("iris.csv")
df = pd.read_csv("iris.csv")

# normalize values 
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
for col in cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

dataset_size = df.shape[0]
# %%

# %% compute training and test set 

# random indexis 
idxs = np.arange(0, dataset_size)  
np.random.shuffle(idxs) 

training_slice = 0.7 # 70% training set and 30% test set 
break_point = int(training_slice * dataset_size)

training_idx = idxs[:break_point]
test_idx = idxs[break_point:]

training_set = df.iloc[training_idx]
test_set = df.iloc[test_idx]

# %% predict
preds = [compute_knn(3,training_set, test_flower[1]) for test_flower in test_set.iterrows()]

# %%

# %%
