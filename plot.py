#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
folder = "./experiment_data/deeplab_multitask_naive"
#%%
versions = sorted(os.listdir(folder))
versions
#%%
train_val_data_path = os.path.join(folder, versions[0], "results", "train_val_data.csv")
train_val_data = pd.read_csv(train_val_data_path, index_col="epoch")
train_val_metrics = list(train_val_data.columns)
#%%
train_val_dict = {metric: {} for metric in train_val_metrics}
test_dataframes = []
for version in versions:
    # CHANGE TO YOUR NEED
    alpha = version.split("_")[-1]
    # name = f"naive {version.split('_')[-1]}"
    train_val_data_path = os.path.join(folder, version, "results", "train_val_data.csv")
    train_val_data = pd.read_csv(train_val_data_path, index_col="epoch")
    for metric in train_val_metrics:
        train_val_dict[metric][alpha] = list(train_val_data[metric])

    # TEST DATA
    test_data_path = os.path.join(folder, version, "results", "test_data.csv")
    test_data = pd.read_csv(test_data_path, index_col="epoch")
    test_data["alpha"] = alpha
    test_dataframes.append(test_data)


#%%
for metric in train_val_metrics:
    train_df = pd.DataFrame(train_val_dict[metric])
    train_df.index.name = "epoch"
    train_df.columns.name = "alphas"
    train_df.plot(figsize=(10, 6))
    plt.title(metric)
    plt.savefig(f"{folder}/{metric}.png", bbox_inches="tight")

#%%
test_df = pd.concat(test_dataframes)
test_df.reset_index(inplace=True)
test_df.set_index("alpha", inplace=True)
test_df.to_csv(f"{folder}/test_data.csv")
test_df
#%%
