import pandas as pd

input_datapath_0 = "../../nobu-data/s23-thesis-paper/0_lkernel_params_mutable.csv"
df0 = pd.read_csv(input_datapath_0, index_col=0)
input_datapath_1 = "../../nobu-data/s23-thesis-paper/1_lkernel_params_mutable.csv"
df1 = pd.read_csv(input_datapath_1, index_col=0)
print(df0.shape)
print(df1.shape)
frames = [df0, df1]
result = pd.concat(frames)
print(result.shape)
result.to_csv("../../nobu-data/s23-thesis-paper/lkernel_fxparams_use_labeled.csv")