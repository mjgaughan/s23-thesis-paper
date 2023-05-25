import pandas as pd

'''
input_datapath_0 = "../../nobu-data/s23-thesis-paper/0_lkernel_params_mutable.csv"
input_datapath_1 = "../../nobu-data/s23-thesis-paper/1_lkernel_params_mutable.csv"
result.to_csv("../../nobu-data/s23-thesis-paper/lkernel_fxparams_use_labeled.csv")
'''
def concat_two_csv(data_filepath_0, data_filepath_1, data_filepath_out):
    df0 = pd.read_csv(data_filepath_0, index_col=0)
    df1 = pd.read_csv(data_filepath_1, index_col=0)
    print(df0.shape)
    print(df1.shape)
    frames = [df0, df1]
    result = pd.concat(frames)
    print(result.shape)
    result.to_csv(data_filepath_out)