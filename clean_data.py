import pandas as pd
import tqdm

input_datapath = "../../nobu-data/s23-thesis-paper/labeled_body_shuffled_ds_1.csv" 
df = pd.read_csv(input_datapath, index_col=0)
print(df.head())
target_params = []
param_index = []

#cleaning to isolate the param entry as its own (should work)
with tqdm.tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        pbar.update(1)
        for i in range(10):
            current_param_entry = row["a" + str(i)]
            if current_param_entry != "ignore" and current_param_entry != "u":
                target_params.append(current_param_entry)
                param_index.append(i)
#this should now contain target_param as the parameter we're looking at
df['target_param'] = target_params
df['parameter_index'] = param_index
#remove misc columns 
for i in range(10):
    df = df.drop(["a" + str(i)], axis=1)

#drop in_macro
df = df.drop(['in_macro'], axis=1)

#rename column titles 
df = df.rename(columns={"file": "filename", "label_time": "label_duration", "immutable":"not_written_to" })

print(df.head())

df.to_csv("../../nobu-data/s23-thesis-paper/1_lkernel_params_mutable.csv")