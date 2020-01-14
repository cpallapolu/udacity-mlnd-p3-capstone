
# from data.process_data import ProcessData


# processData = ProcessData()

# # processData.copy_from_v2()

# processData.load()
# # processData.fill_na_values()
# # processData.missing_data()
# # processData.remove_cols()

# # processData.normalize()


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.DataFrame({'col1': [i for i in range(0, 10)], 'col2': [i for i in range(10, 20)]})

print(df)

minMaxScaler = MinMaxScaler()
n_df = pd.DataFrame(minMaxScaler.fit_transform(df))
n_df.columns = df.columns
n_df.index = df.index

print(n_df)

print(minMaxScaler.inverse_transform([[10]]))
