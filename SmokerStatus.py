from My_Libraries import pd

from Preprocessing import process_data, data_preprocessing


df_train = pd.read_csv('Data/train.csv')
df_train = df_train.drop('id',axis=1)
df_test = pd.read_csv('Data/test.csv')
df_test = df_test.drop('id',axis=1)
df_orig = pd.read_csv('Data/train_dataset.csv')
df = pd.concat([df_train, df_orig], ignore_index=True)
df = df.drop_duplicates()

temp = process_data(df)
y = temp['smoking'].to_frame()
temp = temp.drop('smoking', axis=1)
X = data_preprocessing(temp)

temp = process_data(df_test)
X_test = data_preprocessing(temp)

X_train = X.copy()
y_train = y.copy()

X_train.reset_index(drop='index', inplace=True)
y_train.reset_index(drop='index', inplace=True)
X_test.reset_index(drop='index', inplace=True)

scale_pos_weight = len(y[y['smoking'] == 0]) / len(y[y['smoking'] == 1])
