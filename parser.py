import pandas as pd
import numpy as np

df_u = pd.read_csv('SferaStatic/0_3M/U.csv', header=None, names=['U1', 'U2', 'U3'], float_precision='round_trip')

df_c = pd.read_csv('SferaStatic/0_3M/C.csv', header=None, names=['C1', 'C2', 'C3'], float_precision='round_trip')

combined_array = np.concatenate((df_u.values, df_c.values), axis=1)

df_combined = pd.DataFrame(combined_array, columns=['U1', 'U2', 'U3', 'C1', 'C2', 'C3'])

df_p = pd.read_csv('SferaStatic/0_3M/P.csv', header=None, names=['P'], float_precision='round_trip')

df_combined['P'] = df_p.values

df_combined.to_csv('SferaStatic/0_3M/input.csv', index=False, float_format='%.12f')
