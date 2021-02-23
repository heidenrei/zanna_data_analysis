import pandas_alive
import pandas as pd
import matplotlib.pyplot as plt

input_file = '/home/gavin/zanna_data_analysis/output1.csv'

df = pd.read_csv(input_file)

# df['paw_euc_dist_from_origin'].head(500).fillna(0).plot_animated(filename='test.gif', kind="line", period_fmt="%Y", period_length=200,fixed_max=True)

ax = df['paw_euc_dist_from_origin'].dropna().plot(kind='line')
plt.show()