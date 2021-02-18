import pandas as pd
import numpy as np

absolute_path = '/home/gavin/zanna_data_analysis/01_Experiment 1_2019-11-08_S1_T1DLC_resnet50_ORDER_leftarm v2Feb10shuffle1_750000.csv'
class ETL:
    def __init__(self, absolute_path):
        self.df = pd.read_csv(absolute_path, header=[1,2], index_col=0)
        self.get_roi()



    def get_roi(self):
        trough_r_x = self.df['troughR']['x'].mean()
        trough_r_y = self.df['troughR']['y'].mean()
        trough_l_x = self.df['troughL']['x'].mean()
        trough_l_y = self.df['troughL']['y'].mean()
        print(trough_r_x, trough_r_y)

    @property
    def print_head(self):
        print(self.df.head())


def main():
    etl = ETL(absolute_path)
    # etl.print_head

if __name__ == "__main__":
    DEBUG = False

    if DEBUG:
        pass
    else:
        main()