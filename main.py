import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

absolute_path = '/home/gavin/zanna_data_analysis/01_Experiment 1_2019-11-08_S1_T1DLC_resnet50_ORDER_leftarm v2Feb10shuffle1_750000.csv'
trough_real_world_length = 9.5
height_of_ROI = 2.2
sample_image = r'/home/gavin/Downloads/01_Experiment 1_2019-11-08_S1_T1.png'

class ETL:
    def __init__(self, absolute_path, trough_real_world_length, height_of_ROI):
        self.df = pd.read_csv(absolute_path, header=[1,2], index_col=0)
        self.trough_real_world_length = trough_real_world_length
        self.height_of_ROI = height_of_ROI
        self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y = self.roi_corners()
        self.bottom_right_x, self.bottom_right_y, self.bottom_left_x, self.bottom_left_y = self.trough_coords

    def euc_dist(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        # new idea:
        # get vectors of the 4 borders of the roi and see if x product with the point is > 0
        # need top two points...
        # 

    def roi_corners(self):
        trough_r_x, trough_r_y, trough_l_x, trough_l_y = self.trough_coords
        xy_deg = math.degrees(math.atan2(trough_l_y - trough_r_y, trough_r_x - trough_r_y))
        angle = 90 - xy_deg

        hypotenuse = self.height_of_ROI*self.pixels_to_real

        dy = hypotenuse*math.cos(angle)
        dx = math.sqrt(hypotenuse**2 - dy**2)

        top_left_x, top_left_y = trough_l_x + dx, trough_l_y +dy
        top_right_x, top_right_y = trough_r_x + dx, trough_r_y + dy
        return top_left_x, top_left_y, top_right_x, top_right_y

    # uses x product to check if given coordinate is inside the ROI..
    def in_roi(self, x, y) -> bool:
        top, right, bottom, left = False, False, False, False

        # check x products clockwise (top, right, bottom, left)
        # let a = the vector between the two points making our boundary
        # let b = the vector between one point in our boundary and the given coordinates
        # if a x b > 0 condition = True else False
        # return all([conditions])

        a = [self.top_left_x - self.top_right_x, self.top_left_y - self.top_right_y]
        b = [self.top_right_x - x, self.top_right_y - y]

        if a[0]*b[1] - a[1]*b[0] >= 0:
            top = True

        c = [self.top_right_x - self.bottom_right_x, self.top_right_y - self.bottom_right_y]
        d = [self.bottom_right_x - x, self.bottom_right_y - y]

        if c[0]*d[1] - c[1]*d[0] >= 0:
            right = True

        e = [self.bottom_right_x - self.bottom_left_x, self.bottom_right_y - self.bottom_left_y]
        f = [self.bottom_left_x - x, self.bottom_left_y - y]

        if e[0]*f[1] - e[1]*f[0] >= 0:
            bottom = True
        
        g = [self.bottom_left_x - self.top_left_x, self.bottom_left_y - self.top_left_y]
        h = [self.top_left_x - x, self.top_left_y - y]

        if g[0]*h[1] - g[1]*h[0] >= 0:
            left = True

        return all([top, right, bottom, left])

    # gets the coords of x given a y value
    def get_x_coords(self, y, intercept, angle):
        return (y/angle - intercept/angle)

    # returns the average point of the troughl and troughr labels
    @property
    def trough_coords(self):
        trough_r_x = self.df['troughR']['x'].mean()
        trough_r_y = self.df['troughR']['y'].mean()
        trough_l_x = self.df['troughL']['x'].mean()
        trough_l_y = self.df['troughL']['y'].mean()

        return trough_r_x, trough_r_y, trough_l_x, trough_l_y

    @property
    def print_head(self):
        print(self.df.head())

    # ratio of pixels to real world cm
    @property
    def pixels_to_real(self):
        trough_r_x, trough_r_y, trough_l_x, trough_l_y = self.trough_coords
        return self.euc_dist(trough_r_x, trough_r_y, trough_l_x, trough_l_y)/self.trough_real_world_length


class utils:
    def __init__(self, absolute_path, trough_real_world_length, height_of_ROI):
        self.etl = ETL(absolute_path, trough_real_world_length, height_of_ROI)
        self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y = self.etl.roi_corners()
        self.bottom_right_x, self.bottom_right_y, self.bottom_left_x, self.bottom_left_y = self.etl.trough_coords

    def plot_roi(self):
        xs = []
        for i in range(0, 1000, 5):
            for j in range(0, 1000, 5):
                xs.append([i, j, self.etl.in_roi(i, j)])

        x = np.asarray(xs)
        df = pd.DataFrame(x)
        plt.scatter(df[0], df[1], c=df[2], cmap=plt.cm.autumn)
        plt.show()

    def outline_roi(self, input_frame):
        img = Image.open(input_frame)
        drawer = ImageDraw.Draw(img)
        print('1')
        drawer.polygon([self.bottom_right_x, self.bottom_right_y, self.bottom_left_x, self.bottom_left_y, self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y])
        img.show()

    def get_convex_hull(self):
        df = self.etl.df['paw']
        df.drop(columns=['likelihood'], inplace=True)
        coords = df.values.tolist()
        
        start = min(coords, key = lambda x: x[0])

        


def main():
    etl = ETL(absolute_path, trough_real_world_length, height_of_ROI)


if __name__ == "__main__":
    DEBUG = True

    if DEBUG:
        ut = utils(absolute_path, trough_real_world_length, height_of_ROI)
        # ut.plot_roi()
        # ut.outline_roi(sample_image)
        ut.get_convex_hull()

    else:
        main()