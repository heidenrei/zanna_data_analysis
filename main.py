import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from collections import defaultdict

absolute_path = '/home/gavin/zanna_data_analysis/01_Experiment 1_2019-11-08_S1_T1DLC_resnet50_ORDER_leftarm v2Feb10shuffle1_750000.csv' 
trough_real_world_length = 9.5
height_of_ROI = 2.2
depth_of_trough = 1.3
trough_left_offset = 0.35
trough_right_offset = 0.5
sample_image = '/home/gavin/zanna_data_analysis/tmprom_zgx1.PNG'
p_cutoff = 0.9
FPS = 40.50
output_file_path = 'output_tester.csv'
reach_threshold = 0.6
retract_threshold_3_frame = 0.31
retract_threshold_5_frame = 0.4
path_to_dir_containing_dlc_outputs = './dlc_output_dir'

class ETL:
    def __init__(self, absolute_path, FPS, reach_threshold, retract_threshold_3_frame, retract_threshold_5_frame, trough_real_world_length, height_of_ROI, depth_of_trough, p_cutoff, output_file_path, trough_left_offset=0, trough_right_offset=0):
        self.df = pd.read_csv(absolute_path, header=[1,2], index_col=0)
        self.trough_real_world_length = trough_real_world_length
        self.height_of_ROI = height_of_ROI
        self.depth_of_trough = depth_of_trough
        self.trough_left_offset = trough_left_offset
        self.trough_right_offset = trough_right_offset
        self.FPS = FPS
        self.p_cutoff = p_cutoff
        self.filter_low_probas()
        self.reach_threshold = reach_threshold #in cm
        self.retract_threshold_3_frame = retract_threshold_3_frame
        self.retract_threshold_5_frame = retract_threshold_5_frame
        self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y = self.roi_corners()
        self.bottom_right_x, self.bottom_right_y, self.bottom_left_x, self.bottom_left_y = self.roi_bottom_corners()
        self.add_time_metrics()
        self.add_velocity_metrics()
        self.add_velocity_metrics_from_origin()
        self.smooth_reaches()
        self.add_changes_in_reach()
        self.print_head
        self.df.to_csv(output_file_path)

    def filter_low_probas(self):
        print('before: ')
        print(self.df.isna().sum())

        for i, row in self.df.iterrows():
            if self.df.loc[i]['paw']['likelihood'] < self.p_cutoff:
                self.df.loc[i]['paw']['x'] = np.nan
                self.df.loc[i]['paw']['y'] = np.nan

        for i, row in self.df.iterrows():
            if self.df.loc[i]['nose']['likelihood'] < self.p_cutoff:
                self.df.loc[i]['nose']['x'] = np.nan
                self.df.loc[i]['nose']['y'] = np.nan

        print('')
        print('after: ')
        print(self.df.isna().sum())


    def euc_dist(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        # new idea:
        # get vectors of the 4 borders of the roi and see if x product with the point is > 0
        # need top two points...
        # 

    def roi_corners(self):
        trough_r_x, trough_r_y, trough_l_x, trough_l_y = self.trough_coords
        xy_deg = math.degrees(math.atan2(trough_l_y - trough_r_y, trough_r_x - trough_r_y))
        angle = xy_deg - 90

        # need to get bottom corners of roi first
        hypotenuse = self.height_of_ROI*self.pixels_to_real

        dy = hypotenuse*math.sin(math.radians(angle))
        dx = math.sqrt(hypotenuse**2 - dy**2)

        if xy_deg > 0:
            dx *= -1

        top_left_x, top_left_y = trough_l_x + dx, trough_l_y +dy
        top_right_x, top_right_y = trough_r_x + dx, trough_r_y + dy
        return top_left_x, top_left_y, top_right_x, top_right_y
        
    def roi_bottom_corners(self):
        trough_r_x, trough_r_y, trough_l_x, trough_l_y = self.trough_coords
        xy_deg = math.degrees(math.atan2(trough_l_y - trough_r_y, trough_r_x - trough_r_y))
        angle = xy_deg - 90

        # need to get bottom corners of roi first
        hypotenuse = self.depth_of_trough*self.pixels_to_real

        dy = hypotenuse*math.sin(math.radians(angle))
        dx = math.sqrt(hypotenuse**2 - dy**2)

        if xy_deg < 0:
            dx *= -1

        bottom_left_x, bottom_left_y = trough_l_x - dx, trough_l_y - dy
        bottom_right_x, bottom_right_y = trough_r_x - dx, trough_r_y - dy
        return bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y

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

    def add_time_metrics(self):
        # self.df['nose_in_roi'] = self.in_roi(self.df['nose']['x'], self.df['nose']['y'])
        # self.df['nose_in_roi'] = self.df.apply(lambda v: 1 if self.in_roi(v.nose.x, v.nose.y) else 0)
        # self.df = self.df.assign(nose_in_roi=lambda v: (self.in_roi(v['nose']['x'], v['nose']['y'])))
        # self.df['paw_in_roi'] = self.in_roi(self.df['paw']['x'], self.df['paw']['y'])

        for i, row in self.df.iterrows():
            nose_in_roi_bool = True
            paw_in_roi_bool = True
            if not self.in_roi(self.df.loc[i]['nose']['x'], self.df.loc[i]['nose']['y']):
                nose_in_roi_bool = False
            if not self.in_roi(self.df.loc[i]['paw']['x'], self.df.loc[i]['paw']['y']):
                paw_in_roi_bool = False

            self.df.at[i, 'nose_in_roi'] = nose_in_roi_bool
            self.df.at[i, 'paw_in_roi'] = paw_in_roi_bool

        self.df['cumsum_paw_in_roi'] = (self.df['paw_in_roi'] == True).cumsum()
        self.df['cumsum_nose_in_roi'] = (self.df['nose_in_roi'] == True).cumsum()

        self.df['paw_in_roi_total_time'] = self.df['cumsum_paw_in_roi'] / self.FPS
        self.df['nose_in_roi_total_time'] = self.df['cumsum_nose_in_roi'] / self.FPS


        self.df['consecutive_frames_with_paw_in_roi'] = self.df['paw_in_roi'].groupby((self.df['paw_in_roi'] == False).cumsum()).cumcount()
        self.df['consecutive_frames_with_nose_in_roi'] = self.df['nose_in_roi'].groupby((self.df['nose_in_roi'] == False).cumsum()).cumcount()

        self.df['paw_in_roi_this_time'] = self.df['consecutive_frames_with_paw_in_roi'] / self.FPS
        self.df['nose_in_roi_this_time'] = self.df['consecutive_frames_with_nose_in_roi'] / self.FPS


    def add_velocity_metrics(self):
        cm_conversion = self.pixels_to_real
        for i, row in self.df.iterrows():
            if i > 0:
                paw_euc_dist = self.euc_dist(self.df.loc[i]['paw']['x'], self.df.loc[i]['paw']['y'], self.df.loc[i-1]['paw']['x'], self.df.loc[i-1]['paw']['y'])
                nose_euc_dist = self.euc_dist(self.df.loc[i]['nose']['x'], self.df.loc[i]['nose']['y'], self.df.loc[i-1]['nose']['x'], self.df.loc[i-1]['nose']['y'])
            else:
                paw_euc_dist = 0
                nose_euc_dist = 0

            self.df.at[i, 'paw_euc_dist_with_last_row'] = paw_euc_dist / cm_conversion
            self.df.at[i, 'nose_euc_dist_with_last_row'] = nose_euc_dist / cm_conversion

        self.df['paw_velocity_5_frame'] = self.df['paw_euc_dist_with_last_row'].rolling(5).sum()
        self.df['paw_velocity_10_frame'] = self.df['paw_euc_dist_with_last_row'].rolling(10).sum()

        self.df['nose_velocity_5_frame'] = self.df['nose_euc_dist_with_last_row'].rolling(5).sum()
        self.df['nose_velocity_10_frame'] = self.df['nose_euc_dist_with_last_row'].rolling(10).sum()

    # origin should probably be top right or bottom right... 1295, 0 is top right
    def add_velocity_metrics_from_origin(self):
        cm_conversion = self.pixels_to_real

        for i, row in self.df.iterrows():
            paw_euc_dist_from_origin = self.euc_dist(self.df.loc[i]['paw']['x'], self.df.loc[i]['paw']['y'], 1295, 0)
            nose_euc_dist_from_origin = self.euc_dist(self.df.loc[i]['nose']['x'], self.df.loc[i]['nose']['y'], 1295, 0)

            self.df.at[i, 'paw_euc_dist_from_origin'] = paw_euc_dist_from_origin / cm_conversion
            self.df.at[i, 'nose_euc_dist_from_origin'] = nose_euc_dist_from_origin / cm_conversion

        for i, row in self.df.iterrows():
            if i > 0:
                paw_euc_dist_d = self.df.loc[i]['paw_euc_dist_from_origin'] - self.df.loc[i-1]['paw_euc_dist_from_origin']
                nose_euc_dist_d = self.df.loc[i]['nose_euc_dist_from_origin'] - self.df.loc[i-1]['nose_euc_dist_from_origin']
                paw_euc_dist_d = paw_euc_dist_d[0]
                nose_euc_dist_d = nose_euc_dist_d[0]
            else:
                paw_euc_dist_d = 0
                nose_euc_dist_d = 0

            self.df.at[i, 'paw_euc_dist_from_origin_2_frame_d'] = paw_euc_dist_d
            self.df.at[i, 'nose_euc_dist_from_origin_2_frame_d'] = nose_euc_dist_d

        self.df['paw_euc_dist_from_origin_5_frame_d_sum'] = self.df['paw_euc_dist_from_origin_2_frame_d'].rolling(5).sum()
        self.df['paw_euc_dist_from_origin_10_frame_d_sum'] = self.df['paw_euc_dist_from_origin_2_frame_d'].rolling(10).sum()

        self.df['nose_euc_dist_from_origin_5_frame_d_sum'] = self.df['nose_euc_dist_from_origin_2_frame_d'].rolling(5).sum()
        self.df['nose_euc_dist_from_origin_10_frame_d_sum'] = self.df['nose_euc_dist_from_origin_2_frame_d'].rolling(10).sum()

        # count number of consecutive negative frames, and when there is a positive frame calculate (r[i-1]-r[i-cnt])
        # look back 3, 4, 5 frames and see if the different between r[i] - r[i-x] > threshold

        self.df['paw_diff_3_frame'] = self.df['paw_euc_dist_from_origin'].diff(periods=3)
        self.df['paw_diff_5_frame'] = self.df['paw_euc_dist_from_origin'].diff(periods=5)

        self.df['paw_reach_3_frame'] = (self.df['paw_diff_3_frame'] >= self.reach_threshold) & (self.df['paw_in_roi'] == True)
        self.df['paw_reach_5_frame'] = (self.df['paw_diff_5_frame'] >= self.reach_threshold) & (self.df['paw_in_roi'] == True)

        self.df['paw_retract_3_frame'] = (self.df['paw_diff_3_frame'] <= -1*(self.retract_threshold_3_frame)) & (self.df['paw_in_roi'] == True)
        self.df['paw_retract_5_frame'] = (self.df['paw_diff_5_frame'] <= -1*(self.retract_threshold_5_frame)) & (self.df['paw_in_roi'] == True)

    def add_changes_in_reach(self):
        for i, row in self.df.iterrows():
            if i > 0:
                change_in_reach_5_frame = True if (self.df.loc[i]['paw_reach_5_frame'][0] == True and self.df.loc[i-1]['paw_reach_5_frame'][0] == False) else False
            else:
                change_in_reach_5_frame = False
            self.df.at[i, 'change_in_reach_5_frame'] = change_in_reach_5_frame

        for i, row in self.df.iterrows():
            if i > 0:
                change_in_reach_3_frame = True if (self.df.loc[i]['paw_reach_3_frame'][0] == True and self.df.loc[i-1]['paw_reach_3_frame'][0] == False) else False
            else:
                change_in_reach_3_frame = False
            self.df.at[i, 'change_in_reach_3_frame'] = change_in_reach_3_frame

        for i, row in self.df.iterrows():
            if i > 0:
                change_in_retract_5_frame = True if (self.df.loc[i]['paw_retract_5_frame'][0] == True and self.df.loc[i-1]['paw_retract_5_frame'][0] == False) else False
            else:
                change_in_retract_5_frame = False
            self.df.at[i, 'change_in_retract_5_frame'] = change_in_retract_5_frame

        for i, row in self.df.iterrows():
            if i > 0:
                change_in_retract_3_frame = True if (self.df.loc[i]['paw_retract_3_frame'][0] == True and self.df.loc[i-1]['paw_retract_3_frame'][0] == False) else False
            else:
                change_in_retract_3_frame = False
            self.df.at[i, 'change_in_retract_3_frame'] = change_in_retract_3_frame

    # gets the coords of x given a y value
    def get_x_coords(self, y, intercept, angle):
        return (y/angle - intercept/angle)
        
    def smooth_reaches(self, reach_3=True, reach_5=True, retract_3=False, retract_5=False):
        print(self.df.shape)
        print('smoothing reaches')
        if reach_3:
            for i, row in self.df.iterrows():
                if i > 0 and i < self.df.shape[0] - 1:
                    if self.df.loc[i-1]['paw_reach_3_frame'][0] == True and self.df.loc[i+1]['paw_reach_3_frame'][0] == True and self.df.loc[i]['paw_reach_3_frame'][0] == False:
                        self.df.at[i, 'paw_reach_3_frame'] = True
        if reach_5:
            for i, row in self.df.iterrows():
                if i > 0 and i < self.df.shape[0] - 1:
                    if self.df.loc[i-1]['paw_reach_5_frame'][0] == True and self.df.loc[i+1]['paw_reach_5_frame'][0] == True and self.df.loc[i]['paw_reach_5_frame'][0] == False:
                        self.df.at[i, 'paw_reach_5_frame'] = True
       
        if retract_3:
            for i, row in self.df.iterrows():
                if i > 0 and i < self.df.shape[0] - 1:
                    if self.df.loc[i-1]['paw_retract_3_frame'][0] == True and self.df.loc[i+1]['paw_retract_3_frame'][0] == True and self.df.loc[i]['paw_retract_3_frame'][0] == False:
                        self.df.at[i, 'paw_retract_3_frame'] = True

        if retract_5:
            for i, row in self.df.iterrows():
                if i > 0 and i < self.df.shape[0] - 1:
                    if self.df.loc[i-1]['paw_retract_5_frame'][0] == True and self.df.loc[i+1]['paw_retract_5_frame'][0] == True and self.df.loc[i]['paw_retract_5_frame'][0] == False:
                        self.df.at[i, 'paw_retract_5_frame'] = True


    # returns the average point of the troughl and troughr labels
    @property
    def trough_coords(self):
        trough_r_x = self.df['troughR']['x'].mean()
        trough_r_y = self.df['troughR']['y'].mean()
        trough_l_x = self.df['troughL']['x'].mean()
        trough_l_y = self.df['troughL']['y'].mean()

        xy_deg = math.degrees(math.atan2(trough_l_y - trough_r_y, trough_r_x - trough_r_y))
        pix_to_real = self.euc_dist(trough_r_x, trough_r_y, trough_l_x, trough_l_y)/self.trough_real_world_length

        l_dy = math.sin(math.radians(xy_deg)) * (pix_to_real*self.trough_left_offset) # inverse sign of slope
        l_dx = math.sqrt(l_dy**2 + (pix_to_real*self.trough_left_offset)**2)

        r_dy = math.sin(math.radians(xy_deg)) * (pix_to_real*self.trough_right_offset) # same as sign of slope
        r_dx = math.sqrt(r_dy**2 + (pix_to_real*self.trough_right_offset)**2)

        return trough_r_x + r_dx, trough_r_y - r_dy, trough_l_x - l_dx, trough_l_y + l_dy

    @property
    def print_head(self):
        print(self.df.head())

    # ratio of pixels to real world cm
    @property
    def pixels_to_real(self):
        trough_r_x = self.df['troughR']['x'].mean()
        trough_r_y = self.df['troughR']['y'].mean()
        trough_l_x = self.df['troughL']['x'].mean()
        trough_l_y = self.df['troughL']['y'].mean()

        return self.euc_dist(trough_r_x, trough_r_y, trough_l_x, trough_l_y)/self.trough_real_world_length


class utils:
    def __init__(self, absolute_path, FPS, reach_threshold, retract_threshold_3_frame, retract_threshold_5_frame, trough_real_world_length, height_of_ROI, depth_of_trough, p_cutoff, output_file_path, trough_left_offset, trough_right_offset):
        self.etl = ETL(absolute_path, FPS, reach_threshold, retract_threshold_3_frame, retract_threshold_5_frame, trough_real_world_length, height_of_ROI, depth_of_trough, p_cutoff, output_file_path, trough_left_offset, trough_right_offset)
        self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y = self.etl.roi_corners()
        self.bottom_right_x, self.bottom_right_y, self.bottom_left_x, self.bottom_left_y = self.etl.roi_bottom_corners()

    # make a plot in ugly mcdonalds colors of where the roi is in the frame
    def plot_roi(self):
        xs = []
        for i in range(0, 1000, 5):
            for j in range(0, 1000, 5):
                xs.append([i, j, self.etl.in_roi(i, j)])

        x = np.asarray(xs)
        df = pd.DataFrame(x)
        plt.scatter(df[0], df[1], c=df[2], cmap=plt.cm.autumn)
        plt.show()

    # plot a polygon on a sample frame around where the roi is
    def outline_roi(self, input_frame):
        img = Image.open(input_frame)
        drawer = ImageDraw.Draw(img)
        drawer.polygon([self.bottom_right_x, self.bottom_right_y, self.bottom_left_x, self.bottom_left_y, self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y])
        img.show()

    # x product
    def orientation(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p3
        x3, y3 = p2

        return (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2)

    def get_convex_hull(self):
        df = self.etl.df.copy(deep=True)['paw']
        indices = df[ df['likelihood'] < p_cutoff].index
        df.drop(indices, inplace=True)
        df.drop(columns=['likelihood'], inplace=True)
        coords = df.values.tolist()

        N = len(coords)
        start = point = coords.index(min(coords, key = lambda x: x[0]))
        hull = [start]

        far_point = None

        while far_point is not start:

            p1 = None
            for i in range(N):
                if i is point:
                    continue
                else:
                    p1 = i
                    break
            far_point = p1

            for j in range(N):
                if j is point or j is p1:
                    continue
                else:
                    direction = self.orientation(coords[point], coords[far_point], coords[j])
                    if direction > 0:
                        far_point = j

            if far_point == start:
                break
            
            hull.append(far_point)
            point = far_point

        return [coords[x] for x in hull]

    # plot the convex hull of the points where the paw was labeled > p_cuttoff
    def outline_hull(self, input_frame):
        hull = self.get_convex_hull()
        hull = [tuple(x) for x in hull]
        img = Image.open(input_frame)
        drawer = ImageDraw.Draw(img)
        drawer.polygon(hull)
        img.show()


def main():
    for fn in os.listdir(path_to_dir_containing_dlc_outputs):
        if fn[:3] != 'out' and fn[:3] != 'mas' and fn[0] != '.':
            output_file_path = path_to_dir_containing_dlc_outputs + os.sep + 'output_' + fn
            absolute_path = path_to_dir_containing_dlc_outputs + os.sep + fn
            print(absolute_path)
            etl = ETL(absolute_path, FPS, reach_threshold, retract_threshold_3_frame, retract_threshold_5_frame, trough_real_world_length, height_of_ROI, depth_of_trough, p_cutoff, output_file_path, trough_left_offset, trough_right_offset)
    master_df = pd.DataFrame(columns=['video_id', 'reaches_frame_459_690', 'tot_frame_459_690', 'path_length_frame_459_690', 'reaches_minute_total', 'tot_minute_total', 'nose_tot_minute_total', 'path_length_minute_total', 'reaches_session_total', 'tot_session_total', 'path_length_session_total'])
    master_df.to_csv(path_to_dir_containing_dlc_outputs + os.sep + 'master_df.csv')
    for fn in os.listdir(path_to_dir_containing_dlc_outputs):
        if fn[:3] == 'out':
            animal_id = fn.split('_')[2]
            video_id = fn[7:-4]
            df = pd.read_csv(path_to_dir_containing_dlc_outputs + os.sep + fn)
            reaches_frame_459_690 = df['change_in_reach_3_frame'].iloc[459:690].sum()
            tot_frame_459_690 = df['paw_in_roi'].loc[459:690].sum() * (1 / FPS)
            path_length_frame_459_690 = df['paw_euc_dist_with_last_row'].loc[459:690].sum()
            reaches_minute_total = df['change_in_reach_3_frame'].sum()
            tot_minute_total = df['paw_in_roi'].sum() * (1 / FPS)
            path_length_minute_total = df['paw_euc_dist_with_last_row'].sum()
            nose_tot_minute_total = df['nose_in_roi'].sum() * (1 / FPS)
            data = {'video_id': [video_id], 'reaches_frame_459_690': [reaches_frame_459_690], 'tot_frame_459_690': [tot_frame_459_690], 'path_length_frame_459_690': [path_length_frame_459_690], 'reaches_minute_total': [reaches_minute_total], 'tot_minute_total': [tot_minute_total], 'nose_tot_minute_total': [nose_tot_minute_total], 'path_length_minute_total': [path_length_minute_total], 'reaches_session_total': [np.NaN], 'tot_session_total': [np.NaN], 'path_length_session_total': [np.NaN]}
            tmp_df = pd.DataFrame(data, columns=['video_id', 'reaches_frame_459_690', 'tot_frame_459_690', 'path_length_frame_459_690', 'reaches_minute_total', 'tot_minute_total', 'nose_tot_minute_total', 'path_length_minute_total', 'reaches_session_total', 'tot_session_total', 'path_length_session_total'])
            master_df = pd.read_csv(path_to_dir_containing_dlc_outputs + os.sep + 'master_df.csv', index_col=0)
            output_df = pd.concat([master_df, tmp_df])
            output_df.to_csv(path_to_dir_containing_dlc_outputs + os.sep + 'master_df.csv')
            
    df = pd.read_csv(path_to_dir_containing_dlc_outputs + os.sep + 'master_df.csv', index_col=0)
    sort_values = []
    animal_ids = []
    reaches_session_totals = defaultdict(int)
    tot_session_totals = defaultdict(int)
    path_length_session_totals = defaultdict(int)
    reaches_session_totals_l = []
    tot_session_totals_l = []
    path_length_session_totals_l = []

    for i, row in df.iterrows():
        animal_id = row[0].split('_')[1]
        reaches_session_totals[animal_id] += row['reaches_minute_total']
        tot_session_totals[animal_id] += row['tot_minute_total']
        path_length_session_totals[animal_id] += row['path_length_minute_total']

        sort_value = animal_id + '_' + row[0].split('_')[0]
        sort_values.append(sort_value)
        animal_ids.append(animal_id)
        

    df['sort_values'] = sort_values
    df['animal_id'] = animal_ids
    
    for i, row in df.iterrows():
        reaches_session_totals_l.append(reaches_session_totals[row['animal_id']])
        tot_session_totals_l.append(tot_session_totals[row['animal_id']])
        path_length_session_totals_l.append(path_length_session_totals[row['animal_id']])
        
    df['reaches_session_total'] = reaches_session_totals_l
    df['tot_session_total'] = tot_session_totals_l
    df['path_length_session_total'] = path_length_session_totals_l
    
    df.sort_values(by='sort_values', inplace=True)
    
    
    df.to_csv(path_to_dir_containing_dlc_outputs + os.sep + 'master_df.csv')
     
    # etl = ETL(absolute_path, FPS, reach_threshold, retract_threshold_3_frame, retract_threshold_5_frame, trough_real_world_length, height_of_ROI, depth_of_trough, p_cutoff, output_file_path, trough_left_offset, trough_right_offset)
if __name__ == "__main__":
    DEBUG = False

    if DEBUG:
        ut = utils(absolute_path, FPS, reach_threshold, retract_threshold_3_frame, retract_threshold_5_frame, trough_real_world_length, height_of_ROI, depth_of_trough, p_cutoff, output_file_path, trough_left_offset, trough_right_offset)
        ut.outline_hull(sample_image)
        ut.outline_roi(sample_image)
        #ut.plot_roi()

    else:
        main()
