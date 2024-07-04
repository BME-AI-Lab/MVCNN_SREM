import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import parse
from numba import jit
from sklearn.preprocessing import normalize
import pathlib
from tqdm import tqdm
import numpy as np


class MapGenerator:
    # CM is used as the default unit
    def __init__(self):
        self.start_position = np.array([0, 0])
        self.end_position = np.array([0.9, 1.96]) * 100  # cm
        self.distance_per_bin = 9.87 * 100 / 1536  # cm

        self.x_bins = int(self.end_position[0] / self.distance_per_bin)
        self.y_bins = int(self.end_position[1] / self.distance_per_bin)

        self.array_size = (self.x_bins, self.y_bins)

        # make a 3d array of (x_bins,ybins,(x,y positions) )
        xs = np.linspace(self.start_position[0], self.end_position[0], self.x_bins)
        ys = np.linspace(self.start_position[1], self.end_position[1], self.y_bins)
        self.xv, self.yv = np.meshgrid(xs, ys)

    def calculate(self, radar_position, data, data_distance):

        # calculate distance from radar to each point
        distance = np.sqrt(
            (self.xv - radar_position[0]) ** 2 + (self.yv - radar_position[1]) ** 2
        )

        # interpolate the data by distance and redistribute to the map

        interpolated_data = np.interp(distance, data_distance.flatten(), data.flatten())
        interpolated_data = interpolated_data.reshape((self.y_bins, self.x_bins))
        cos_threshold = np.cos(np.radians(25))

        if radar_position[0] == -20:
            angle = np.arctan2(self.xv - radar_position[0], self.yv - radar_position[1])
            interpolated_data = np.where(np.abs(np.cos(angle)) > cos_threshold, 0, interpolated_data)
        if radar_position[1] == 216:
            angle = np.arctan2(self.yv - radar_position[1], self.xv - radar_position[0])
            interpolated_data = np.where(np.abs(np.cos(angle)) > cos_threshold, 0, interpolated_data)
        
        return interpolated_data


def get_radar_map(files):
    radars = []
    radar_position = np.array([-20, 10])
    radar_fnames = glob.glob(files)

    radar_dict = {}
    count = 0
    for radar_fname in radar_fnames:
        radar_frames = pd.read_csv(radar_fname)
        
        # radar_frames = radar_frames - radar_frames.mean()
        radar_frame = normalize(abs(radar_frames), axis=1, norm='l2')
        time, radar_num = parse.parse("{}-radar{}.csv", radar_fname)
        radar_dict[radar_num] = radar_frames
        count += 1

    for i in range(1, count + 1):
        try:
            radars.append(radar_dict[str(i)])
        except:
            pass

    radars = [np.array(i)[:, :-1] for i in radars]
    distance_per_bin = 9.87 * 100 / 1536  # cm
    # total_distance = distance_per_bin * bins  # cm
    max_distance = 200  # cm
    #print(radars[0].shape)
    total_distance = distance_per_bin * radars[0].shape[1]  # cm
    bins = radars[0].shape[1]
    data_distances = np.linspace(0, total_distance, bins)
    generator = MapGenerator2()
    bed_y = 1.96 * 100
    bed_x = 0.92 * 100
    x_position = [
        (-20, bed_y / 2 + 30 * i) for i in range(-2, 2 + 1)
    ]

    y_position = [
        (
            bed_x / 2 + 15 * i,
            bed_y + 20,
        )
        for i in range(-1, 1 + 1)
    ]
    com_positions = np.concatenate([x_position, y_position])
    #print(dict(zip(range(8), com_positions)))
    null = np.zeros((generator.y_bins, generator.x_bins))
    count = 0
    

    all_map = []
    null = np.zeros((generator.y_bins, generator.x_bins))   

    for radar, radar_position in zip(radars, com_positions):
        #
            # for column in radar:
        column = radar[25]  # radar[:, column_index]
        
        radar_bins = column.shape[0]
        maps = generator.calculate(
            radar_position, column, data_distance=data_distances[:radar_bins]
        )
        all_map.append(np.abs(maps))
      
    return all_map