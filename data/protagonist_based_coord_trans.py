from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset

class ProtagonistBasedCoordTrans:
    """
    This data model is generated based on reconstruction.
    Transform the coordinate to protagonist-based coordinate.
    Data shape: xycoord x total_joints x frames
    """
    def __init__(self, 
                 data: str,
                 data_path: str,
                 protagonist: int,
                 bodypart: List[str],
                 zone_radius: float,
                 mice_num: int,
                 frame_len: int,
                 **kwargs) -> None:
        self.datadir = data
        self.dataset = Path(data).parents[0].stem
        self.datapath = data_path
        self.prog = protagonist
        self.bodypart = bodypart
        self.joints_num = len(bodypart)
        self.mice_num = mice_num
        self.threshold = zone_radius
        self.frame_len = frame_len
        self.coord = 2
        self.image_size = [400,300]

        # check if the processed data exists
        self.dataname = Path(self.datapath)/(f"{self.dataset}-id{protagonist}-{self.threshold}px.csv")
        if self.dataname.exists():
            self.data = pd.read_csv(self.dataname,
                                    header=[0,1,2],
                                    index_col=0,
                                    dtype=np.float32)
        else:
            self.transform()
        
    def transform(self):
        # concatenate all csv files chronologically
        print("Loading data ...")
        
        csv_files = sorted(list(Path(self.datadir).glob("*preprocessed.csv")))
        csv = [pd.read_csv(c, header=[0,1,2], index_col=0, dtype=np.float32) for c in csv_files]
        self.data = pd.concat(csv, ignore_index=True)
        assert self.data.shape[1] == 24

        # transform coordinate to protagonist view point
        print("Transforming data ...")
        self.data = self.coord_transform_protagonist_marker2nose()
        
        # filter out mouse who is far from a distance to reduce noise
        self.data = self.distance_filter()

        # normailization
        self.data = self.normailization()

        # filter out 

        # save the data
        print("Saving data ...")
        self.data.to_csv(self.dataname)

    def train(self):
        data = torch.tensor(self.data.values.reshape(-1,
                                                     self.mice_num*self.joints_num,
                                                     self.coord))

        # Split data via specific length of frame and discard last frame
        # in case of nonequal length.
        data = torch.stack(data.split(self.frame_len)[:-1])
        return TensorDataset(data.transpose(1,3), torch.zeros(len(data)))
    
    def predict(self):
        data = torch.tensor(self.data.values.reshape(-1,
                                                     self.mice_num*self.joints_num,
                                                     self.coord))

        # Split data via specific length of frame and discard last frame
        # in case of nonequal length.
        data = torch.stack(data.split(self.frame_len)[:-1])
        return TensorDataset(data.transpose(1,3), torch.zeros(len(data)))
        
    def coord_transform_protagonist_marker2nose(self,) -> pd.DataFrame:
        """
        This function will transform the orignial screen coordinate(orignial point at top left)
        to a moving coordinate that its orignial point locate at protagonist's marker and its nose
        is where the direction of y axis orientate.
        More specifically, this function move the orignial point to the marker and then rotate its
        direction align with the vector of marker-to-nose.
        """
        col_marker = self.data.columns.isin([(f'individual{self.prog}','marker')])

        # translate orignal coordinate to protagonist's marker
        prog_marker = np.tile(self.data.loc[:, col_marker].values,
                              self.joints_num*self.mice_num)
        translated = self.data - prog_marker

        # rotate coordinate for align the nose to y axis
        col_nose_x = translated.columns.isin([(f'individual{self.prog}','nose', 'x')])
        col_nose_y = translated.columns.isin([(f'individual{self.prog}','nose', 'y')])
        theta = np.arctan(-translated.loc[:,col_nose_x].values/(translated.loc[:,col_nose_y].values+0.0001))
        rot_fun = np.vectorize(self.rotate, signature="(n,m),()->(n,m)")
        result = rot_fun(translated.to_numpy().reshape(-1, self.joints_num*self.mice_num, 2),
                         np.squeeze(theta))
        
        col = self.data.columns
        df = pd.DataFrame(result.reshape(-1, 2*(self.joints_num*self.mice_num)),
                          columns=col)
        
        # set the protagonist's marker as original coordinate
        df.loc[:, col_marker] = self.data.loc[:, col_marker]

        return df

    @staticmethod
    def rotate(xy: np.array, theta: np.array) -> np.array:
        """
        Function used for rotation during coordinate transformation.
        """
        rot = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
        result = rot@xy.T
        return result.T

    def distance_filter(self):
        """
        Filter out other mice when they fall out of social distance(threshold)
        """
        distance = self.data.pow(2).T.groupby(level=(0, 1)).sum().pow(0.5)
        df = self.data.where(distance.T.lt(self.threshold), 0.0)

        # refill the marker of protagonist by original data
        col_marker = self.data.columns.isin([(f'individual{self.prog}','marker')])
        df.loc[:, col_marker] = self.data.loc[:, col_marker]

        # filter out noninteractive
        col_prog = self.data.columns.get_level_values('individuals').isin([f'individual{self.prog}'])
        other_zero = df.loc[:, ~col_prog] == 0
        df.loc[other_zero.all(axis=1), col_prog] = 0

        return df
    
    def normailization(self):
        """
        Normalize protagonist's marker coordinate with respect to image size.
        Normalize other joints with respect to social distance.
        """
        df = pd.DataFrame(self.data.values,
                          columns=self.data.columns)
        
        # normalize all joints with a quarter of the threshold
        df = df/np.array(self.threshold/4)

        # normalize protagonist's marker
        # mean is the half of the image (800, 600)
        # std is the half of the mean
        col_marker = self.data.columns.isin([(f'individual{self.prog}','marker')])
        center = self.data.loc[:, col_marker]-np.array(self.image_size, dtype=np.float32)
        df.loc[:, col_marker] = center/np.array(self.image_size, dtype=np.float32)/2
        return df


if __name__ == "__main__":
    # file = ["/media/felan/G-DRIVE/AR-LABO/P45_S2_C1/trajectory/P45_S2_C1_01_preprocessed.csv",
    #         "/media/felan/G-DRIVE/AR-LABO/P45_S2_C1/trajectory/P45_S2_C1_02_preprocessed.csv"]
    # csv = [pd.read_csv(c, header=[0,1,2], index_col=0) for c in file]
    # data = pd.concat(csv, ignore_index=True)
    result = ProtagonistBasedCoordTrans(data="/media/felan/G-DRIVE/AR-LABO/P45_S6_C1/trajectory",
                                        data_path="/home/felan/LZL/research/interactive_behavior_model/anomal_detection/data/train",
                                        protagonist=7,
                                        bodypart=['nose', 'marker', 'tail'],
                                        zone_radius=180.0,
                                        mice_num=4,
                                        frame_len=10)
    
    result.train()
