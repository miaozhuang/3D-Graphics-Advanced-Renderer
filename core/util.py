
import math
import time

import numpy as np


class Util(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
    @staticmethod
    def radian(degree):
        return degree * math.pi / 180
    
    @staticmethod
    def get_rot_x_mat(degree):
        r = Util.radian(degree)
        return np.matrix([
                       [1, 0, 0],
                       [0, math.cos(r), -math.sin(r)],
                       [0, math.sin(r), math.cos(r)]
                         ])
    @staticmethod
    def get_rot_y_mat(degree):
        r = Util.radian(degree)
        return np.matrix([
                         [math.cos(r), 0, math.sin(r)],
                         [0, 1, 0],
                         [-math.sin(r), 0, math.cos(r)]])
    @staticmethod
    def normalize(v):
        return np.divide(v, np.linalg.norm(v))
    
    @staticmethod
    def cross_product(u, v):
        return np.cross(u, v)
    
    @staticmethod 
    def flat(mat):
#         return np.array(mat).reshape(-1).tolist()[0]
        return np.array(mat).reshape(-1).astype('f')
    
    @staticmethod
    def current_milli_time():
        return int(round(time.time() * 1000))
