
# dataframes
import pandas as pd
import h5py

#suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.TimeSeries = pd.Series 

#math
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import signal
from scipy import stats

## parallel plume sim

import parallel_plume_sim as pps


# performance
import multiprocessing as mp
from os import getpid
from threading import current_thread
#misc
np.set_printoptions(suppress=True)


## Read filenames

dir='~/DataAnalysis/data/puff_data/'
puff_data_filename = '~/DataAnalysis/data/puff_data/puff_data_20230123_131923.pickle'
# This should match your directory name from above -- you need to change this
dirname = '/home/gadfly/DataAnalysis/data/puff_data/puff_data_20230123_131923'
# Class for reading puff data
puff_reader = pps.read_puff_data.PuffReader(dirname)


def create_puff_data():
   # how many seconds to include per chunk?
    max_time = 4 

    # maximum distance in x and y to consider puffs before ignoring them?
    min_x = -5   
    max_x = 5
    min_y = -10
    max_y = 10

    pps.read_puff_data.split_puff_data_into_smaller_dataframes_and_cull_old_puffs(puff_data_filename, 
                                                                                max_time, 
                                                                                min_x, max_x, 
                                                                                min_y, max_y)
    
    # This should match your directory name from above -- you need to change this
    dirname = '/home/gadfly/DataAnalysis/data/puff_data/puff_data_20230123_131923'
    # Class for reading puff data
    puff_reader = pps.read_puff_data.PuffReader(dirname)

    return puff_reader

def get_concentration(inputs):


    i,df=inputs
    # concentration, wind_x, wind_y = puff_reader.get_odor_concentration_at_t_and_position(df.time[i], 
    #                                                                                      df.x[i], df.y[i])


    concentration, wind_x, wind_y = puff_reader.get_odor_concentration_at_t_and_position(df.time[i],df.xsrc[i],df.ysrc[i])
    return concentration, wind_x,wind_y


def main():

    # df=pd.DataFrame()
    # df['time']=np.arange(0,100,0.0001)
    # np.random.seed(42)
    # df['x']=np.random.uniform(-4, 4, size=len(df))
    # df['y']=np.random.uniform(-8, 8, size=len(df))
    dir_odor='~/DataAnalysis/data/Sprints/HighRes/'
    t=pd.read_hdf(dir_odor+'Windy/WindyMASigned.h5')
    print('Started Loop')


    pool = mp.Pool(processes=(mp.cpu_count()-1))
    inputs = [[i, t] for i in range(0,len(t))]
        
    
    concentration, w_x, w_y = zip(*pool.map(get_concentration, inputs))
    pool.terminate()      

    print('Finished Loop')
    df=pd.DataFrame()
    df['odor']=concentration
    df['windx']=w_x
    df['windy']=w_y
    df.to_hdf(dir+'puff_data_windy.h5', key='df', mode='w')

if __name__ == "__main__":
  # execute only if run as a script
  main()