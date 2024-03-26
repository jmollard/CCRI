####
# Python code to run as a script to create heatwave data from CMIP6 ScenarioMIP files
# Requires input from a job submission script for a model and scenario
# Run initially on JASMIN
# Produces an npy file of 10 year running mean of number of 3 day minimum CTX90pct heatwaves




import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal as SS
import pylab as pl
from netCDF4 import Dataset
import matplotlib as mpl
import os
import matplotlib.cm as cm
import pylab as pl
from scipy import stats, linalg
import struct
import cartopy.crs as ccrs

mod = sys.argv[1]
scen = sys.argv[2]


#scenarios = ('ssp126','ssp245','ssp370','ssp585')
#mods = ('MRI_ESM2_0',
#mods = ('KIOST_ESM','KACE_1_0_G','INM_CM5_0','ACCESS_ESM1','ARCCSS_ACCESS_CM2','CNRM_ESM2_1','CanESM5')
#MOD FOR SINGLE FILE ONLY)
mod_file_dict_ssp585 = {'MRI_ESM2_0':'/badc/cmip6/data/CMIP6/ScenarioMIP/MRI/MRI-ESM2-0/ssp585/r1i1p1f1/day/tasmax/gn/v20191108/tasmax_day_MRI-ESM2-0_ssp585_r1i1p1f1_gn_20150101-20641231.nc','KIOST_ESM': '/badc/cmip6/data/CMIP6/ScenarioMIP/KIOST/KIOST-ESM/ssp585/r1i1p1f1/day/tasmax/gr1/v20210125/tasmax_day_KIOST-ESM_ssp585_r1i1p1f1_gr1_20150101-21001231.nc','KACE_1_0_G': '/badc/cmip6/data/CMIP6/ScenarioMIP/NIMS-KMA/KACE-1-0-G/ssp585/r1i1p1f1/day/tasmax/gr/v20200317/tasmax_day_KACE-1-0-G_ssp585_r1i1p1f1_gr_20150101-21001230.nc','INM_CM5_0': '/badc/cmip6/data/CMIP6/ScenarioMIP/INM/INM-CM5-0/ssp585/r1i1p1f1/day/tasmax/gr1/v20190724/tasmax_day_INM-CM5-0_ssp585_r1i1p1f1_gr1_20150101-20641231.nc','ACCESS_ESM1': '/badc/cmip6/data/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp585/r1i1p1f1/day/tasmax/gn/v20191115/tasmax_day_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_20150101-20641231.nc','ARCCSS_ACCESS_CM2': '/badc/cmip6/data/CMIP6/ScenarioMIP/CSIRO-ARCCSS/ACCESS-CM2/ssp585/r1i1p1f1/day/tasmax/gn/v20191108/tasmax_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_20150101-20641231.nc','CNRM_ESM2_1': '/badc/cmip6/data/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-ESM2-1/ssp585/r1i1p1f2/day/tasmax/gr/v20191021/tasmax_day_CNRM-ESM2-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc','CanESM5': '/badc/cmip6/data/CMIP6/ScenarioMIP/CAMS/CAMS-CSM1-0/ssp585/r2i1p1f1/day/tasmax/gn/v20191106/tasmax_day_CAMS-CSM1-0_ssp585_r2i1p1f1_gn_20150101-20991231.nc'}

mod_file_dict_ssp245 = {'MRI_ESM2_0':'/badc/cmip6/data/CMIP6/ScenarioMIP/MRI/MRI-ESM2-0/ssp245/r1i1p1f1/day/tasmax/gn/v20190603/tasmax_day_MRI-ESM2-0_ssp245_r1i1p1f1_gn_20150101-20641231.nc','KIOST_ESM': '/badc/cmip6/data/CMIP6/ScenarioMIP/KIOST/KIOST-ESM/ssp245/r1i1p1f1/day/tasmax/gr1/v20201012/tasmax_day_KIOST-ESM_ssp245_r1i1p1f1_gr1_20150101-21001231.nc','KACE_1_0_G': '/badc/cmip6/data/CMIP6/ScenarioMIP/NIMS-KMA/KACE-1-0-G/ssp245/r1i1p1f1/day/tasmax/gr/v20200317/tasmax_day_KACE-1-0-G_ssp245_r1i1p1f1_gr_20150101-21001230.nc','INM_CM5_0': '/badc/cmip6/data/CMIP6/ScenarioMIP/INM/INM-CM5-0/ssp245/r1i1p1f1/day/tasmax/gr1/v20190619/tasmax_day_INM-CM5-0_ssp245_r1i1p1f1_gr1_20150101-20641231.nc','ACCESS_ESM1': '/badc/cmip6/data/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp245/r1i1p1f1/day/tasmax/gn/v20191115/tasmax_day_ACCESS-ESM1-5_ssp245_r1i1p1f1_gn_20150101-20641231.nc','ARCCSS_ACCESS_CM2': '/badc/cmip6/data/CMIP6/ScenarioMIP/CSIRO-ARCCSS/ACCESS-CM2/ssp245/r1i1p1f1/day/tasmax/gn/v20191108/tasmax_day_ACCESS-CM2_ssp245_r1i1p1f1_gn_20150101-20641231.nc','CNRM_ESM2_1': '/badc/cmip6/data/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-ESM2-1/ssp245/r1i1p1f2/day/tasmax/gr/v20191021/tasmax_day_CNRM-ESM2-1_ssp245_r1i1p1f2_gr_20150101-21001231.nc','CanESM5': '/badc/cmip6/data/CMIP6/ScenarioMIP/CAMS/CAMS-CSM1-0/ssp245/r2i1p1f1/day/tasmax/gn/v20191106/tasmax_day_CAMS-CSM1-0_ssp245_r2i1p1f1_gn_20150101-20991231.nc'}


mod_file_dict_ssp126 = {'MRI_ESM2_0':'/badc/cmip6/data/CMIP6/ScenarioMIP/MRI/MRI-ESM2-0/ssp126/r1i1p1f1/day/tasmax/gn/v20191108/tasmax_day_MRI-ESM2-0_ssp126_r1i1p1f1_gn_20150101-20641231.nc','KIOST_ESM': '/badc/cmip6/data/CMIP6/ScenarioMIP/KIOST/KIOST-ESM/ssp126/r1i1p1f1/day/tasmax/gr1/v20201012/tasmax_day_KIOST-ESM_ssp126_r1i1p1f1_gr1_20150101-21001231.nc','KACE_1_0_G': '/badc/cmip6/data/CMIP6/ScenarioMIP/NIMS-KMA/KACE-1-0-G/ssp126/r1i1p1f1/day/tasmax/gr/v20200317/tasmax_day_KACE-1-0-G_ssp126_r1i1p1f1_gr_20150101-21001230.nc','INM_CM5_0': '/badc/cmip6/data/CMIP6/ScenarioMIP/INM/INM-CM5-0/ssp126/r1i1p1f1/day/tasmax/gr1/v20190619/tasmax_day_INM-CM5-0_ssp126_r1i1p1f1_gr1_20150101-20641231.nc','ACCESS_ESM1': '/badc/cmip6/data/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp126/r1i1p1f1/day/tasmax/gn/v20191115/tasmax_day_ACCESS-ESM1-5_ssp126_r1i1p1f1_gn_20150101-20641231.nc','ARCCSS_ACCESS_CM2': '/badc/cmip6/data/CMIP6/ScenarioMIP/CSIRO-ARCCSS/ACCESS-CM2/ssp126/r1i1p1f1/day/tasmax/gn/v20191108/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_20150101-20641231.nc','CNRM_ESM2_1': '/badc/cmip6/data/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-ESM2-1/ssp126/r1i1p1f2/day/tasmax/gr/v20190328/tasmax_day_CNRM-ESM2-1_ssp126_r1i1p1f2_gr_20150101-21001231.nc','CanESM5': '/badc/cmip6/data/CMIP6/ScenarioMIP/CAMS/CAMS-CSM1-0/ssp126/r2i1p1f1/day/tasmax/gn/v20191106/tasmax_day_CAMS-CSM1-0_ssp126_r2i1p1f1_gn_20150101-20991231.nc'}


mod_file_dict_ssp370 = {'MRI_ESM2_0':'/badc/cmip6/data/CMIP6/ScenarioMIP/MRI/MRI-ESM2-0/ssp370/r1i1p1f1/day/tasmax/gn/v20190603/tasmax_day_MRI-ESM2-0_ssp370_r1i1p1f1_gn_20150101-20641231.nc','KIOST_ESM': '/badc/cmip6/data/CMIP6/ScenarioMIP/KIOST/KIOST-ESM/ssp370/r1i1p1f1/day/tasmax/gr1/v20210125/tasmax_day_KIOST-ESM_ssp370_r1i1p1f1_gr1_20150101-21001231.nc','KACE_1_0_G': '/badc/cmip6/data/CMIP6/ScenarioMIP/NIMS-KMA/KACE-1-0-G/ssp370/r1i1p1f1/day/tasmax/gr/v20200317/tasmax_day_KACE-1-0-G_ssp370_r1i1p1f1_gr_20150101-21001230.nc','INM_CM5_0': '/badc/cmip6/data/CMIP6/ScenarioMIP/INM/INM-CM5-0/ssp370/r1i1p1f1/day/tasmax/gr1/v20190618/tasmax_day_INM-CM5-0_ssp370_r1i1p1f1_gr1_20150101-20641231.nc','ACCESS_ESM1': '/badc/cmip6/data/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp370/r1i1p1f1/day/tasmax/gn/v20191115/tasmax_day_ACCESS-ESM1-5_ssp370_r1i1p1f1_gn_20150101-20641231.nc','ARCCSS_ACCESS_CM2': '/badc/cmip6/data/CMIP6/ScenarioMIP/CSIRO-ARCCSS/ACCESS-CM2/ssp370/r1i1p1f1/day/tasmax/gn/v20191108/tasmax_day_ACCESS-CM2_ssp370_r1i1p1f1_gn_20150101-20641231.nc','CNRM_ESM2_1': '/badc/cmip6/data/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-ESM2-1/ssp370/r1i1p1f2/day/tasmax/gr/v20191021/tasmax_day_CNRM-ESM2-1_ssp370_r1i1p1f2_gr_20150101-21001231.nc','CanESM5': '/badc/cmip6/data/CMIP6/ScenarioMIP/CAMS/CAMS-CSM1-0/ssp370/r2i1p1f1/day/tasmax/gn/v20191106/tasmax_day_CAMS-CSM1-0_ssp370_r2i1p1f1_gn_20150101-20991231.nc'}

#scen = 'ssp585'
if scen == 'ssp585':
    filename = mod_file_dict_ssp585[mod]
elif scen == 'ssp245':
    filename = mod_file_dict_ssp245[mod]        
elif scen == 'ssp126':
    filename = mod_file_dict_ssp126[mod]  
elif scen == 'ssp370':
    filename = mod_file_dict_ssp370[mod]

data = Dataset(filename)
tasmaxall = data.variables['tasmax'][:]
if mod == 'CNRM_ESM2_1':
    print "No lat lon data available"
else:
    lat_bnds = data.variables['lat_bnds'][:]
    lon_bnds = data.variables['lon_bnds'][:]
    lat = (lat_bnds[:,0]+lat_bnds[:,1])/2.0
    lon = (lon_bnds[:,0]+lon_bnds[:,1])/2.0
    np.save('/home/users/jmollard/CCRI/data/model_metadata/%s_latitudes.npy'%(mod),lat)
    np.save('/home/users/jmollard/CCRI/data/model_metadata/%s_longitudes.npy'%(mod),lon)
#tasmax2020 = tasmaxall[0:3650,:,:]
#tasmax2050 = tasmaxall[10950:14600,:,:]
dy,lt,ln = np.shape(tasmaxall)
if dy%360 == 0: 
    day_year = 360
    tasmax2020 = tasmaxall[0:3600,:,:]
    tasmax2050 = tasmaxall[10800:14400,:,:]
    start_years = np.arange(0,dy-3599,day_year)
    end_years = np.arange(3600,dy+1,day_year)
elif dy%365 == 0:
    day_year = 365
    tasmax2020 = tasmaxall[0:3650,:,:]
    tasmax2050 = tasmaxall[10950:14600,:,:]
    start_years = np.arange(0,dy-3599,day_year)
    end_years = np.arange(3650,dy+1,day_year)
else:
    leap_days = np.arange(525,dy,1461)
    tasmaxall = np.delete(tasmaxall,leap_days,0)
    dy,lt,ln = np.shape(tasmaxall)
    day_year = 365
    tasmax2020 = tasmaxall[0:3650,:,:]
    tasmax2050 = tasmaxall[10950:14600,:,:]
    start_years = np.arange(0,dy-3599,day_year)
    end_years = np.arange(3650,dy+1,day_year)
testx = dy%day_year
print "If this is zero", testx, "Then there is no trouble so far"
tasmax2020 = np.reshape(tasmax2020,(10,day_year,lt,ln))

CTX90pct = np.zeros([day_year,lt,ln])
for i in range(0,day_year):
    X_T = np.roll(tasmax2020,7+i,axis=1)
    x = X_T[:,0:15,:,:]
    CTX90pct[i,:,:] = np.percentile(x,90,axis=(0,1))

CTXgrid = np.vstack([CTX90pct,CTX90pct,CTX90pct,CTX90pct,CTX90pct,CTX90pct,CTX90pct,CTX90pct,CTX90pct,CTX90pct])
print "CTX grid and 2020 set made"
#create 10 year windows

#start_years = np.arange(0,dy-3599,day_year)
#end_years = np.arange(3650,dy,day_year)
#years = np.arange(2020,2061,1)
CTall = np.zeros([len(start_years),lt,ln])
#tasmaxall = np.reshape(tasmaxall,(dy/365,365,lt,ln))
for i in xrange(0,len(start_years)):
    decadewindow = tasmaxall[start_years[i]:end_years[i],:,:]
    hotdays = decadewindow > CTXgrid
    #hotdays = np.reshape(hotdays,(day_year*10,lt,ln))    
    for j in xrange(0,lt):
        for k in xrange(0,ln):
            Y = np.diff(np.where(np.concatenate(([hotdays[0,j,k]],hotdays[:-1,j,k] != hotdays[1:,j,k],[1])))[0])[::2]
            CTall[i,j,k] = sum(np.array(Y)>3)

np.save('/home/users/jmollard/CCRI/data/Heatwaves/%s/World_%s_%s_heatwave_sens_10yearrunning_3day.npy'%(scen,mod,scen),CTall)

print mod, 'is now complete'
