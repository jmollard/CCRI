#### 
# A python script that takes numpy model files at their original resolution, for CMIP6 #
# SSP126, SSP585 and historical run, and produces the same data at a 1 x 1 degree grid.#
# Originally run on JASMIN
# Requires previous code to run first to create npy files
# Output multi-model mean TX35 for 2 models



import numpy as np
import mpl_toolkits
mpl_toolkits.__path__.append('/home/users/jmollard/miniconda3/envs/py37/lib/python3.7/site-packages/mpl_toolkits/')
from mpl_toolkits.basemap import interp #import basemap
import matplotlib.pyplot as plt
import netCDF4 as nc

H2020 = np.load('CMIP6_HadGEM3_2020_TX35.npy')
HS585 = np.load('CMIP6_HadGEM3_SSP585_TX35.npy')
HS126 = np.load('CMIP6_HadGEM3_SSP126_TX35.npy')

E2020 = np.load('CMIP6_ECEARTH3_2020_TX35.npy')
ES585 = np.load('CMIP6_ECEARTH3_SSP585_TX35.npy')
ES126 = np.load('CMIP6_ECEARTH3_SSP126_TX35.npy')


E5DIFF = (ES585 - E2020)/365.0
E1DIFF = (ES126 - E2020)/365.0
H5DIFF = (HS585 - H2020)/360.0
H1DIFF = (HS126 - H2020)/360.0

deslon = np.arange(0.5,360.0,1.0)
deslat = np.arange(-89.5,89.6,1.0)
desiremesh = np.meshgrid(deslon,deslat)
desiretotal2020 = np.zeros([2,180,360])
desiretotalS585 = np.zeros([2,180,360])
desiretotalS126 = np.zeros([2,180,360])

Efile = nc.Dataset('/badc/cmip6/data/CMIP6/ScenarioMIP/EC-Earth-Consortium/EC-Earth3/ssp126/r1i1p1f1/day/tasmax/gr/latest/tasmax_day_EC-Earth3_ssp126_r1i1p1f1_gr_20450101-20451231.nc')
Elat = Efile.variables['lat_bnds'][:]
Elon = Efile.variables['lon_bnds'][:]


Hfile = nc.Dataset('/badc/cmip6/data/CMIP6/ScenarioMIP/MOHC/HadGEM3-GC31-MM/ssp126/r1i1p1f3/day/tasmax/gn/latest/tasmax_day_HadGEM3-GC31-MM_ssp126_r1i1p1f3_gn_20150101-20191230.nc')
Hlat = Hfile.variables['lat_bnds'][:]
Hlon = Hfile.variables['lon_bnds'][:]

Hlata = np.nanmean(Hlat,axis=1)
Hlona = np.nanmean(Hlon,axis=1)
Elata = np.nanmean(Elat,axis=1)
Elona = np.nanmean(Elon,axis=1)


Hdlon = Hlona[13]-Hlona[12]
Edlon = Elona[13]-Elona[12]
H2020extend = np.c_[H2020[:,:],H2020[:,0]]
HS126extend = np.c_[HS126[:,:],HS126[:,0]]
HS585extend = np.c_[HS585[:,:],HS585[:,0]]
E2020extend = np.c_[E2020[:,:],E2020[:,0]]
ES126extend = np.c_[ES126[:,:],ES126[:,0]]
ES585extend = np.c_[ES585[:,:],ES585[:,0]]

#data2050extend = np.c_[data2050[:,:],data2050[:,0]]
Hlon_extended = np.hstack([Hlona,Hlona[-1]+Hdlon])
Elon_extended = np.hstack([Elona,Elona[-1]+Edlon])

desiretotal2020[0] = interp(datain=H2020extend,xin=Hlon_extended,yin=Hlata,xout=desiremesh[0],yout=desiremesh[1],order=1)
desiretotal2020[1] = interp(datain=E2020extend,xin=Elon_extended,yin=Elata,xout=desiremesh[0],yout=desiremesh[1],order=1)

desiretotalS126[0] = interp(datain=HS126extend,xin=Hlon_extended,yin=Hlata,xout=desiremesh[0],yout=desiremesh[1],order=1)
desiretotalS126[1] = interp(datain=ES126extend,xin=Elon_extended,yin=Elata,xout=desiremesh[0],yout=desiremesh[1],order=1)

desiretotalS585[0] = interp(datain=HS585extend,xin=Hlon_extended,yin=Hlata,xout=desiremesh[0],yout=desiremesh[1],order=1)
desiretotalS585[1] = interp(datain=ES585extend,xin=Elon_extended,yin=Elata,xout=desiremesh[0],yout=desiremesh[1],order=1)


MEAN2020 = np.nanmean(desiretotal2020,axis=0)
MEANS126 = np.nanmean(desiretotalS126,axis=0)
MEANS585 = np.nanmean(desiretotalS585,axis=0)


np.save('TX35_CMIP6_1degree_2020.npy',MEAN2020/360.0)
np.save('TX35_CMIP6_1degree_ssp585.npy',MEANS585/360.0)
np.save('TX35_CMIP6_1degree_ssp126.npy',MEANS126/360.0)
