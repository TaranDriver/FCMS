# -*- coding: utf-8 -*-
"""
@author: Taran Driver

This work is licensed under CC BY-NC-SA 4.0

To enquire about licensing opportunities, please contact the author:
    
tarandriver(at)gmail.com
"""

import numpy as np
import FCMS
import FCMSUtils

mgfFile="PATH_TO_MGF_FILE"
path="PATH_TO_SAVE_FCMS_ANALYSIS"

saveName=path+'/3000feats_default-params' # this is where the resampled
# features are saved
numScans='all'

#%%
print 'path:', path
print numScans, 'scans'
print 'save in', saveName

FCMSUtils.readmgf2(mgfFile, path)
scan1=FCMS.Scan(path)
map1=FCMS.CCovMap(scan1, scan1.tic(), numScans=numScans)
features=map1.analyse(3000)
np.save(saveName+'.npy', features)