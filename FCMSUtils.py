# -*- coding: utf-8 -*-
"""
@author: Taran Driver

This work is licensed under CC BY-NC-SA 4.0

To enquire about licensing opportunities, please contact the author:
    
tarandriver(at)gmail.com
"""

import numpy as np
import os
import scipy.io

def bsResamps(numScans, numResamp):
    "Returns numpy array of (resamples, scans in resample)."
#    return np.array([[randint(0, numScans-1) for x in range(numScans)] for y\
#     in range(numResamp)]) #this can be vectorised, as it is below
    return np.random.randint(0, numScans, size=(numResamp, numScans)) 

def circList(r):
    return [[x, y] for x in range(r+1) for y in range(r+1) if x**2+y**2<=r**2]
    
def clearCirc(array, r, c, circList, setVal=0):
    for x in circList:
        try:
            array[r+x[0], c+x[1]]=setVal
        except:
            pass
        try:
            array[r+x[0], c-x[1]]=setVal        
        except:
            pass
        try:
            array[r-x[0], c+x[1]]=setVal
        except:
            pass
        try:
            array[r-x[0], c-x[1]]=setVal
        except:
            pass
    
def covXI(scanList, varList):
    'Compute the covariance between a scan list and a single variable'   
    assert len(varList) == scanList.shape[0]
    
    numScans = len(varList)
    numSlices = scanList.shape[1]
    
    SiSx = (varList.sum(axis=0)) * (scanList.sum(axis=0)) #can be kept in
    #np.array format because this is multiplication of array by scalar
    
    Six=np.zeros(numSlices)

    for scanIndex in range(numScans): #scanIndex corresponds to scan
    #number (scanIndex+1) 

        Six += varList[scanIndex] * scanList[scanIndex] #Six for the 
        #single scan indexed by 'scanindex'. Can be kept in np.array format
        #because this is multiplication of array by scalar.
        
    return (Six/(numScans - 1))-(SiSx/(numScans * (numScans - 1)))
   
def cutAC(arrayToCut, pixAcross=8, threshFrac=1e-6):
    """Cuts along diagonal (x=y) line that is pixAcross pixels wide of a 
    SQUARE array (array is map, in which case 
    the x=y line is the autocorrelation (AC) line). All values above the 
    threshold value (= threshFrac *  max value of array) along this line 
    are set to the threshold value. 
    Runs quicker with these 3 clumsy loops than with 'try' clause.
    Returns the cut array."""
    
    array=np.copy(arrayToCut) #copy otherwise it changes the array in place
    
    thresh = np.nanmax(array) * threshFrac
    arrayWidth = len(array)
    
    for row in range(pixAcross, (arrayWidth - pixAcross)):
        for pixel in range((row - pixAcross), (row + (pixAcross + 1))):
            if array[row, pixel] > thresh:
                array[row, pixel] = thresh 
    
    for row2 in range((arrayWidth - pixAcross), arrayWidth):
        for col in range((row2 - pixAcross), arrayWidth):
            if array[row2, col] > thresh:
                array[row2, col] = thresh 
    
    for row3 in range(0, pixAcross):
        for col3 in range(0, (row3 + (pixAcross + 1))):
            if array[row3, col3] > thresh:
                array[row3, col3] = thresh          
                
    return array

def sampleSpectra(array_orig, samp_step):
    "take every samp_step m/z bin, starting with the zeroeth"
    return array_orig[:,np.arange(0, array_orig.shape[1], samp_step)]

def binSpectra(array_orig, bin_size):
    "bins, then divides each bin by the number of elements put into it. Each\
    bin is of same size in this implementation"
    quot,rem=divmod(array_orig.shape[1], bin_size)
    array_orig=array_orig[:,:bin_size*quot] #'trim'so that reshaping works
    
    return array_orig.reshape(array_orig.shape[0], quot, bin_size).sum(axis=2)\
           /float(bin_size)

def isInFeatList(mzs, featList, ionThresh=1.5, withNumber=False):
    """Is an m/z pair in a given list of features. mzs is tuple, length 2."""
    if not (isinstance(mzs, tuple) and len(mzs)==2):
        raise TypeError('mzs must be tuple of length 2')
    mz1,mz2=sorted(mzs)
    compFeats=[sorted((x[0],x[1])) for x in featList]
    
    num=-1 #number of feature (in Python indexing, so if it is the first
    #feature then 0 is returned)
    for y in compFeats:
        num+=1
        if abs(mz1-y[0])<=ionThresh and abs(mz2-y[1])<=ionThresh:
            if withNumber:
                return True, num
            else:
                return True
    #this only executes if a 'return' condition hasn't been met above 
    #(i.e. the feature not found)
    if withNumber:
        return False, np.nan #so that you can generally call 
        #"isIn,num=isInFeatList(...,...,withNumber=False)"
    else:
        return False
        
def maxIndices(array):
    """Returns row and column index of the maximum value in a 2D array"""
    posMax = np.unravel_index(array.argmax(), array.shape)
    return posMax[0], posMax[1]
    
def minIndices(array):
    """Returns row and column index of the minimum value in a 2D array"""
    posMin = np.unravel_index(array.argmin(), array.shape)
    return posMin[0], posMin[1]

def readmgf(path, maxScans=1e8): #maxScans=1 -> 1e8 on 20230413  
    scanCountA=0
    with open(path) as f:
        for i, lineA in enumerate(f):
            if lineA=='END IONS\n' or lineA=='END IONS':
                scanCountA+=1
                if scanCountA>=maxScans:
                    break    
    f.close()            
    
    array=np.zeros((i, 2))*np.nan
    scanCount=0
    count=0
    inIons=False
    
    with open(path) as f:
        for line in f:
            if line=='END IONS\n' or line=='END IONS': #if there is only one
            #scan in the mgf file, the new line character ('\n') is lost.
                inIons=False            
                scanCount+=1
                if scanCount>=maxScans:
                    break
                count+=1
                
            elif inIons:
                lspl=line.split()
                array[count,0], array[count,1]=float(lspl[0]), float(lspl[1])
                count+=1
                
            elif line!='BEGIN IONS\n' and '=' not in line:
                inIons=True
                lspl=line.split()
                array[count,0], array[count,1]=float(lspl[0]), float(lspl[1])
                count+=1
    f.close()
    
    return array[:count,:] 

def removeACFeats(featList, acThresh=5.5):
    'Removes all features closer than or equal to acThresh Da in m/z. Returns numpy array.'
    """Takes list or numpy array as featList"""
    featList=np.asarray(featList) #ensures input is numpy array to allow the 
    #fancy indexing below
    return featList[np.array([(abs(feat[0]-feat[1])>acThresh) for feat in \
    featList])]
    
def saveMatFile(array, saveTo, fieldName='array'):
    'No need to specify .mat extension'
    scipy.io.savemat(saveTo, {fieldName: array})
    
def saveSyx(scanFolder, numScansList=['all']):
    """Calculate Syx and save in scan folder, this is a computationally
    heavy operation and it is useful to have the result saved for further 
    analyses."""    
    
    numScansList.sort()

    numScansListStr = [str(x) for x in numScansList] #make a list of strings
    #of the numbers in numScansList, to enable printing below
    print('Performing saveSyx for '+scanFolder+' for '+\
    ', '.join(numScansListStr)+' scans') 
    
    params = np.load(scanFolder + '/array_parameters.npy')    
    
    if 'all' in numScansList:
        numScansList.remove('all')
        numScansList.append(int(params[1]))
        print('(all scans = '+str(int(params[1]))+')')
        
    numSlices = int(params[0]) #required to declare Syx
    Syx = np.zeros((numSlices, numSlices))
    array = np.load(scanFolder + '/array.npy')
    
    for scan in range(1, max(numScansList) + 1): #row 0 of array holds
    #m/z values so start from from 1
        spectrumX = np.matrix(array[scan,:]) #require it to be matrix for 
        #subsequent matrix multiplication. By default here, this is a row
        #vector.
        Syx += np.matrix.transpose(spectrumX) * spectrumX #column vector * row
        #vector -> outer product
        
        if scan % 100 == 0:
            print('Syx calculated for first '+str(scan)+' scans')
    
        if scan in numScansList:
            np.save(scanFolder + '/Syx_' + str(scan) + '_scans.npy', Syx)
            print('Syx saved for '+str(scan)+' scans')
        
    print('Completed saveSyx for '+scanFolder+' for '+\
    ', '.join(numScansListStr)+' scans')
    if 'all' in numScansListStr:
        print('(all scans = '+str(int(params[1]))+')')
    
    return
    
def saveSyxEinSum(scanFolder, numScans=13):
    """Calculate Syx and save in scan folder, this is a computationally
    heavy operation and it is useful to have the result saved for further 
    analyses."""    
    
    print('Performing saveSyxEinSum for '+scanFolder+' for '+str(numScans)+\
    ' scans')
    
    if numScans=='all':
        params = np.load(scanFolder + '/array_parameters.npy')
        numScans=int(params[1])
        print('(all scans = '+str(numScans)+')')

    array = np.load(scanFolder + '/array.npy')
    
    Syx=np.einsum('ij,ik->jk', array[1:numScans+1], array[1:numScans+1])  
    #row 0 of array holds m/z values so start from from 1
    np.save(scanFolder + '/Syx_' + str(numScans) + '_scans.npy', Syx)
    
    print('Completed saveSyxEinSum for '+scanFolder+' for '+str(numScans)+\
    ' scans')
    if numScans=='all':
        print('(all scans = '+str(int(params[1]))+')')
    
    return
    
def scaleToPower(array, power=0.25):
    """Scales an array to a given power"""    

    arrayabs = abs(array)
    arraysign = arrayabs/array
    arraysign[np.isnan(arraysign)] = 1.0
    arrayscaled = arrayabs**power
        
    return arrayscaled * arraysign

def sortList(listToSort, sortCol=3):
    'Default sortCol is 3 (significance). Returns numpy array.'
    """Takes list or numpy array as listToSort"""
    scores=[entry[sortCol] for entry in listToSort]
    return np.array([listToSort[index] for index in \
    reversed(np.argsort(scores))])
    
def varII(varList):
    'Compute the variance of a single variable, i'
    numScans = len(varList)
    Si2 = (varList**2).sum(axis=0)    
    S2i = varList.sum(axis=0)**2

    return (Si2 - S2i/numScans)/(numScans-1)
    
def topNfilt(array, topN, binSize=100):
    'Performs top N filtering on numpy nx2 array (1st col m/z, 2nd col ints)'
    """Doesn't require array to be sorted"""
    minMZ=np.nanmin(array[:,0])
    maxMZ=np.nanmax(array[:,0])
    binFloor=minMZ #this could be changed, but currently defines bin edges
    #according to min m/z in data
    
    filtList=np.zeros((1, 2)) #initialise array so that it can be appended to 
    binFloor=minMZ
    
    while (binFloor+binSize)<=maxMZ:
        inBin=array[(array[:,0]>=binFloor)&(array[:,0]<binFloor+binSize)]
        filtList=np.append(filtList, inBin[inBin[:,1].argsort()][-topN:,:], \
        axis=0)
        binFloor+=binSize
    
    finalTopN=int(np.around(topN*((maxMZ-binFloor)/np.float(binSize)))) #no.
    #of peaks to take from last bin is normalised according to how wide
    #last bin is
    inLastBin=array[(array[:,0]>=binFloor)]
    filtList=np.append(filtList, \
    inLastBin[inLastBin[:,1].argsort()][-finalTopN:,:], axis=0)
    
    filtList=filtList[1:,:] #remove the first zeros used to initialise
    
    return filtList[filtList[:,0].argsort()] #sort according to m/z
    


def readmgf2(mgfFile, scanFolder, maxScans=1e8): #added 20220816

    fullScanFolder = scanFolder
    
    print('Reading MGF file (readmgf2) '+mgfFile+' to '+fullScanFolder)
    
    import os
    os.mkdir(fullScanFolder)
    
    # first loop over to figure out how many scans there are
    # and how many spectral bins there are
    numScans=0
    binCount=0
    with open(mgfFile) as f:
        for i, line in enumerate(f):
            binCount += 1
            if line[:8]=='END IONS':
                if numScans == 0: numBins = binCount - 1
                numScans += 1
                if numScans >= maxScans:
                    break    
            if line[:7]=='CHARGE=' or line[:8]=='PEPMASS=': # changed 20240816
            # if line[:8]=='PEPMASS=': # added 20230413
                binCount = 0

    f.close()            
    
    array=np.zeros((numScans + 1, numBins))*np.nan
    
    scanCount=0
    binInd=0
    inIons=False
    
    with open(mgfFile) as f:
        for line in f:
            if line[:8]=='END IONS': #if there is only one
            #scan in the mgf file, the new line character ('\n') is lost.
                inIons=False            
                scanCount+=1
                if scanCount>=maxScans:
                    break
                
            elif inIons:
                if line[:7]=='CHARGE=' or line[:8]=='PEPMASS=': continue
                lspl=line.split()
                if scanCount==0:
                    array[0,binInd]=float(lspl[0])
                
                array[scanCount+1,binInd]=float(lspl[1])
                binInd+=1
                
            elif line[:7]=='CHARGE=' or line[:8]=='PEPMASS=':
                inIons=True
                binInd=0
    f.close()
    
    #Parameters for 'array_parameters.npy'
    params = np.zeros(8)

    params[0] = numBins
    params[1] = scanCount 
    params[2] = numScans 
    sliceSize = np.average(np.diff(array[0, :])) 
    params[3] = sliceSize
    params[4] = 0 
    params[5] = np.nanmin(array[0, :])
    params[6] = np.nanmax(array[0, :])
    params[7] = sliceSize
    
    array = array[:numScans + 1,:] #truncates the array on the occasion that the
    #number of scans read in is lower than the number of scans declared at the
    #text file, see above.
    
    np.save(fullScanFolder + '/array.npy', array)
    np.save(fullScanFolder + '/array_parameters.npy', params)
    
    print('readmgf2 complete for '+fullScanFolder)
    
    return

