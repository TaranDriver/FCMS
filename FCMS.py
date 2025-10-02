# -*- coding: utf-8 -*-
"""
@author: Taran Driver

This work is licensed under CC BY-NC-SA 4.0

To enquire about licensing opportunities, please contact the author:
    
tarandriver(at)gmail.com
"""

import os
import numpy as np
from scipy import interpolate
from scipy import ndimage
from FCMSUtils import maxIndices, varII, covXI, cutAC, saveSyxEinSum, scaleToPower, circList, clearCirc, bsResamps, createArrayParams
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import time
import h5py

class Scan:
    def __init__(self, scanFolder, AGCtarget=100):
        self.scanFolder = scanFolder
        self.AGCtarget = AGCtarget
        self.normFactor = AGCtarget/10000.
        self.scanList = self.normFactor*np.load(scanFolder + '/array.npy')[1:]
        params = np.load(scanFolder + '/array_parameters.npy')
        self.sliceSize = params[3]
        self.minMZ = params[5] 
        self.maxMZ = params[6]
        #The text file that gets read in reports a min mz and max mz and then
        #gives intensity readings at each m/z value. The min m/z value at 
        #which is gives an intensity reading for is one sliceSize higher than
        #the reported min m/z, the max m/z value that it gives an intensity 
        #reading for is the same as the reported max m/z.
        self.fullNumScans = int(params[1])
        
    def tic(self):
        return self.scanList.sum(axis=1)
    
    def aveTIC(self):
        'Mean-averaged total ion count across all scans'
        return self.totIonCount().sum()/self.fullNumScans
        
    def ionCount(self, centralMZ, width, numScans='all', returnBounds=False):
        """returns ion count across a certain m/z range defined by central 
        m/z and width. Output is 1D array of count for each scan"""
        binsOut = np.ceil(float(width)/(self.sliceSize*2.)) #so as to err on
        #the side of the range being too large        
        centralIndex = self.mz2index(centralMZ)
        
        fromIndex = centralIndex - binsOut
        toIndex = centralIndex + binsOut
        
        x=self.fullNumScans if numScans=='all' else numScans
        
        fromMZ=round(self.index2mz(fromIndex), 2)
        toMZ=round(self.index2mz(toIndex), 2)
        
        print('ionCount from bin m/z '+ str(fromMZ)+' to bin m/z '+\
        str(toMZ)+' for '+str(x)+' scans')
        
        if returnBounds:
            return self.scanList[:x,fromIndex:toIndex+1].sum(axis=1), fromMZ,\
            toMZ
        else:
            return self.scanList[:x,fromIndex:toIndex+1].sum(axis=1)
    
    def index2mz(self, index):
        """Provides the m/z value relating to the specified index for the 
        relevant scan dataset. index need not be an integer value."""
        return self.minMZ + index*self.sliceSize
        
    def mz2index(self, mz):
        """Provides the closest (rounded) integer m/z slice index relating to  
        the m/z value for the relevant scan dataset."""
        if mz < self.minMZ or mz > self.maxMZ:
            raise ValueError('m/z value outside range of m/z values for this scan')
        if mz % self.sliceSize < self.sliceSize/2:
            return int((mz - self.minMZ)/self.sliceSize)
        else:
            return int((mz - self.minMZ)/self.sliceSize + 1)
        #this doesn't work with arrays, yet. (index2mz does)
    
    def oneD(self, numScans='all'):
        x=self.fullNumScans if numScans=='all' else numScans
        return self.scanList[:x].sum(0)    

    def plot1D(self, numScans='all'):
        'Plot the averaged 1D spectrum from array.npy'        
        oneD=self.oneD(numScans=numScans)
        plt.plot(np.linspace(self.minMZ, self.maxMZ, self.scanList.shape[1], \
        endpoint=True), oneD/(np.nanmax(oneD)*0.01))
        plt.xlabel('m/z')
        plt.ylabel('Relative abundance, %')  
        
class Map:
    'Simple or contingent covariance map'
    def __init__(self, scan, numScans='all'):
        self.scan = scan
        if numScans=='all':
            self.numScans = self.scan.fullNumScans
        else:
            self.numScans = numScans
        self.build()
    
    def syx(self):
        'Syx is attribute, syx is method'
        try: 
            return self.Syx
        except:
            syxPath = self.scan.scanFolder + \
            '/Syx_'+str(self.numScans)+'_scans.npy'
            
            if os.path.isfile(syxPath):
                return self.scan.normFactor**2 * np.load(syxPath)
            else:
                print('Syx not saved for this map, beginning calculation with saveSyxEinSum...')
                saveSyxEinSum(self.scan.scanFolder, numScans=self.numScans)
                return self.scan.normFactor**2 * np.load(syxPath)
            
    def loadSyx(self):
        'Syx is attribute, syx is method'
        self.Syx = self.syx()
            
    def plot(self, mapTitle='', save=False, figFileName='', \
            fullRange=True, minMZx=100, maxMZx=101, minMZy=100, maxMZy=101):
                
        cdict =     {'red':       ((0.0,     1.0, 1.0),
                           (0.16667, 0.0, 0.0),
                           (0.33333, 0.5, 0.5),
                           (0.5,     0.0, 0.0),
                           (0.66667, 1.0, 1.0),
                           (1,       1.0, 1.0)),
                          
             'green':     ((0.0,     0.0, 0.0),
                           (0.16667, 0.0, 0.0),
                           (0.33333, 1.0, 1.0), 
                           (0.5,     0.5, 0.5),
                           (0.66667, 1.0, 1.0),
                           (0.83333, 0.0, 0.0),
                           (1.0,     1.0, 1.0)),
                 
             'blue':      ((0.0,     1.0, 1.0),
                           (0.33333, 1.0, 1.0),
                           (0.5,     0.0, 0.0),
                           (0.83333, 0.0, 0.0),
                           (1.0,     1.0, 1.0))}
        
        Alps1 = LinearSegmentedColormap('Alps1', cdict)
        
        toPlot = scaleToPower(cutAC(self.array))
        
        if not fullRange:
           
            minIndexX = (minMZx - self.scan.minMZ)/self.scan.sliceSize
            maxIndexX = (maxMZx - self.scan.minMZ)/self.scan.sliceSize
            
            minIndexY = (minMZy - self.scan.minMZ)/self.scan.sliceSize
            maxIndexY = (maxMZy - self.scan.minMZ)/self.scan.sliceSize
            
            toPlot = toPlot[int(minIndexY):int(maxIndexY), \
            int(minIndexX):int(maxIndexX)]
            
        v_max = np.nanmax(toPlot)
        v_min = -v_max 
        
        plt.figure(figsize=((200/9), 20))
        
        fig1 = gridspec.GridSpec(36, 40) #Set up GridSpec to allow custom
        #placement of figures
        cvMap = plt.subplot(fig1[0:29, 7:36])
        cvMap1 = cvMap.pcolorfast(toPlot, vmin=v_min, vmax=v_max, cmap=Alps1)
        
        plt.xlabel('Mass-to-Charge Ratio, Da/C')
        plt.ylabel('Mass-to-Charge Ratio, Da/C')
        plt.title(mapTitle, fontsize=14)
        
        ax = plt.gca()
        ax.set_xticks([])    
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])    
        
        cbar = plt.subplot(fig1[2:28, 38])
        plt.colorbar(cvMap1, cax=cbar)

        ints = self.scan.scanList[:self.numScans].sum(axis=0)        
        mzs = np.linspace(self.scan.minMZ, self.scan.maxMZ, len(ints))
        
        if fullRange:
            oneDDataXHor = mzs
            oneDDataYHor = ints
           
            oneDDataXVer = mzs
            oneDDataYVer = ints
            
        else:
            
            minIndex1DHor = (minMZx - self.scan.minMZ)/self.scan.sliceSize
            maxIndex1DHor = (maxMZx - self.scan.minMZ)/self.scan.sliceSize
            oneDDataXHor = mzs[int(minIndex1DHor):int(maxIndex1DHor)]
            oneDDataYHor = ints[int(minIndex1DHor):int(maxIndex1DHor)]
            
            minIndex1DVer = (minMZy - self.scan.minMZ)/self.scan.sliceSize
            maxIndex1DVer = (maxMZy - self.scan.minMZ)/self.scan.sliceSize
            oneDDataXVer = mzs[int(minIndex1DVer):int(maxIndex1DVer)]
            oneDDataYVer = ints[int(minIndex1DVer):int(maxIndex1DVer)]
                         
        oneDSpectrumHor = plt.subplot(fig1[30:36, 7:36]) #horizontal
        #1D spectrum
        oneDSpectrumHor.plot(oneDDataXHor, oneDDataYHor)
        plt.axis('tight')
        plt.xlabel('Mass-to-Charge Ratio, Da/C')
        plt.ylabel('Normalised Signal Intensity')     
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        oneDSpectrumVer = plt.subplot(fig1[0:29, 0:6]) #vertical
        #1D spectrum
        oneDSpectrumVer.plot(oneDDataYVer, oneDDataXVer)
        plt.axis('tight')
        plt.xlabel('Normalised Signal Intensity')
        plt.ylabel('Mass-to-Charge Ratio, Da/C')
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        #Set parameters ticks on the plots
        majorLocator = MultipleLocator(50) #how far apart labelled ticks are
        minorLocator = MultipleLocator(5) #how far apart unlabelled ticks are
        
        oneDSpectrumHor.xaxis.set_major_locator(majorLocator)
        oneDSpectrumHor.xaxis.set_minor_locator(minorLocator)
    
        oneDSpectrumVer.yaxis.set_major_locator(majorLocator)
        oneDSpectrumVer.yaxis.set_minor_locator(minorLocator)
    
        oneDSpectrumHor.tick_params(axis='both', length=6, width=1.5)
        oneDSpectrumHor.tick_params(which='minor', axis='both', length=4, \
        width=1.5)
        oneDSpectrumVer.tick_params(axis='both', length=6, width=1.5)
        oneDSpectrumVer.tick_params(which='minor', axis='both', length=4, \
        width=1.5)
        
        plt.show()
        
        if save:
            plt.savefig(figFileName)
            
        return
        
    def analyse(self, numFeats, clearRad=25, chemFilt=[], \
        chemFiltTol=2.0, shapeFilt=False, shFiltThr=-0.2, shFiltRef='map',\
        shFiltRad=15, breakTime=3600,\
        pixOut=15, comPixOut=3, cutPeaks=True, integThresh=0, numRays=100,\
        perimOffset=-0.5, pixWidth=1, sampling='jackknife', bsRS=None,\
        bsRS_diffs=None, saveAt=False, basePath=None, useT4R=False, \
        printAt=50): 
        #last 13 kwargs (after break) are parameters for sampleFeats. bsRs is 
        #the bootstrap resamples provided
        'Picks numFeats feats from map and then samples them using jackknife'

        indexList=self.topNfeats(numFeats, clearRad=clearRad,\
        chemFilt=chemFilt, chemFiltTol=chemFiltTol, shapeFilt=shapeFilt,\
        shFiltThr=shFiltThr, shFiltRef=shFiltRef, shFiltRad=shFiltRad,\
        breakTime=breakTime, boundFilt=True, boundLimit=pixOut,\
        returnDiscards=False) 
        #boundFilt must be true with boundLimit as pixOut for sampleFeats 
        #to not raise an exception
        
        return self.sampleFeats(indexList, pixOut=pixOut, \
        comPixOut=comPixOut, cutPeaks=cutPeaks, integThresh=integThresh, \
        numRays=numRays, perimOffset=perimOffset, pixWidth=pixWidth, \
        sampling=sampling, bsRS=bsRS, bsRS_diffs=bsRS_diffs, saveAt=saveAt,\
        basePath=basePath, useT4R=useT4R, printAt=printAt)
    
    def sampleFeats(self, indexList, pixOut=15, comPixOut=3,\
        cutPeaks=True, integThresh=0, numRays=100, perimOffset=-0.5,\
        pixWidth=1, sampling='jackknife', bsRS=None, bsRS_diffs=None,\
        useT4R=False, saveAt=False, basePath=None, printAt=50):
        'Returns m/z\'s, volume and sig for feats with r,c in indexList'    
        """pixOut is for peak dimensions, comPixOut is for centre of mass
        routine, pixWidth is for peak integration (how many Da each pixel 
        corresponds to, if we care)."""
        
        featList=np.zeros((len(indexList), 4)) 
        featNo=0
        
        if type(saveAt) is int and basePath is None:
            basePath=raw_input('base file name for peaks file:')        
        if sampling=='bootstrap' and bsRS is None:
            print('calculating bootstrap resamples...')
            bsRS=bsResamps(self.numScans, self.numScans)
            print('calculating bootstrap resamples FINISHED')
        elif sampling=='bootstrap_with_diffs' and bsRS_diffs is None:
            raise StandardError('must provide a (numResamples+1,numScans) differences array for bootstrap_with_diffs')
        
        for indices in indexList:
            peak=self.getPeak(indices[0], indices[1], pixOut=pixOut)
            com_r, com_c=peak.com(pixEachSide=comPixOut)
            
            if cutPeaks:
                if useT4R:
                    template=peak.template4ray(integThresh=integThresh)
                else:
                    template=peak.templateNray(integThresh=integThresh,\
                    numRays=numRays, perimOffset=perimOffset)
                peak.cutPeak(template)
            else:
                template=None #because template is passed as kwarg to the 
                #***ResampleVar method below and needs to be defined
                
            peakVol=peak.bivSplIntegrate(pixWidth=pixWidth)
            if sampling=='jackknife':
                peakVar=peak.jkResampleVar(cutPeak=cutPeaks, template=template)
            elif sampling=='bootstrap':
                peakVar=peak.bsResampleVar(bsRS, cutPeak=cutPeaks, \
                template=template)
            elif sampling=='bootstrap_with_diffs':
                peakVar=peak.bsDiffResampleVar(bsRS_diffs, cutPeak=cutPeaks, \
                template=template)
            else:
                raise TypeError('\''+sampling+'\''+' not a recognised method of resampling')
            
            featList[featNo]=round(self.scan.index2mz(indices[0]-\
            comPixOut+com_r),2), round(self.scan.index2mz(indices[1]-\
            comPixOut+com_c),2), peakVol, peakVol/np.sqrt(peakVar)
            
            featNo+=1
            if featNo%printAt==0:            
                print('sig calculated for feature '+str(featNo))
            if type(saveAt) is int and featNo%saveAt==0:
                np.save(basePath+'_feats'+str(int(featNo-saveAt+1))+'to'+\
                str(int(featNo))+'.npy', featList[featNo-saveAt:featNo])
            
        return featList
        
    def topNfeats(self, numFeats, clearRad=25, chemFilt=[], chemFiltTol=2.0,
                  shapeFilt=False, shFiltThr=-0.2, shFiltRef='map',\
                  shFiltRad=15, boundFilt=True, boundLimit=15, breakTime=3600,\
                  returnDiscards=False):
        #chemFiltTol in Da, chemFilt condition is less than or equal to.
        #shFiltThr is fraction of highest peak on a/c cut map.
        """'returnDiscards' allows to return features discarded by any applied
        filters as well (this array is the second element in the tuple)."""
        
        array=np.triu(cutAC(self.array))
        if shFiltRef=='map': #'shape filter reference' taken globally from map
            shFiltThrAbs=shFiltThr*np.nanmax(array)
        
        #'Picks the top N highest legitimate features'
        feats=np.zeros((numFeats, 2))
        featCount=0
        
        if returnDiscards:
            discardFeats=[] #list of all feats discarded by any of the filters
            #not declared as array because you do not know how many features
            #will be discarded (it could certainly be more than those which
            #aren't discarded)
        
        circListClear=circList(clearRad)
        if shapeFilt:
            circListFilt=circList(shFiltRad)
        
        startTime=time.time()
        
        while featCount<numFeats:
            
            featPass=True
            r,c = maxIndices(array)
            if shFiltRef=='peak': #shape filter reference taken as height of
                shFiltThrAbs=shFiltThr*array[r,c] #each individual peak
            
            #Apply chemical filter if requested                    
            for chemFiltmz in chemFilt:
                if abs(self.scan.index2mz(r) - chemFiltmz) <= chemFiltTol or \
                abs(self.scan.index2mz(c) - chemFiltmz)  <= chemFiltTol: #this 
                #takes m/z of highest pixel, not m/z of CoM of feature
                    featPass=False
                    break           
                
            #Apply shape filter if requested  
            if shapeFilt and featPass:
                for x in circListFilt:
                    if array[r+x[0],c+x[1]]<=shFiltThrAbs or \
                    array[r+x[0],c-x[1]]<=shFiltThrAbs or \
                    array[r-x[0],c+x[1]]<=shFiltThrAbs or \
                    array[r-x[0],c-x[1]]<=shFiltThrAbs:
                        featPass=False
                        break
            
            #Apply boundary filter so that features too close to the edge of 
            #the map to be sampled are not counted
            if boundFilt and featPass:            
                if r < boundLimit or c < boundLimit or r  > len(array) - \
                (boundLimit+1) or c > len(array) - (boundLimit+1):
                    featPass=False
                
            if featPass:
                feats[featCount,0], feats[featCount,1] = r, c
                featCount+=1

                if featCount%100==0:
                    print('found '+str(featCount)+' good features')
                    
            elif returnDiscards:
                discardFeats.append([r, c])
                
            clearCirc(array, r, c, circListClear)
            
            if time.time()-startTime>breakTime:
                print('topNfeats breaking out at '+str(featCount)+' features'\
                +' - running time exceeded '+str(breakTime)+' secs')
                if not returnDiscards:
                    return feats[:featCount] #cut to the last appended feature
                else:
                    return feats[:featCount], np.array(discardFeats) 
                    #discardFeats was a list so is converted to array for 
                    #consistency of output
        
        if not returnDiscards:
            return feats
        else:    
            return feats, np.array(discardFeats) #discardFeats was a list
            #so is converted to array for consistency of output
        
class CCovMap(Map):
    'There are many possible ways to split up scans! Starting with simplest'
    
    def __init__(self, scan, cCovParams, numScans='all', sliceMeth='naive', \
                 scansPerSet=100): # sliceMeth is 'slice method', naive is 
                                   # just dividing into same size sets
        if numScans=='all':
            self.cCovParams = cCovParams
        else:
            self.cCovParams = cCovParams[:numScans]
        self.sliceMeth = sliceMeth
        self.scansPerSet = scansPerSet
        self.setIdcs = self.setIdcs()
        self.setSize = np.array([len(x) for x in self.setIdcs])
        Map.__init__(self, scan, numScans)
        
    def setIdcs(self):
        'returns a list of scan indices for each set'
        cCovParOrd = np.argsort(self.cCovParams) # argsort gives indices from
        # lowest to highest
        if self.sliceMeth=='naive':
            self.numSets = int(np.ceil(len(cCovParOrd) / \
                                       float(self.scansPerSet)))
            return [cCovParOrd[x*self.scansPerSet:(x+1)*self.scansPerSet] \
                        for x in np.arange(self.numSets)]
            
    def syx(self):
        'Syx is attribute, syx is method. This is being overloaded for cCov \
        because now we need the way the scans are being divided up to define.\
        We also want to save Syx as h5 so we can index into it quickly'
        try: 
            return self.Syx # if numpy array already in RAM
        except: pass
        
        if os.path.isfile(self.syxPath()):
            return self.scan.normFactor**2 * \
                   h5py.File(self.syxPath(), 'r')['Syx'][:]
        else:
            print('Syx not saved for this map, beginning calculation with saveSyxEinSumCC...')
            self.saveSyxEinSumCC()
            return self.scan.normFactor**2 * \
                   h5py.File(self.syxPath(), 'r')['Syx'][:]
                   
    def syxH5(self, rIdcs=(None,None), cIdcs=(None,None), ssIdcs=(None,None)):
        'for the large Syx arrays (with new dimension for scan sets) we are \
         using h5 files. the slicing is done here in the h5 file, so the \
         array which is returned is not the complete thing [unlike for syx()]'
        try:
            return self.scan.normFactor**2 * \
                   self.SyxH5[rIdcs[0]:rIdcs[1],cIdcs[0]:cIdcs[1],\
                            ssIdcs[0]:ssIdcs[1]]
        # the above clause is different to its equivalent in syx() because
        # the SyxH5 is loaded as an h5 file so has not had the normFactor
        # normalisation already applied to it
                   
        except:            
            if os.path.isfile(self.syxPath()):
                return self.scan.normFactor**2 * \
                       h5py.File(self.syxPath(),'r')['Syx'][rIdcs[0]:rIdcs[1],\
                                       cIdcs[0]:cIdcs[1], ssIdcs[0]:ssIdcs[1]]
            else:
                print('Syx not saved for this map, beginning calculation with saveSyxEinSumCC...')
                self.saveSyxEinSumCC()
                return self.scan.normFactor**2 * \
                       h5py.File(self.syxPath(),'r')['Syx'][rIdcs[0]:rIdcs[1],\
                                       cIdcs[0]:cIdcs[1], ssIdcs[0]:ssIdcs[1]]
                       
    def loadSyxH5(self):
        
        if os.path.isfile(self.syxPath()):
            self.SyxH5 = h5py.File(self.syxPath(), 'r')['Syx']
        else:
            print('Syx not saved for this map, beginning calculation with saveSyxEinSumCC...')
            self.saveSyxEinSumCC()
            self.SyxH5 = h5py.File(self.syxPath(), 'r')['Syx']
        
    def scanSetID(self):
        
        if self.sliceMeth == 'naive':
            # this is the string required to define the particular set
            # of scans being used
            scanSetID = 'sps'+str(self.scansPerSet) #scans per set
            
        return scanSetID
        
    def syxPath(self):
        return self.scan.scanFolder+'/Syx_'+str(self.numScans)+\
              '_scans_cc_'+self.sliceMeth+'_'+self.scanSetID()+'.h5'
              
    def mapPath(self):
        return self.scan.scanFolder+'/CCyx_'+str(self.numScans)+\
              '_scans_cc_'+self.sliceMeth+'_'+self.scanSetID()+'.npy'
        
            
    def saveSyxEinSumCC(self):
        """Calculate Syx and save in scan folder, this is a computationally
        heavy operation and it is useful to have the result saved for further 
        analyses. This is for c cov so Syx has a third dimension which is
        number of sets""" 
        
        print('Performing saveSyxEinSumCC for ' + self.syxPath())

        array = np.load(self.scan.scanFolder + '/array.npy')

        Syx = np.zeros((array.shape[1], array.shape[1], len(self.setIdcs)),\
                       dtype=np.single)
        for i, setIdx in enumerate(self.setIdcs):
            if i%10==0:
                print('calculating Syx for set %i / %i' % (i, len(self.setIdcs)))
            scans_i = array[setIdx+1] # row 0 of array holds m/z values so 
            # start from from 1
            Syx[:,:,i] = np.einsum('ij,ik->jk', scans_i, scans_i)  

        import h5py
        with h5py.File(self.syxPath(), 'w') as f:
            f.create_dataset('Syx', data=Syx.astype(np.single))
        f.close()
        
        print('Completed saveSyxEinSumCC for ' + self.syxPath())
        
        return
              
    def build(self):
        """Calculate full c covariance map with single c 
        covariance parameter cCovParams"""
        
        if os.path.isfile(self.mapPath()):
            self.array = np.load(self.mapPath())
            
        else:
            print('cCov not yet calculated for this map, beginning calculation...') 
            mapScanList = self.scan.scanList[:self.numScans, :]
            self.loadSyxH5() # intialize, we're going to slice into it for each
            # scan set. This is slower than slicing into an array
            # but probably worth it for the RAM saved.
            covSumFull = np.zeros((mapScanList.shape[1], mapScanList.shape[1]))
                    
            for i in np.arange(len(self.setIdcs)):
                if i%1==0:
                    print('calculating cCov for set %i / %i' % \
                    (i, len(self.setIdcs)))
                Syx_i = np.squeeze(self.syxH5(ssIdcs=(i,i+1)))
                Sx_i = mapScanList[self.setIdcs[i]].sum(0)
                covSumFull += (Syx_i - np.outer(Sx_i, Sx_i) / \
                              float(self.setSize[i])) / \
                               float(self.setSize[i] - 1)
            cCovYX = covSumFull / float(len(self.setIdcs))
            print('completed cCov calculation')
            np.save(self.mapPath(), cCovYX)
            self.array = cCovYX

    def getPeak(self, r, c, pixOut=15):
        return CCovPeak(self, r, c, pixOut=pixOut)
        
class Peak:
    'Any peak from a 2D map'
    def __init__(self, array):
        self.array=array
        
    def com(self, pixEachSide=3):
        """Returns the row and column index of the centre of mass of a square 
        on the 2D array 'array', centred on the pixel indexed by r, c and of 
        width (2 * pixEachSide + 1)"""    

        square = np.zeros((2*pixEachSide+1, 2*pixEachSide+1))
        
        for i in range(2*pixEachSide+1):
            for j in range(2*pixEachSide+1):
                square[i,j] = self.array[self.pixOut-pixEachSide+i, \
                self.pixOut-pixEachSide+j]    
                #It is clearer to cast the indexing of the array like this 
                #because it is consistent with how the index of the COM is 
                #returned
                
        squareMin = np.nanmin(square)
        if squareMin < 0:    
            squareTwo = abs(squareMin) * \
            np.ones((2*pixEachSide+1, 2*pixEachSide+1))
            square += squareTwo
            
        COMi, COMj = ndimage.measurements.center_of_mass(square)
        
        return COMi, COMj #self.r-pixEachSide+COMi, self.c-\
        #pixEachSide+COMj

    def template4ray(self, integThresh=0):
        "This should be replaced with the improved templateNray() method"
        """peakSquare must be square"""
    
        north = south = east = west = 1
        valueN = valueS = valueE = valueW = integThresh + 1 #to make sure it 
        #starts off above the integThresh
    
        peakShape = np.ones((len(self.array), \
        len(self.array)), dtype=bool)     
        
        rad, rem = divmod(len(self.array), 2)
        if rem != 1:
            raise ValueError('square width not odd integer -> ambiguous apex')
        
        while valueN > integThresh:
            north += 1
            if north <= rad:
                valueN = self.array[rad+north,rad]
            else:
                break
        north -= 1 #gives index of last pixel above threshold
        
        while valueS > integThresh:
            south += 1
            if south <= rad:
                valueS = self.array[rad-south,rad]
            else:
                break
        south -= 1 #gives index of last pixel above threshold
            
        while valueE > integThresh:
            east += 1
            if east <= rad:
                valueE = self.array[rad,rad+east]
            else:
                break
        east -= 1 #gives index of last pixel above threshold
            
        while valueW > integThresh:
            west += 1
            if west <= rad:
                valueW = self.array[rad,rad-west]
            else:
                break
        west -= 1 #gives index of last pixel above threshold
        
        #Now cut out the parts of the square you don't want
        for plusi in range(north+1, rad+1):
            peakShape[rad+plusi, rad] = False
            
        for minusi in range(south+1, rad+1):
            peakShape[rad-minusi, rad] = False
            
        for plusj in range(east+1, rad+1):
            peakShape[rad, rad+plusj] = False
            
        for minusj in range(west+1, rad+1):
            peakShape[rad, rad-minusj] = False
               
        #Quadrant 1
        m = -north/float(east)
        for j1 in range(1, rad+1):
            boundUp = m*j1+north
            for i1 in range(1, rad+1):
                if i1 > boundUp:
                    peakShape[rad+i1, rad+j1] = False
                    
        #Quadrant 2
        m = south/float(east)
        for j2 in range(1, rad+1):
            boundDown = m*j2-south
            for i2 in range(-1, -(rad+1), -1):
                if i2 < boundDown:
                    peakShape[rad+i2, rad+j2] = False
        
        #Quadrant 3
        m = -south/float(west)
        for j3 in range(-1, -(rad+1), -1):
            boundDown1 = m*j3-south
            for i3 in range(-1,-(rad+1),-1):
                if i3 < boundDown1:
                    peakShape[rad+i3, rad+j3] = False
        #            
        #Quadrant 4
        m = north/float(west)
        for j4 in range(-1, -(rad+1), -1):
            boundUp1 = m*j4+north
            for i4 in range(1, rad+1):
                if i4 > boundUp1:
                    peakShape[rad+i4, rad+j4] = False
            
        return peakShape

    def templateNray(self, numRays,perimOffset=-0.5,integThresh=0,\
    maxRayLength='square_width',r_i='square_centre',c_i='square_centre'):
        "Return boolean template of peak, created by joining end of N rays"
        """integThresh condition is <=. r_i and c_i are row and column indices
        to cast rays from. perimOffset is perimeter offset - offset between
        vertices and the perimeter of the template outline (passed as radius 
        to Path.contains_points() method)."""
    
        array=self.array
        dim=len(array) #if dim odd, maxRayLength falls one pixel short of 
        #'bottom' and 'right' edges of array
        cent=int(np.floor((dim-1)/2)) #either central pixel if dim is odd or 
        #'top left' of central 4 pixels if dim is even
        
        if maxRayLength=='square_width':
            maxRayLength=cent
        if r_i=='square_centre':
            r_i=cent
        if c_i=='square_centre':
            c_i=cent
        
        vertices=[] #first point at end of each ray where value<=integThresh
        for theta in np.linspace(0, 2*np.pi, numRays, endpoint=False):
            #endpoint=False because ray at 2*pi is same direction as ray at 0
            r=r_i
            c=c_i
            
            incR=np.cos(theta) #increment in Row index - so first ray cast 
            #directly 'down' (when theta==0, cos(theta)=1, sin(theta)=0)
            incC=np.sin(theta) #increment in Column index
            
            for x in range(maxRayLength):
                r+=incR
                c+=incC
                if array[int(np.round(r)), int(np.round(c))]<=integThresh:
                    if (np.round(r), np.round(c)) not in vertices:
                        vertices.append((np.round(r), np.round(c)))
                    break
            else:#this is equivalent to saying the pixel the next step out 
            #would have been below the integThresh
                r+=incR
                c+=incC
                vertices.append((np.round(r), np.round(c)))
                       
        vertices=Path(vertices) #instance of matplotlib.path.Path class,
        #efficiently finds points within arbitrary polygon defined by
        #vertices
        
        points=np.zeros((dim**2, 2), dtype=int)
        points[:,0]=np.repeat(np.arange(0, dim, 1), dim)
        points[:,1]=np.tile(np.arange(0, dim, 1), dim)
        
        points=points[vertices.contains_points(points, radius=perimOffset)] 
        #only choose those points which are inside the polygon traced by the 
        #cast rays. 'radius' kwarg poorly documented but setting to -0.5 draws 
        #polygon inside vertices as required (because vertices are elements 
        #with value <= thresh).
        template=np.zeros((dim, dim), dtype=bool)
        for x in points:
            template[x[0], x[1]]=True
            
        return template
    
    def cutPeak(self, peakTemplate):
        self.array[peakTemplate == False] = 0
    
    def bivSplIntegrate(self, pixWidth=1): # could also have
        #pixWidth=self.fromMap.scan.sliceSize
        mesh_array = np.arange(len(self.array)) * pixWidth 
        spline = interpolate.RectBivariateSpline(mesh_array, mesh_array, \
        self.array)
        #make the bivariate spline, default degree is 3
        return spline.integral(np.nanmin(mesh_array), np.nanmax(mesh_array),\
        np.nanmin(mesh_array), np.nanmax(mesh_array))
        #returns the integral across this spline
        
    def sumIntegrate(self, pixWidth=1):
        # integrate a peak array by summing pixels and dividing by total area
        return self.array.sum()/float(np.product(self.array.shape)*pixWidth**2)
        
    def lineout(self, axis, out=0):
    #the 'x' lineout is the lineout *across* the c dimension, and so the 
    #lineout *along* the m/z indexed by self.r.
        if axis=='x':
            line=self.array[self.pixOut-out:self.pixOut+out+1,\
                            :].sum(0)
        elif axis=='y':
            line=self.array[:,self.pixOut-out:\
                            self.pixOut+out+1].sum(1)
        return line

    def fwhm(self, axis, plot=False, inDa=False, outforlo=0):
        # the 'x' FWHM is the FWHM *across* the c dimension, and so the FWHM 
        # *along* the m/z indexed by self.r. outforlo is 'pixels out for line\
        # out', i.e. take the lineout for the univariate spline going out how
        # many pixels each side?
        line=self.lineout(axis, out=outforlo)
            
        xs=np.arange(self.pixOut*2+1)*(self.fromMap.scan.sliceSize if \
                    inDa else 1.)
        spline=interpolate.UnivariateSpline(xs, line-\
               line[self.pixOut]/2., s=0)
               #s=0 -> spline interpolates through all data points
        roots=spline.roots()
        if plot:
            fig0=plt.figure()
            ax0=fig0.add_subplot(111)
            ax0.plot(xs, line)
            xsfine=np.arange(self.pixOut*2+1, step=0.1)*\
                     (self.fromMap.scan.sliceSize if inDa else 1.)
            ax0.plot(xsfine, spline(xsfine)+line[self.pixOut]/2.) #add half
            #the max back on for the plotting
            ax0.vlines(roots, np.nanmin(line), np.nanmax(line))
            plt.show()
            
        if len(roots)!=2: #return nan if more (or fewer) 
        #than 2 roots are found
            return np.nan
        else:
            return abs(roots[1]-roots[0])
        
class tempPeak(Peak):
    pass #this is so that when you resample you can e.g. integrate and find
    #the CoM if for some reason you would like to.

class CovPeak(Peak):
        
    def __init__(self, fromMap, r, c, pixOut=15):
        self.fromMap = fromMap
        self.r = int(r)
        self.c = int(c)
        self.pixOut = int(pixOut)
        self.build()

    def build(self):
        numScans = self.fromMap.numScans
        self.array = (self.Syx() - self.SySx()/(numScans))/(numScans - 1)
        
    def jkResampleVar(self, cutPeak=True, template=None):
        
        numScans=self.fromMap.numScans
        yScans=self.yScans()
        xScans=self.xScans()
        SyxFull=self.Syx()
        SyFull= self.Sy() 
        SxFull=self.Sx()

        covSum=0
        covSumSqd=0
        
        for missingScan in range(numScans):
            
            Syx = SyxFull - \
            np.array(np.matrix.transpose(np.matrix(yScans[missingScan,:])) \
            * np.matrix(xScans[missingScan,:]))
            
            Sy = SyFull - yScans[missingScan,:]
            Sx = SxFull - xScans[missingScan,:]
            
            SySx = np.matrix.transpose(np.matrix(Sy)) * \
            np.matrix(Sx)

            #Number of scans for each square on the resample
            # = numScans - 1            
            covSquare = Peak((Syx - SySx/(numScans - 1))/(numScans - 2))
            
            if cutPeak:
                covSquare.cutPeak(template)
            
            vol=covSquare.bivSplIntegrate()
                         
            covSum += vol
            covSumSqd += vol**2
            
        return (covSumSqd - (covSum**2)/(numScans))/numScans 
    
    def reCentre(self, maxChange=3, printOut=False):
        """For when you are not entirely sure where the maximum is. Max change
        is the number of pixels you are willing to change x and y by."""
        rf, cf = maxIndices(self.array\
        [self.pixOut-maxChange:self.pixOut+maxChange+1, \
        self.pixOut-maxChange:self.pixOut+maxChange+1])

        if rf == maxChange and cf == maxChange and printOut:
            print('no shift in peak apex')
        elif cf == maxChange:
            if printOut:
                print('r shifted down by '+str(rf-maxChange)+' pixels')
            self.r += rf-maxChange
            self.build()
        elif rf == maxChange:
            if printOut:
                print('c shifted right by '+str(cf-maxChange)+' pixels')
            self.c += cf-maxChange
            self.build()
        else:
            if printOut:
                print('r shifted down by '+str(rf-maxChange)+\
                ' pixels and c shifted right by '+str(cf-maxChange)+' pixels')
            self.r += rf-maxChange
            self.c += cf-maxChange
            self.build()
            
    def reCentre2(self, maxChange=3):
        """For when you are not entirely sure where the maximum is. Max change
        is the number of pixels you are willing to change x and y by. Instead
        of printing, this returns change_in_r,change_in_c and rebuilds the 
        peak array (as well as changing self.r & self.c)"""
        rf, cf = maxIndices(self.array\
        [self.pixOut-maxChange:self.pixOut+maxChange+1, \
        self.pixOut-maxChange:self.pixOut+maxChange+1])

        if rf == maxChange and cf == maxChange: #no need to rebuild in this 
        #case
            return 0,0 #no change in r or c
        else:
            self.r += rf-maxChange
            self.c += cf-maxChange
            self.build()
            return rf-maxChange, cf-maxChange
            
    def Sy(self):
        return self.yScans().sum(axis=0)
    
    def Sx(self):
        return self.xScans().sum(axis=0)
    
    def Syx(self):
        fromIndexRow = self.r - self.pixOut
        toIndexRow = self.r + self.pixOut
        fromIndexCol = self.c - self.pixOut
        toIndexCol = self.c + self.pixOut
        return self.fromMap.syx()[fromIndexRow:toIndexRow + 1,\
        fromIndexCol:toIndexCol + 1]
        
    def SySx(self):
        return np.array(np.matrix.transpose(np.matrix(self.Sy())) * \
        np.matrix(self.Sx()))
        
    def yScans(self):
        fromIndexRow = self.r - self.pixOut
        toIndexRow = self.r + self.pixOut
        return self.fromMap.scan.scanList[:self.fromMap.numScans, \
        fromIndexRow:toIndexRow + 1]
        
    def xScans(self):
        fromIndexCol = self.c - self.pixOut
        toIndexCol = self.c + self.pixOut
        return self.fromMap.scan.scanList[:self.fromMap.numScans, \
        fromIndexCol:toIndexCol + 1]
        
class CCovPeak(CovPeak):

    def build(self, loadMapsIndv=True):
        self.syx=None
        mapsIndv = self.mapsIndv()
        if loadMapsIndv:
            self.mapsIndv=mapsIndv
        else: self.mapsIndv=None
        self.array=np.mean(mapsIndv, axis=2)

    def Syx(self): # overload, it takes a while to load in the Syx
        if self.syx is None:
            fromIndexRow = self.r - self.pixOut
            toIndexRow = self.r + self.pixOut
            fromIndexCol = self.c - self.pixOut
            toIndexCol = self.c + self.pixOut
        
            self.syx = self.fromMap.syxH5(rIdcs=(fromIndexRow,toIndexRow + 1),\
                                  cIdcs=(fromIndexCol,toIndexCol + 1))
        return self.syx

    def Sy(self): # overload
        yScansAll = self.yScans()
        return np.moveaxis(np.array(([yScansAll[self.fromMap.setIdcs[i]].sum(0)\
                       for i in np.arange(len(self.fromMap.setIdcs))])), 0, -1)
        # last axis is the sets
        
    def Sx(self): # overload
        xScansAll = self.xScans()
        return np.moveaxis(np.array(([xScansAll[self.fromMap.setIdcs[i]].sum(0)\
                       for i in np.arange(len(self.fromMap.setIdcs))])), 0, -1)
        # last axis is the sets
            
    def SySx(self): # overload
        return np.einsum('ij,kj->ikj', self.Sy(), self.Sx())
    
    def mapsIndv(self): # individual maps
        return self.Syx() / (self.fromMap.setSize[None,None,:] - 1) - \
                             self.SySx() / (self.fromMap.setSize * \
                            (self.fromMap.setSize - 1))[None,None,:]
    
    def jkResampleVar(self, cutPeak=True, template=None):
        
        numScans=self.fromMap.numScans
        yScans=self.yScans()
        xScans=self.xScans()
        SyxFull=self.Syx()
        SyFull=self.Sy()
        SxFull=self.Sx()

        if self.mapsIndv is None:
            print('calculating mapsIndv within jkResampleVar, not loaded in already')
            self.mapsIndv=self.mapsIndv()
        mapsIndvSum=self.mapsIndv.sum(2)
        
        cCovSum=0
        cCovSumSqd=0
        
        for i, setElim in enumerate(self.fromMap.setIdcs):
            mapIndv_i = self.mapsIndv[:,:,i] # individual map corresponding to this 
            # set
            covSumInc = mapsIndvSum - mapIndv_i # covariance sum, incomplete
            
            Syx_i = SyxFull[:,:,i]
            
            Sy_i = SyFull[:,i] # i indexes the set with scan being eliminated
            Sx_i = SxFull[:,i] # i indexes the set with scan being eliminated
            
            for j, scanElim in enumerate(setElim):
                # j is now indexing the scan being eliminated, within the set
                Syx_j = Syx_i - np.outer(yScans[scanElim], xScans[scanElim])
                
                Sy_j = Sy_i - yScans[scanElim]
                Sx_j = Sx_i - xScans[scanElim]
                
                SySx_j = np.outer(Sy_j, Sx_j)
                
                covYX_j = (Syx_j - (SySx_j / float(len(setElim) - 1))) / \
                         float(len(setElim) - 2)
                
                cCovSquare = Peak((covSumInc + covYX_j) / \
                                  len(self.fromMap.setIdcs))
                if cutPeak:
                    cCovSquare.cutPeak(template)
                
                vol=cCovSquare.bivSplIntegrate()
#                vol=cCovSquare.sumIntegrate()
                             
                cCovSum += vol
                cCovSumSqd += vol**2
            

        return (cCovSumSqd - (cCovSum**2)/(numScans))/numScans

