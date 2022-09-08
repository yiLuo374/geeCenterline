# -*- coding: utf-8 -*-
"""
Main script for geeRiverCl
geeRiverCl: A river centerline extraction and sediment bar identification 
toolbox based on Google Earth Engine
MIT License
Author: Yi Luo, Univ. of Illinois Urbana-Champaign
Contact: yiluo7[at]illinois[dot]edu

"""

# /*
# MIT License
#
# Copyright (c) [2018] [Xiao Yang]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Original Author:  Xiao Yang, UNC Chapel Hill, United States
# Contact: yangxiao@live.unc.edu
# Contributing Authors:
# Tamlin Pavelsky, UNC Chapel Hill, United States
# George Allen, Texas A&M, United States
# Genna Dontchyts, Deltares, NL
#
# NOTE: THIS IS A PRERELEASE VERSION (Edited on: 2019/02/18)
# */

# /* functions to extract river mask */
# GitHub: https://github.com/seanyx/RivWidthCloudPaper

import geemap
import ee
import numpy as np
geemap.ee_initialize()

#load data
def load_image(geeID, bands, names):
    image = ee.Image(geeID).select(bands).rename(names)
    return image

def load_Landsat(geeID, start, end):
    geometry = ee.FeatureCollection(geeID).geometry()
    image = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')\
        .filterBounds(geometry)\
        .filterDate(start, end)\
        .filterMetadata('CLOUD_COVER', 'less_than', 0.5)\
        .mosaic()\
        .clip(geometry)\
        .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'], ['Blue', 'Green', 'Red', 'NIR', 'SWIR1'])
    return image

def load_data(geeID, data_type, bandNames=None, **date): 
    #image: data_type=0; ImageCollection: data_type=1; FeatureCollection: data_type
    #bandNames: rename bands for Image or ImageCollection; i.e. ['Red', 'Green', 'Blue', 'NIR'] for Planet Scope
    if data_type == 0:
        image = ee.Image(geeID)
        if bandNames != None:
            image = image.rename(bandNames)
        return image
    elif data_type == 1:
        imageCollection = ee.ImageCollection(geeID)
        return imageCollection
    elif data_type == 2:
        geometry = ee.FeatureCollection(geeID).geometry()
        start = date['start']
        end = date['end']
        # Generates an Landsat ImageCollection
        image = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')\
                .filterBounds(geometry)\
                .filterDate(start, end)\
                .filterMetadata('CLOUD_COVER', 'less_than', 0.5)\
                .mosaic()\
                .clip(geometry)\
                .select(['B2', 'B3', 'B4', 'B5', 'B6'], ['Blue', 'Green', 'Red', 'NIR', 'SWIR1'])
        return image

def show_image_geemap(image, Map, scale, crs, gamma=1, LayerName='Image'):
    #Map = geemap.Map()
    Map.centerObject(image)
    maxValue = image.reduceRegion(ee.Reducer.max(), bestEffort=True, scale=scale, crs=crs).getInfo()['Blue']
    minValue = image.reduceRegion(ee.Reducer.min(), bestEffort=True, scale=scale, crs=crs).getInfo()['Red']
    img_viz_params = {
        'bands': ['Red', 'Green', 'Blue'],
        'min': minValue,
        'max': maxValue,
        'gamma': gamma
    }
    Map.addLayer(image, img_viz_params, LayerName)
    return()

def geo_to_featureCollection(*args, labelName='class'):
    # args: geometries
    l = []
    n = len(args)
    for i in args:
        l.append(ee.Feature(i, {labelName: (n-1)}))
    fC = ee.FeatureCollection(l)
    return fC

def Evi(image):
    """
    calculate the enhanced vegetation index

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    evi = image.expression('2.5 * (NIR - Red) / (1 + NIR + 6 * Red - 7.5 * Blue)', {
    'NIR': image.select(['NIR']),
    'Red': image.select(['Red']),
    'Blue': image.select(['Blue'])
    })
    return evi.rename(['evi'])

def classification(image, scale, method='minDis'):
    """
    Identify water surface from spectral images

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'minDis'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    crs = image.projection().crs().getInfo()
    ndvi = image.normalizedDifference(['NIR', 'Red'])
    # methods: minDis (minimum distance), Zou(method from Zou[2018]), ModZou(modified from Zou[2018])
    if method == 'minDis':
        #minDis
        #Three land type simple regions: water, plant, sand (other)
        if 'SWIR1' in image.bandNames().getInfo():
            #bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
            bands = ['Blue', 'Green', 'Red', 'NIR']
        else: 
            bands = ['Blue', 'Green', 'Red', 'NIR']
        labels = 'samples'
        water = ndvi.lt(0)
        sand = ndvi.lt(0.1).And(ndvi.gt(0)).multiply(2)
        vege = ndvi.gt(0.3).multiply(3) #0.6
        wsv = water.add(sand.add(vege)).selfMask().rename('samples')
        samples = wsv.stratifiedSample(**{'numPoints': 100, 'projection': crs, 'scale':30, 'geometries': True})
        training = image.select(bands).sampleRegions(**{
                    'collection': samples,
                    'properties': ['samples'],
                    'scale' : scale
                })
        trained = ee.Classifier.minimumDistance().train(training, 'samples', bands)
        classified = image.select(bands).classify(trained).reproject(crs = crs, scale = scale)
        return classified
    
    if method == 'RF':
        #minDis
        #Three land type simple regions: water, plant, sand (other)
        if 'SWIR1' in image.bandNames().getInfo():
            #bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
            bands = ['Blue', 'Green', 'Red', 'NIR']
        else: 
            bands = ['Blue', 'Green', 'Red', 'NIR']
        labels = 'samples'
        water = ndvi.lt(0)
        sand = ndvi.lt(0.1).And(ndvi.gt(0)).multiply(2)
        vege = ndvi.gt(0.3).multiply(3) #0.6
        wsv = water.add(sand.add(vege)).selfMask().rename('samples')
        samples = wsv.stratifiedSample(**{'numPoints': 100, 'projection': crs, 'scale':30, 'geometries': True})
        training = image.select(bands).sampleRegions(**{
                    'collection': samples,
                    'properties': ['samples'],
                    'scale' : scale
                })
        trained = ee.Classifier.smileRandomForest(10).train(training, 'samples', bands)
        classified = image.select(bands).classify(trained).reproject(crs = crs, scale = scale)
        return classified
    
    elif method == 'Zou':
        #crs = image.projection().crs().getInfo()
        #ndvi = image.normalizedDifference(['NIR', 'Red'])#.rename('ndvi')
        mndwi = image.normalizedDifference(['Green', 'SWIR1'])#.rename('mndwi')
        evi = Evi(image)
        water_mask = (mndwi.gt(ndvi).Or(mndwi.gt(evi))).And(evi.lt(0.2)).reproject(crs = crs, scale = scale)
        return water_mask
    
    elif method == 'ModZou':
        #crs = image.projection().crs().getInfo()
        #ndvi = image.normalizedDifference(['NIR', 'Red']).rename('ndvi')
        ndwi = image.normalizedDifference(['Green', 'NIR']).rename('ndwi')
        evi = Evi(image)
        water_mask = (ndwi.gt(ndvi).Or(ndwi.gt(evi))).And(evi.lt(0.2)).reproject(crs = crs, scale = scale)
        return water_mask

def classification1(image, scale, method='minDis'):
    """
    Identify water surface from spectral images

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'minDis'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    crs = image.projection().crs().getInfo()
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndwi = image.normalizedDifference(['Green', 'NIR']).rename('NDWI')
    evi = Evi(image).rename('EVI')
    image = image.addBands(ndvi).addBands(evi)
    # methods: minDis (minimum distance), Zou(method from Zou[2018]), ModZou(modified from Zou[2018])
    if method == 'minDis':
        #minDis
        #Three land type simple regions: water, plant, sand (other)
        if 'SWIR1' in image.bandNames().getInfo():
            bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
        else: 
            bands = ['Blue', 'Green', 'Red', 'NIR']
        labels = 'samples'
        water = ndvi.lt(0.2)
        sand = ndvi.lt(0.2).And(ndvi.gt(0.1)).And(evi.lt(0.2)).multiply(2)
        vege = ndvi.gt(0.3).And(evi.gt(0.2)).multiply(3) #0.6
        wsv = water.add(sand.add(vege)).selfMask().rename('samples')
        samples = wsv.stratifiedSample(**{'numPoints': 100, 'projection': crs, 'scale':30, 'geometries': True})
        training = image.select(bands).sampleRegions(**{
                    'collection': samples,
                    'properties': ['samples'],
                    'scale' : scale
                })
        trained = ee.Classifier.minimumDistance().train(training, 'samples', bands)
        classified = image.select(bands).classify(trained).reproject(crs = crs, scale = scale)
        return classified
    
    if method == 'RF':
        #minDis
        #Three land type simple regions: water, plant, sand (other)
        if 'SWIR1' in image.bandNames().getInfo():
            #bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
            bands = ['Blue', 'Green', 'Red', 'NIR']
        else: 
            bands = ['Blue', 'Green', 'Red', 'NIR']
        labels = 'samples'
        water = ndvi.lt(0.2)
        sand = ndvi.lt(0.2).And(ndvi.gt(0.1)).And(evi.lt(0.2)).multiply(2)
        vege = ndvi.gt(0.3).And(evi.gt(0.2)).multiply(3) #0.6
        wsv = water.add(sand.add(vege)).selfMask().rename('samples')
        samples = wsv.stratifiedSample(**{'numPoints': 100, 'projection': crs, 'scale':30, 'geometries': True})
        training = image.select(bands).sampleRegions(**{
                    'collection': samples,
                    'properties': ['samples'],
                    'scale' : scale
                })
        trained = ee.Classifier.smileRandomForest(100).train(training, 'samples', bands)
        classified = image.select(bands).classify(trained).reproject(crs = crs, scale = scale)
        return classified
    
    if method == 'RF1':
        #minDis
        #Three land type simple regions: water, plant, sand (other)
        if 'SWIR1' in image.bandNames().getInfo():
            #bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
            bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
        else: 
            bands = ['Blue', 'Green', 'Red', 'NIR']
        labels = 'samples'
        water = ndvi.lt(-0.2).And(evi.lt(0.2))
        sand = ndvi.lt(0.2).And(ndvi.gt(0.1)).And(evi.lt(0.2)).multiply(2)
        vege = ndvi.gt(0.3).And(evi.gt(0.2)).multiply(3) #0.6
        wsv = water.add(sand.add(vege)).selfMask().rename('samples')
        samples = wsv.stratifiedSample(**{'numPoints': 100, 'projection': crs, 'scale':30, 'geometries': True})
        training = image.select(bands).sampleRegions(**{
                    'collection': samples,
                    'properties': ['samples'],
                    'scale' : scale
                })
        trained = ee.Classifier.smileRandomForest(100).train(training, 'samples', bands)
        classified = image.select(bands).classify(trained).reproject(crs = crs, scale = scale)
        return classified
    
    elif method == 'Zou':
        #crs = image.projection().crs().getInfo()
        #ndvi = image.normalizedDifference(['NIR', 'Red'])#.rename('ndvi')
        mndwi = image.normalizedDifference(['Green', 'SWIR1'])#.rename('mndwi')
        evi = Evi(image)
        water_mask = (mndwi.gt(ndvi).Or(mndwi.gt(evi))).And(evi.lt(0.2)).reproject(crs = crs, scale = scale)
        return water_mask
    
    elif method == 'ModZou':
        #crs = image.projection().crs().getInfo()
        #ndvi = image.normalizedDifference(['NIR', 'Red']).rename('ndvi')
        ndwi = image.normalizedDifference(['Green', 'NIR']).rename('ndwi')
        evi = Evi(image)
        water_mask = (ndwi.gt(ndvi).Or(ndwi.gt(evi))).And(evi.lt(0.2)).reproject(crs = crs, scale = scale)
        return water_mask
    
def classification2(image, scale, method='minDis'):
    """
    Identify water surface from spectral images

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'minDis'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    crs = image.projection().crs().getInfo()
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndwi = image.normalizedDifference(['Green', 'NIR']).rename('NDWI')
    evi = Evi(image).rename('EVI')
    image = image.addBands(ndvi).addBands(evi)
    # methods: minDis (minimum distance), Zou(method from Zou[2018]), ModZou(modified from Zou[2018])
    if method == 'minDis':
        #minDis
        #Three land type simple regions: water, plant, sand (other)
        if 'SWIR1' in image.bandNames().getInfo():
            bands = ['Blue', 'Green', 'Red', 'NIR']
        else: 
            bands = ['Blue', 'Green', 'Red', 'NIR']
        labels = 'samples'
        water = evi.lt(0)
        sand = evi.gt(0).And(evi.lt(0.2)).multiply(2)
        vege = evi.gt(0.2).multiply(3) #0.6
        wsv = water.add(sand.add(vege)).selfMask().rename('samples')
        samples = wsv.stratifiedSample(**{'numPoints': 100, 'projection': crs, 'scale':30, 'geometries': True})
        training = image.select(bands).sampleRegions(**{
                    'collection': samples,
                    'properties': ['samples'],
                    'scale' : scale
                })
        trained = ee.Classifier.minimumDistance().train(training, 'samples', bands)
        classified = image.select(bands).classify(trained).reproject(crs = crs, scale = scale)
        return classified
    
    if method == 'RF':
        #minDis
        #Three land type simple regions: water, plant, sand (other)
        if 'SWIR1' in image.bandNames().getInfo():
            #bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
            bands = ['Blue', 'Green', 'Red', 'NIR']
        else: 
            bands = ['Blue', 'Green', 'Red', 'NIR']
        labels = 'samples'
        water = evi.lt(0)
        sand = evi.gt(0).And(evi.lt(0.2)).multiply(2)
        vege = evi.gt(0.2).multiply(3) #0.6
        wsv = water.add(sand.add(vege)).selfMask().rename('samples')
        samples = wsv.stratifiedSample(**{'numPoints': 100, 'projection': crs, 'scale':30, 'geometries': True})
        training = image.select(bands).sampleRegions(**{
                    'collection': samples,
                    'properties': ['samples'],
                    'scale' : scale
                })
        trained = ee.Classifier.smileRandomForest(100).train(training, 'samples', bands)
        classified = image.select(bands).classify(trained).reproject(crs = crs, scale = scale)
        return classified
    
    elif method == 'Zou':
        #crs = image.projection().crs().getInfo()
        #ndvi = image.normalizedDifference(['NIR', 'Red'])#.rename('ndvi')
        mndwi = image.normalizedDifference(['Green', 'SWIR1'])#.rename('mndwi')
        evi = Evi(image)
        water_mask = (mndwi.gt(ndvi).Or(mndwi.gt(evi))).And(evi.lt(0.2)).reproject(crs = crs, scale = scale)
        return water_mask
    
    elif method == 'ModZou':
        #crs = image.projection().crs().getInfo()
        #ndvi = image.normalizedDifference(['NIR', 'Red']).rename('ndvi')
        ndwi = image.normalizedDifference(['Green', 'NIR']).rename('ndwi')
        evi = Evi(image)
        water_mask = (ndwi.gt(ndvi).Or(ndwi.gt(evi))).And(evi.lt(0.2)).reproject(crs = crs, scale = scale)
        return water_mask
    
def close(image, scale, radius=1.5, kernelType='circle', units='pixels', iterations=1, kernel=None):
    """
    Fill river mask gaps
    
    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.
    radius : TYPE, optional
        DESCRIPTION. The default is 1.5.
    kernelType : TYPE, optional
        DESCRIPTION. The default is 'circle'.
    units : TYPE, optional
        DESCRIPTION. The default is 'pixels'.
    iterations : TYPE, optional
        DESCRIPTION. The default is 1.
    kernel : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    image1 : TYPE
        DESCRIPTION.

    """
    crs = image.projection().crs().getInfo()
    image1 = image.focal_max(radius, kernelType, units, iterations, kernel).reproject(crs = crs, scale = scale)\
        .focal_min(radius, kernelType, units, iterations, kernel).reproject(crs = crs, scale = scale)
    return image1
    
#def connect(image, scale, radius, iterations):
#    crs = image.projection().crs().getInfo()
#    water_close = close(image, radius, iterations = iterations).reproject(crs = crs, scale #= scale)
#    return water_close

def noise_removal(image, scale, maxArea, connectedness):
    """
    remove unwanted water masks

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.
    maxArea : TYPE
        DESCRIPTION.
    connectedness : TYPE
        DESCRIPTION.

    Returns
    -------
    image2 : TYPE
        DESCRIPTION.

    """
    crs = image.projection().crs().getInfo()
    image_mask = image.selfMask()
    if connectedness == 4:
        k = [[0,1,0], [1,1,1], [0,1,0]]
    elif connectedness == 8:
        k = [[1,1,1], [1,1,1], [1,1,1]]
    image1 = image_mask.connectedComponents(ee.Kernel.fixed(3, 3, k), maxArea).\
        select('labels').reproject(crs = crs, scale = scale)
    image2 = image.add(image1.eq(0).eq(0).unmask()).eq(1).selfMask()
    return image2

def small_noise_removal(mask0, scale, area):
    # area: the maximum centerbar to identify, <= 1024
    crs = mask0.projection().crs().getInfo()
    mask1 = mask0.connectedPixelCount(area, True).reproject(crs = crs, scale = scale)
    mask2 = mask1.eq(area).add(mask0).eq(2)
    return mask2

def large_centerbar_removal(mask0, scale, area, bandName):
    # area: the maximum centerbar to identify, small_centerbar_removal should be used for area <=1024
    mask0 = mask0.Not()
    crs = mask0.projection().crs().getInfo()
    if bandName == None:
        bandName = mask0.bandNames().getInfo()[0]
    mask = mask0.focal_min(1, kernelType='square')\
        .reproject(crs = crs, scale = scale)
    mask1 = mask.addBands(mask.eq(0))\
        .reduceConnectedComponents(ee.Reducer.median(), bandName, area)\
        .eq(0).unmask()\
        .reproject(crs = crs, scale = scale)
    mask2 = mask1.focal_max(1, kernelType='square')\
        .reproject(crs = crs, scale = scale)
    return mask2

def small_centerbar_removal(mask0, scale, area):
    # area: the maximum centerbar to identify, <= 1024
    mask = mask0.Not()
    crs = mask.projection().crs().getInfo()
    mask1 = mask.connectedPixelCount(area, False).reproject(crs = crs, scale = scale)
    mask2 = mask1.lt(area).selfMask().unmask()
    mask3 = mask0.Or(mask2)
    return mask3

#def mask_small_area(imageH, imageL):
    


def centerbar_removal(mask0, scale, area, bandName = None):
    if area < 1024:
        mask = small_centerbar_removal(mask0, scale, area)
    else:
        mask = large_centerbar_removal(mask0, scale, area, bandName)
    return mask

def hitOrMiss(image, se1, se2):
    """
    perform hitOrMiss transform

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    se1 : TYPE
        DESCRIPTION.
    se2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    e1 = image.reduceNeighborhood(ee.Reducer.min(), se1)
    e2 = image.Not().reduceNeighborhood(ee.Reducer.min(), se2)
    return e1.And(e2)

def splitKernel(kernel, value):
    """
    recalculate the kernel according to the given foreground value

    Parameters
    ----------
    kernel : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    kernel = np.array(kernel)
    result = kernel
    r = 0
    while (r < kernel.shape[0]):
        c = 0
        while (c < kernel.shape[1]):
            if kernel[r][c] == value:
                result[r][c] = 1
            else:
                result[r][c] = 0
            c = c + 1
        r = r + 1
    return result.tolist()

def Skeletonize(image, iterations, method):
    """
    perform skeletonization

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    iterations : TYPE
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    se1w = [[2, 2, 2], [0, 1, 0], [1, 1, 1]]

    if (method == 2):
        se1w = [[2, 2, 2], [0, 1, 0], [0, 1, 0]]

    se11 = ee.Kernel.fixed(3, 3, splitKernel(se1w, 1))
    se12 = ee.Kernel.fixed(3, 3, splitKernel(se1w, 2))

    se2w = [[2, 2, 0], [2, 1, 1], [0, 1, 0]]

    if (method == 2):
        se2w = [[2, 2, 0], [2, 1, 1], [0, 1, 1]]

    se21 = ee.Kernel.fixed(3, 3, splitKernel(se2w, 1))
    se22 = ee.Kernel.fixed(3, 3, splitKernel(se2w, 2))

    result = image

    i = 0
    while (i < iterations):
        j = 0
        while (j < 4):
            result = result.subtract(hitOrMiss(result, se11, se12))
            se11 = se11.rotate(1)
            se12 = se12.rotate(1)
            result = result.subtract(hitOrMiss(result, se21, se22))
            se21 = se21.rotate(1)
            se22 = se22.rotate(1)
            j = j + 1
        i = i + 1

    return result.rename(['clRaw'])


def CalcDistanceMap(img, neighborhoodSize, scale):
    """
    assign each river pixel with the distance (in meter) between itself and the closest non-river pixel

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    neighborhoodSize : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    imgD2 = img.focal_max(1.5, 'circle', 'pixels', 2)
    imgD1 = img.focal_max(1.5, 'circle', 'pixels', 1)
    outline = imgD2.subtract(imgD1)

    dpixel = outline.fastDistanceTransform(neighborhoodSize).sqrt()
    dmeters = dpixel.multiply(scale) #// for a given scale
    DM = dmeters.mask(dpixel.lte(neighborhoodSize).And(imgD2))

    return(DM)

def CalcGradientMap(image, gradMethod, scale):
    """
    Calculate the gradient

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    gradMethod : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if (gradMethod == 1): # GEE .gradient() method
        grad = image.gradient()
        dx = grad.select(['x'])
        dy = grad.select(['y'])
        g = dx.multiply(dx).add(dy.multiply(dy)).sqrt()

    if (gradMethod == 2): # Gena's method
        k_dx = ee.Kernel.fixed(3, 3, [[ 1.0/8, 0.0, -1.0/8], [ 2.0/8, 0.0, -2.0/8], [ 1.0/8,  0.0, -1.0/8]])
        k_dy = ee.Kernel.fixed(3, 3, [[ -1.0/8, -2.0/8, -1.0/8], [ 0.0, 0.0, 0.0], [ 1.0/8, 2.0/8, 1.0/8]])
        dx = image.convolve(k_dx)
        dy = image.convolve(k_dy)
        g = dx.multiply(dx).add(dy.multiply(dy)).divide(scale**2).sqrt()

    if (gradMethod == 3): # RivWidth method
        k_dx = ee.Kernel.fixed(3, 1, [[-0.5, 0.0, 0.5]])
        k_dy = ee.Kernel.fixed(1, 3, [[0.5], [0.0], [-0.5]])
        dx = image.convolve(k_dx)
        dy = image.convolve(k_dy)
        g = dx.multiply(dx).add(dy.multiply(dy)).divide(scale.multiply(scale))

    return(g)

def CalcOnePixelWidthCenterline(img, GM, hGrad):
    """
    calculate the 1px centerline from:
    1. fast distance transform of the river banks
    2. gradient of the distance transform, mask areas where gradient greater than a threshold hGrad
    3. apply skeletonization twice to get a 1px centerline
    thresholding gradient map inspired by Pavelsky and Smith., 2008

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    GM : TYPE
        DESCRIPTION.
    hGrad : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    imgD2 = img.focal_max(1.5, 'circle', 'pixels', 2)
    cl = ee.Image(GM).mask(imgD2).lte(hGrad).And(img)
    # // apply skeletonization twice
    cl1px = Skeletonize(cl, 2, 1)
    return(cl1px)

def ExtractEndpoints(CL1px):
    """
    calculate end points in the one pixel centerline

    Parameters
    ----------
    CL1px : TYPE
        DESCRIPTION.

    Returns
    -------
    endpoints : TYPE
        DESCRIPTION.

    """
    clnot = CL1px.unmask().Not()
    k2 = ee.Kernel.fixed(3, 3, [[1, 1, 1], [1, 0, 1], [0, 0, 1]])
    k3 = ee.Kernel.fixed(3, 3, [[1, 1, 1], [1, 0, 1], [1, 0, 0]])
    
    result = ee.Image()
    
    # // the for loop removes the identified endpoints from the imput image
    i = 0
    while (i<4): # rotate kernels
        result = result.addBands(clnot.reduceNeighborhood(ee.Reducer.sum(), k2).rename(str(i)+'1'))
        result = result.addBands(clnot.reduceNeighborhood(ee.Reducer.sum(), k3).rename(str(i)+'1'))
        k2 = k2.rotate(1)
        k3 = k3.rotate(1)
        i = i + 1
    
    result = result.select(result.bandNames().getInfo()[1:])
    result1 = result.reduce('max').eq(6)
    endpoints = result1.And(CL1px).selfMask()
    return endpoints

def ExtractCorners(CL1px):
    """
    calculate corners in the one pixel centerline

    Parameters
    ----------
    CL1px : TYPE
        DESCRIPTION.

    Returns
    -------
    cornerPoints : TYPE
        DESCRIPTION.

    """

    se1w = [[0, 0, 1], [1, 1, 1], [0, 1, 0]]

    se11 = ee.Kernel.fixed(3, 3, splitKernel(se1w, 1))
    se12 = ee.Kernel.fixed(3, 3, splitKernel(se1w, 2))

    result = CL1px
    # // the for loop removes the identified corners from the imput image

    i = 0
    while(i < 4): # rotate kernels

        result = result.subtract(hitOrMiss(result, se11, se12))

        se11 = se11.rotate(1)
        se12 = se12.rotate(1)

        i = i + 1

    cornerPoints = CL1px.subtract(result)
    return cornerPoints

def ExtractJoints(CL1px):
    """
    identify joints from 1px wide centerline

    Parameters
    ----------
    CL1px : TYPE
        DESCRIPTION.

    Returns
    -------
    jts : TYPE
        DESCRIPTION.

    """
    k4 = ee.Kernel.fixed(3, 3, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    k5 = ee.Kernel.fixed(3, 3, [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    k6 = ee.Kernel.fixed(3, 3, [[0, 1, 0], [0, 1, 1], [1, 0, 0]])
    k7 = ee.Kernel.fixed(3, 3, [[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    
    i = 0
    jt1 = ee.Image()
    jt2 = ee.Image()
    while (i<4): # rotate kernels
        jt1 = jt1.addBands(CL1px.reduceNeighborhood(ee.Reducer.sum(), k6).rename(str(i)))
        k6 = k6.rotate(1)
        jt2 = jt2.addBands(CL1px.reduceNeighborhood(ee.Reducer.sum(), k7).rename(str(i)))
        k7 = k7.rotate(1)
        i = i + 1
    jt1 = jt1.select(jt1.bandNames().getInfo()[1:])
    jt2 = jt2.select(jt2.bandNames().getInfo()[1:])
    jt11 = jt1.reduce('max').eq(4)
    jt21 = jt2.reduce('max').eq(4)
    jts1 = jt11.Or(jt21)
    jts2 = CL1px.reduceNeighborhood(ee.Reducer.sum(), k4).gte(4).Or(CL1px.reduceNeighborhood(ee.Reducer.sum(), k5).gte(4)).selfMask()
    jts = jts2.unmask().Or(jts1).selfMask()
    
    return jts

def ExtractJoints1(CL1px):
    """
    identify joints from 1px wide centerline

    Parameters
    ----------
    CL1px : TYPE
        DESCRIPTION.

    Returns
    -------
    jts : TYPE
        DESCRIPTION.

    """
    k1 = ee.Kernel.fixed(3, 3, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    jts = CL1px.reduceNeighborhood(ee.Reducer.sum(), k1).gte(4).selfMask()
    
    return jts

def CleanCenterline(cl1px, maxBranchLengthToRemove, scale, iterate=5):
    """
    clean the 1px centerline:
	1. remove branches
	2. remove corners to insure 1px width (optional)

    Parameters
    ----------
    cl1px : TYPE
        DESCRIPTION.
    maxBranchLengthToRemove : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.
    iterate : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    cl1px1 : TYPE
        DESCRIPTION.

    """   
    cl1px1 = cl1px
    for i in range(iterate):
        endsByNeighbors = ExtractEndpoints(cl1px1)
        joints = ExtractJoints1(cl1px1)
        sep = cl1px1.add(joints.unmask().focal_max(1.5, "square")).eq(1).selfMask()
        costMap = (sep.cumulativeCost(
            source = endsByNeighbors,
            maxDistance = maxBranchLengthToRemove,
            geodeticDistance = True))
        branchMask = costMap.gte(0).unmask(0)
        cl1px1 = cl1px1.updateMask(branchMask.Not())
    return (cl1px1)

def CalculateCenterline1(imgIn, scale, thre=0.7):

    crs = imgIn.projection().crs().getInfo()
    distM = CalcDistanceMap(imgIn, 60, scale).reproject(crs = crs, scale = scale)
    gradM = CalcGradientMap(distM, 2, scale).reproject(crs = crs, scale = scale)
    cl1 = CalcOnePixelWidthCenterline(imgIn, gradM, thre).reproject(crs = crs, scale = scale)
    return(cl1)

def CalculateCenterline(imgIn, scale, thre=0.9):
    """
    obtain the 1px wide river centerline from river mask

    Parameters
    ----------
    imgIn : ee.Image
        river mask. 1 represents river pixels and 0 represents others.
    scale : int
        scale of raster.
    thre : TYPE, optional
        A value between 0.7-0.9 to thresold centerline. The default is 0.9.

    Returns
    -------
    None.

    """

    distM = CalcDistanceMap(imgIn, 60, scale)
    gradM = CalcGradientMap(distM, 2, scale)
    cl1 = CalcOnePixelWidthCenterline(imgIn, gradM, thre)
    k1 = ee.Kernel.fixed(3, 3, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    clEC = cl1.selfMask().reduceNeighborhood(ee.Reducer.sum(), k1).lte(5).selfMask()
    cl1 = noise_removal(clEC, 3, 500, 8)
    return(cl1)

def trim_centerline(imgIn, scale, length, iterate=5):
    """
    Trim branches from 1px wide centerline

    Parameters
    ----------
    imgIn : ee.Image
        1px wide centerline. The value of centerline is 1.
    scale : int
        scale of raster.
    length : int
        the max distance to be trimed.
    iterate : int
        iteration to trim branches.

    Returns
    -------
    None.

    """
    
    trimed = CleanCenterline(imgIn, length, scale, iterate)
    ep1 = ExtractEndpoints(trimed)
    ep2 = trimed.unmask().add(ep1.unmask()).eq(1)
    ep3 = Skeletonize(ep2.unmask().focal_max().focal_min(), 1, 1)
    
    return(ep3)

