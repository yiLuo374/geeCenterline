import ee
import numpy as np

# -*- coding: utf-8 -*-
"""
Main script for geeRiverCL
geeRiverCl: A river centerline extraction and sediment bar identification 
toolbox based on Google Earth Engine
MIT License
Author: Yi Luo, Univ. of Illinois Urbana-Champaign
Contact: yiluo7[at]illinois[dot]edu

"""

def load_file(assetID, bands=None, bandnames=None):
    tp = ee.data.getAsset(assetID)['type']
    
    if tp == 'TABLE':
        return load_roi(assetID)
    
    if tp == 'IMAGE':
        return load_image(assetID, bands, bandnames)

def batch_load(IDlist, bands, bandnames):
    uris = ee.List(IDlist)
    images = uris.map(ee.Image.loadGeoTIFF).select(bands).rename(bandnames)
    collection = ee.ImageCollection(images)
    return collection

def load_image(assetID, bands, names):
    image = ee.Image(assetID).select(bands).rename(names)
    return image

def load_roi(assetID):
    roi = ee.FeatureCollection(assetID).geometry()
    return roi

def classification(image, thr2=0, thr3=0.2, thr4=0.3, thr5=0.6, numPts=100):
    crs = image.projection().crs().getInfo()
    ndvi = image.normalizedDifference(['NIR', 'Red'])
    scale = image.projection().nominalScale().getInfo()
    #minDis
    #Three land type simple regions: water, plant, sand (other)
    bands = image.bandNames().getInfo()
    labels = 'samples'
    water = ndvi.lt(thr2)
    sand = ndvi.lt(thr4).And(ndvi.gt(thr3)).multiply(2)
    vege = ndvi.gt(thr5).multiply(3)
    wsv = water.add(sand.add(vege)).selfMask().rename('samples')
    samples = wsv.stratifiedSample(**{'numPoints': numPts, 'projection': crs, 'scale':30, 'geometries': True})
    training = image.select(bands).sampleRegions(**{
                'collection': samples,
                'properties': ['samples'],
                'scale' : scale
            })
    trained = ee.Classifier.minimumDistance().train(training, 'samples', bands)
    classified = image.select(bands).classify(trained)
    return classified

def close(image, radius=1.5, kernelType='circle', units='pixels', iterations=1, kernel=None):
    crs = image.projection().crs().getInfo()
    scale = image.projection().nominalScale()
    image1 = image.focal_max(radius, kernelType, units, iterations, kernel).reproject(crs = crs, scale = scale)\
        .focal_min(radius, kernelType, units, iterations, kernel).reproject(crs = crs, scale = scale)
    return image1

def noise_removal(image, maxArea, connectedness):
    crs = image.projection().crs().getInfo()
    scale = image.projection().nominalScale()
    image_mask = image.selfMask()
    if connectedness == 4:
        k = [[0,1,0], [1,1,1], [0,1,0]]
    elif connectedness == 8:
        k = [[1,1,1], [1,1,1], [1,1,1]]
    image1 = image_mask.connectedComponents(ee.Kernel.fixed(3, 3, k), maxArea).\
        select('labels').reproject(crs = crs, scale = scale)
    image2 = image.add(image1.eq(0).eq(0).unmask()).eq(1).selfMask()
    return image2

# river mask from landsat
def mask_hrs(image, r=1.5, i=3):
    crs = image.projection().crs().getInfo()
    scale = image.projection().nominalScale()
    mask = image.focal_max(radius = r, iterations = i).reproject(crs = crs, scale = scale)
    return mask
             
# /*
# Copyright (c) [2018] [Xiao Yang]
#
# Original Author:  Xiao Yang, UNC Chapel Hill, United States
# Contact: yangxiao@live.unc.edu
# Contributing Authors:
# Tamlin Pavelsky, UNC Chapel Hill, United States
# George Allen, Texas A&M, United States
# Genna Dontchyts, Deltares, NL
# */
 
def hitOrMiss(image, se1, se2):
    e1 = image.reduceNeighborhood(ee.Reducer.min(), se1)
    e2 = image.Not().reduceNeighborhood(ee.Reducer.min(), se2)
    return e1.And(e2)

def splitKernel(kernel, value):
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
    imgD2 = img.focal_max(1.5, 'circle', 'pixels', 2)
    imgD1 = img.focal_max(1.5, 'circle', 'pixels', 1)
    outline = imgD2.subtract(imgD1)

    dpixel = outline.fastDistanceTransform(neighborhoodSize).sqrt()
    dmeters = dpixel.multiply(scale) #// for a given scale
    DM = dmeters.mask(dpixel.lte(neighborhoodSize).And(imgD2))

    return DM

def CalcGradientMap(image, gradMethod, scale):
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

    return g

def CalcOnePixelWidthCenterline(img, GM, hGrad):
    imgD2 = img.focal_max(1.5, 'circle', 'pixels', 2)
    cl = ee.Image(GM).mask(imgD2).lte(hGrad).And(img)
    # // apply skeletonization twice
    cl1px = Skeletonize(cl, 2, 1)
    return cl1px

def ExtractEndpoints(CL1px):
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

def CleanCenterline(cl1px, maxBranchLengthToRemove, scale, iterate=5):
    
    cl1px1 = cl1px
    for i in range(iterate):
        endsByNeighbors = ExtractEndpoints(cl1px1)
        joints = ExtractJoints(cl1px1)
        sep = cl1px1.add(joints.unmask().focal_max(1.5, "square")).eq(1).selfMask()
        costMap = (sep.cumulativeCost(
            source = endsByNeighbors,
            maxDistance = maxBranchLengthToRemove,
            geodeticDistance = True))
        branchMask = costMap.gte(0).unmask(0)
        cl1px1 = cl1px1.updateMask(branchMask.Not())
    return cl1px1

def CalculateCenterline(imgIn, thre=0.7):
    scale = imgIn.projection().nominalScale().getInfo()
    crs = imgIn.projection().crs().getInfo()
    distM = CalcDistanceMap(imgIn, 60, scale).reproject(crs = crs, scale = scale)
    gradM = CalcGradientMap(distM, 2, scale).reproject(crs = crs, scale = scale)
    cl1 = CalcOnePixelWidthCenterline(imgIn, gradM, thre).reproject(crs = crs, scale = scale)
    return cl1

def CalculateAngle(clCleaned):
    """calculate the orthogonal direction of each pixel of the centerline
    """

    w3 = (ee.Kernel.fixed(9, 9, [
    [135.0, 126.9, 116.6, 104.0, 90.0, 76.0, 63.4, 53.1, 45.0],
    [143.1, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 36.9],
    [153.4, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 26.6],
    [166.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 14.0],
    [180.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 1e-5],
    [194.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 346.0],
    [206.6, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 333.4],
    [216.9, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 323.1],
    [225.0, 233.1,  243.4,  256.0,  270.0,  284.0,  296.6,  306.9, 315.0]]))

    combinedReducer = ee.Reducer.sum().combine(ee.Reducer.count(), None, True)

    clAngle = (clCleaned.mask(clCleaned)
        .rename(['clCleaned'])
        .reduceNeighborhood(
        reducer = combinedReducer,
        kernel = w3,
        inputWeight = 'kernel',
        skipMasked = True))

	## mask calculating when there are more than two inputs into the angle calculation
    clAngleNorm = (clAngle
        .select('clCleaned_sum')
        .divide(clAngle.select('clCleaned_count'))
        .mask(clAngle.select('clCleaned_count').gt(2).Not()))

	## if only one input into the angle calculation, rotate it by 90 degrees to get the orthogonal
    clAngleNorm = (clAngleNorm
        .where(clAngle.select('clCleaned_count').eq(1), clAngleNorm.add(ee.Image(90))))

    return clAngleNorm.rename(['orthDegree'])

def GetWidth(clAngleNorm, segmentInfo, endInfo, DM, crs, bound, scale, sceneID, note):
    """calculate the width of the river at each centerline pixel, measured according to the orthgonal direction of the river
    """
    def GetXsectionEnds(f):
        xc = ee.Number(f.get('x'))
        yc = ee.Number(f.get('y'))
        orthRad = ee.Number(f.get('angle')).divide(180).multiply(math.pi)

        width = ee.Number(f.get('toBankDistance')).multiply(1.5)
        cosRad = width.multiply(orthRad.cos())
        sinRad = width.multiply(orthRad.sin())
        p1 = ee.Geometry.Point([xc.add(cosRad), yc.add(sinRad)], crs)
        p2 = ee.Geometry.Point([xc.subtract(cosRad), yc.subtract(sinRad)], crs)

        xlEnds = (ee.Feature(ee.Geometry.MultiPoint([p1, p2]).buffer(30), {
            'xc': xc,
            'yc': yc,
            'longitude': f.get('lon'),
            'latitude': f.get('lat'),
            'orthogonalDirection': orthRad,
            'MLength': width.multiply(2),
            'p1': p1,
            'p2': p2,
            'crs': crs,
            'image_id': sceneID,
            'note': note
            }))

        return xlEnds

    def SwitchGeometry(f):
        return (f
        .setGeometry(ee.Geometry.LineString(coords = [f.get('p1'), f.get('p2')], proj = crs, geodesic = False))
        .set('p1', None).set('p2', None)) # remove p1 and p2

    ## convert centerline image to a list. prepare for map function
    clPoints = (clAngleNorm.rename(['angle'])
    	.addBands(ee.Image.pixelCoordinates(crs))
        .addBands(ee.Image.pixelLonLat().rename(['lon', 'lat']))
        .addBands(DM.rename(['toBankDistance']))
        .sample(
            region = bound,
            scale = scale,
            projection = None,
            factor = 1,
            dropNulls = True
        ))

	## calculate the cross-section lines, returning a featureCollection
    xsectionsEnds = clPoints.map(GetXsectionEnds)

	## calculate the flags at the xsection line end points
    endStat = (endInfo.reduceRegions(
        collection = xsectionsEnds,
        reducer = ee.Reducer.anyNonZero().combine(ee.Reducer.count(), None, True), # test endpoints type
        scale = scale,
        crs = crs))

	## calculate the width of the river and other flags along the xsection lines
    xsections1 = endStat.map(SwitchGeometry)
    combinedReducer = ee.Reducer.mean()
    xsections = (segmentInfo.reduceRegions(
        collection = xsections1,
        reducer = combinedReducer,
        scale = scale,
        crs = crs))

    return xsections

def CalculateOrthAngle(imgIn):
    cl1px = imgIn.select(['cleanedCL'])
    angle = CalculateAngle(cl1px)
    imgOut = imgIn.addBands(angle)
    return imgOut

def prepExport(f):
    f = (f.set({
        'width': ee.Number(f.get('MLength')).multiply(f.get('channelMask')),
        'endsInWater': ee.Number(f.get('any')).eq(1),
        'endsOverEdge': ee.Number(f.get('count')).lt(2)}))

    fOut = (ee.Feature(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]), {})
    .copyProperties(f, None, ['any', 'count', 'MLength', 'xc', 'yc', 'channelMask']))
    return fOut

def CalculateWidth(imgIn):
    crs = imgIn.get('crs')
    scale = imgIn.get('scale')
    imgId = imgIn.get('image_id')
    bound = imgIn.select(['riverMask']).geometry()
    angle = imgIn.select(['orthDegree'])
    infoEnds = imgIn.select(['riverMask'])
    infoExport = (imgIn.select('channelMask')
    .addBands(imgIn.select('^flag.*'))
    .addBands(dem.rename(['flag_elevation'])))
    dm = imgIn.select(['distanceMap'])

    widths = GetWidth(angle, infoExport, infoEnds, dm, crs, bound, scale, imgId, '').map(prepExport)

    return widths

def Zhang_Suen(image, crs, scale):
    weights = [[1,1,1], [1,1,1], [1,1,1]]
    kernel = ee.Kernel.fixed(weights = weights)
    weights1 = [[0,1,0], [0,0,1], [0,1,0]]
    kernel1 = ee.Kernel.fixed(weights = weights1)
    weights2 = [[0,0,0], [1,0,1], [0,1,0]]
    kernel2 = ee.Kernel.fixed(weights = weights2)
    weights3 = [[1,0,0], [0,0,0], [0,0,0]]
    kernel3 = ee.Kernel.fixed(weights = weights3)
    weights4 = [[0,1,0], [0,0,0], [0,0,0]]
    kernel4 = ee.Kernel.fixed(weights = weights4)
    weights5 = [[0,0,1], [0,0,0], [0,0,0]]
    kernel5 = ee.Kernel.fixed(weights = weights5)
    weights6 = [[0,0,0], [0,0,1], [0,0,0]]
    kernel6 = ee.Kernel.fixed(weights = weights6)
    weights7 = [[0,0,0], [0,0,0], [0,0,1]]
    kernel7 = ee.Kernel.fixed(weights = weights7)
    weights8 = [[0,0,0], [0,0,0], [0,1,0]]
    kernel8 = ee.Kernel.fixed(weights = weights8)
    weights9 = [[0,0,0], [0,0,0], [1,0,0]]
    kernel9 = ee.Kernel.fixed(weights = weights9)
    weights10 = [[0,0,0], [1,0,0], [0,0,0]]
    kernel10 = ee.Kernel.fixed(weights = weights10)
    weights11 = [[0,1,0], [1,0,1], [0,0,0]]
    kernel11 = ee.Kernel.fixed(weights = weights11)
    weights12 = [[0,1,0], [1,0,0], [0,1,0]]
    kernel12 = ee.Kernel.fixed(weights = weights11)
    clLS_01p = ee.Image()
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel3).reproject(crs = crs, scale = scale).rename('1'))
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel4).reproject(crs = crs, scale = scale).rename('2'))
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel5).reproject(crs = crs, scale = scale).rename('3'))
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel6).reproject(crs = crs, scale = scale).rename('4'))
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel7).reproject(crs = crs, scale = scale).rename('5'))
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel8).reproject(crs = crs, scale = scale).rename('6'))
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel9).reproject(crs = crs, scale = scale).rename('7'))
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel10).reproject(crs = crs, scale = scale).rename('8'))
    clLS_01p = clLS_01p.addBands(image.reduceNeighborhood(ee.Reducer.count(), kernel3).reproject(crs = crs, scale = scale).rename('9'))
    clLS_01p = clLS_01p.select(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    clLS_01p = clLS_01p.toArray()
    mat = ee.Array([
        [10, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 10, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 10, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 10, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 10, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 10, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 10, 1]
    ])
    A = ee.Image(mat).matrixMultiply(clLS_01p.toArray(1)).arrayProject([0]).eq(ee.Array([1,1,1,1,1,1,1,1]))
    A1 = A.arrayDotProduct(ee.Image(ee.Array([1,1,1,1,1,1,1,1])))
    clLS_count = image.reduceNeighborhood(ee.Reducer.count(), kernel)
    clLS_count1 = image.reduceNeighborhood(ee.Reducer.count(), kernel1)
    clLS_count2 = image.reduceNeighborhood(ee.Reducer.count(), kernel2)
    clLS = clLS_count.gt(2).And(clLS_count.lt(6)).And(clLS_count1.lt(3)).And(clLS_count2.lt(3)).And(A1.eq(1)).selfMask()
    clLS_count11 = image.reduceNeighborhood(ee.Reducer.count(), kernel11)
    clLS_count21 = image.reduceNeighborhood(ee.Reducer.count(), kernel12)
    clLS1 = clLS_count.gt(2).And(clLS_count.lt(6)).And(clLS_count11.lt(3)).And(clLS_count21.lt(3)).And(A1.eq(1)).selfMask()
    clLS2 = clLS.Or(clLS1)
    return(clLS1)

def cl(image, n):
    crs = image.projection().crs().getInfo()
    scale = image.projection().nominalScale().getInfo()
    image1 = image
    for i in range(n):
        remove = Zhang_Suen(image1, crs, scale)
        image1 = image1.subtract(remove.unmask()).eq(1).selfMask()
    return image1.reproject(crs = crs, scale = scale)


def batch_load_LS(collection, bandnames):
    l = list()
    bands_in_LS1 = ['SR_B3', 'SR_B2', 'SR_B1', 'SR_B4', 'SR_B5']
    bands_in_LS2 = ['SR_B4', 'SR_B3', 'SR_B2', 'SR_B5', 'SR_B6']
    for i in collection:
        n = int(i.split('/')[1][3])
        if n < 8:
            l.append(ee.Image(i).select(bands_in_LS1).rename(bandnames))
        else:
            l.append(ee.Image(i).select(bands_in_LS2).rename(bandnames))
    img_c = ee.ImageCollection(l)
    return img_c

def ref_cr(img):
    return img.multiply(0.0000275).add(-0.2)

def wrap_classification(collection, thr2=0, thr3=0.2, thr4=0.3, thr5=0.6, numPts=100):
    thr2=0
    thr3=0.2
    thr4=0.3
    thr5=0.6
    numPts=100
    def classification(image):
        crs = image.projection().crs()
        ndvi = image.normalizedDifference(['NIR', 'Red'])
        scale = image.projection().nominalScale()
        #minDis
        #Three land type simple regions: water, plant, sand (other)
        bands = image.bandNames()
        labels = 'samples'
        water = ndvi.lt(thr2)
        sand = ndvi.lt(thr4).And(ndvi.gt(thr3)).multiply(2)
        vege = ndvi.gt(thr5).multiply(3)
        wsv = water.add(sand.add(vege)).selfMask().rename('samples')
        samples = wsv.stratifiedSample(**{'numPoints': numPts, 'projection': crs, 'scale':30, 'geometries': True})
        training = image.select(bands).sampleRegions(**{
                    'collection': samples,
                    'properties': ['samples'],
                    'scale' : scale
                })
        trained = ee.Classifier.minimumDistance().train(training, 'samples', bands)
        classified = image.select(bands).classify(trained)
        return classified
    return collection.map(classification)