import ee

def NDVI(img):
    ndvi = img.normalizedDifference(['NIR', 'Red']).rename("NDVI")
    return(ndvi)

def NDWI(img):
    ndwi = img.normalizedDifference(['Green', 'NIR']).rename('NDWI')
    return(ndwi)

def MNDWI(img):
    mndwi = img.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI')
    return(mndwi)

def EVI(img):
    evi = img.expression('2.5 * (NIR - Red) / (1 + NIR + 6 * Red - 7.5 * Blue)', {
    'NIR': img.select(['NIR']),
    'Red': img.select(['Red']),
    'Blue': img.select(['Blue'])
    }).rename('EVI')
    return(evi)

def classification(img, thr1=-1, thr2=0, thr3=0.2, thr4=0.3, thr5=0.6, thr6=1, numPts=1000):
    crs = img.projection().crs().getInfo()
    scl = img.projection().nominalScale()
    ndvi = NDVI(img)
    if 'SWIR1' in img.bandNames().getInfo():
        bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
    else: 
        bands = ['Blue', 'Green', 'Red', 'NIR']
    labels = 'samples'
    water = ndvi.lt(thr2).And(ndvi.gt(thr1))
    sand = ndvi.lt(thr4).And(ndvi.gt(thr3)).multiply(2)
    vege = ndvi.gt(thr5).And(ndvi.lt(thr6)).multiply(3) #0.6
    wsv = water.add(sand.add(vege)).selfMask().rename('samples')
    samples = wsv.stratifiedSample(**{'numPoints': numPts, 'projection': crs, 'scale':scl, 'geometries': True})
    training = img.select(bands).sampleRegions(**{
                'collection': samples,
                'properties': ['samples'],
                'scale' : 3
            })
    trained = ee.Classifier.smileRandomForest(10).train(training, 'samples', bands)
    classified = img.select(bands).classify(trained)
    return classified

def otsu_hist(img, scale, ltb):
    histogram = img.reduceRegion(
        **{
            'reducer': ee.Reducer.histogram(255, 2),
            'geometry': ltb,
            'scale': scale,
            'bestEffort': True,
        }
    )
    return histogram

def otsu(histogram):
    counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
    means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = sum.divide(total)

    indices = ee.List.sequence(1, size)

    # Compute between sum of squares, where each mean partitions the data.

    def func_xxx(i):
        aCounts = counts.slice(0, 0, i)
        aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
        aMeans = means.slice(0, 0, i)
        aMean = (
            aMeans.multiply(aCounts)
            .reduce(ee.Reducer.sum(), [0])
            .get([0])
            .divide(aCount)
        )
        bCount = total.subtract(aCount)
        bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
        return aCount.multiply(aMean.subtract(mean).pow(2)).add(
            bCount.multiply(bMean.subtract(mean).pow(2))
        )

    bss = indices.map(func_xxx)

    # Return the mean value corresponding to the maximum BSS.
    return means.sort(bss).get([-1])

def thr_otsu(img, name, scale, ltb):
    histogram = otsu_hist(img, scale, ltb)
    threshold = otsu(histogram.get(name))
    return threshold