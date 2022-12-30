import ee

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