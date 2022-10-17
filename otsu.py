import ee

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