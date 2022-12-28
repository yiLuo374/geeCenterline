import ee

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