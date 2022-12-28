import ee
import pandas as pd

def load_file(assetID, bands=None, bandnames=None):
    tp = ee.data.getAsset(assetID)['type']
    
    if tp == 'TABLE':
        return load_roi(assetID)
    
    if tp == 'IMAGE':
        return load_image(assetID, bands, bandnames)
        
def load_image(assetID, bands, names):
    image = ee.Image(assetID).select(bands).rename(names)
    return image

def load_roi(assetID):
    roi = ee.FeatureCollection(assetID).geometry()
    return roi
