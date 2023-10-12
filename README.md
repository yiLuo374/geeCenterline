# geeCenterline
**For *river centerline extraction* and *sediment bar identification* from remote sensing images based on [Google Earth Engine (GEE) Python API](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api)**

[![Static Badge](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/mit/)

## What Can geeCenterline Do?
- Water and sand identification.
- Automatic river identification from water mask.
- One-pixel wide river centerline extraction.

## Why geeCenterline?
Friendly to river planform and centerline extraction on large areas and from a collection of multiple remote sensing images:
- Highly automated.
- Free cloud space for high-speed computation provided by GEE.
- Performed well on different imagery collections ([PlanetScope](https://developers.planet.com/docs/data/planetscope/) and [Landsat](https://landsat.gsfc.nasa.gov/) in the example).
- Require fewer spectral bands (RGB and near-infrared only).

## Extract River Centerline
![alt text](https://github.com/yiLuo374/geeRiverCl/blob/main/img/workflow.jpg)
## Masked River and Sandbar Migration
<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="https://github.com/yiLuo374/geeRiverCl/blob/main/img/original.gif" width="300"/>
      	    </td>
            <td style="padding:10px">
            	<img src="https://github.com/yiLuo374/geeRiverCl/blob/main/img/classified.gif" width="300"/>
            </td>
        </tr>
    </table>
</div>

## Prerequisites
 - Sign up for [Google Earth Engine](https://earthengine.google.com/)
 -  Python >= 3.7
 - NumPy
 -  [earthengine-api](https://developers.google.com/earth-engine/guides/python_install)
 - [geemap](https://github.com/giswqs/geemap#installation)
 - [Jupyter Notebook](https://jupyter.org/)
 It is recommended to install [Anaconda](https://jupyter.org/) and create a virtual environment for Google Earth Engine and install other packages:
```
conda create -n gee python numpy
conda activate gee
conda install -c conda-forge jupyter
conda install -c conda-forge earthengine-api
conda install geemap -c conda-forge
```
You can also install these packages by `pip`:
```
pip install jupyter notebook
pip install numpy earthengine-api geemap
```
