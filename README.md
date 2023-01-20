# geeCenterline
**A framework for river centerline extraction and sediment bar identification from satellite images based on [Google Earth Engine Python API](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api).** 
## Introduction
**geeCenterline** is a python package designed for river centerline extraction and sediment bar identification from multi-scale remote sensing images. This package is based on Google Earth Engine, which allow users to make the use of Google Cloud for computation to save local resources. 

This package aims to provide a general workflow for users to identify water and sand surfaces from multiple remote sensing image products, such as Landsat and PlanetScope, and obtain river planform geometries, including one-pixel-wide river centerline and river width, from the river mask obtained from the remote sensing images.
![alt text](https://github.com/yiLuo374/geeRiverCl/blob/main/img/workflow.jpg)

<div style="display:flex">
    <div style="flex:1;padding-right:10px;">
        <img src="https://github.com/yiLuo374/geeRiverCl/blob/main/img/original.gif" width="200"/>
    </div>
    <div style="flex:1;padding-left:10px;">
        <img src="https://github.com/yiLuo374/geeRiverCl/blob/main/img/classified.gif" width="200"/>
    </div>
</div>

<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="https://github.com/yiLuo374/geeRiverCl/blob/main/img/original.gif" width="200"/>
      	    </td>
            <td style="padding:10px">
            	<img src="https://github.com/yiLuo374/geeRiverCl/blob/main/img/classified.gif" width="300"/>
            </td>
        </tr>
    </table>
</div>


## Features

 - Water surface and sand surface classification for remote sencsing images with at lest four bands: Red, Green, Blue, and Near-infrared.
 - Identify the river mask from the water mask.
 - Extract one-pixel-wide river centerline.
 - Obtain the width of the river.

## Prerequests
 - Sign up for [Google Earth Engine](https://earthengine.google.com/)
 -  Python >= 3.7
 - NumPy
 -  [earthengine-api](https://developers.google.com/earth-engine/guides/python_install)
 - [geemap](https://github.com/giswqs/geemap#installation)
 - [Jupyter Notebook](https://jupyter.org/)
 It is recomanded to install [Anaconda](https://jupyter.org/) and create a virtual environment for Google Earth Engine and install other packages:
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
