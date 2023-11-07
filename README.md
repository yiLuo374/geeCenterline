# geeCenterline
**For *river centerline extraction* and *sediment bar identification* from remote sensing images based on [Google Earth Engine (GEE) Python API](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api)**

[![Static Badge](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/mit/)

## What Can geeCenterline Do?
- Water and sand identification.
- Automatic river identification from water mask.
- One-pixel wide river centerline extraction.

## Why geeCenterline?
The algorithm is friendly to river planform and centerline extraction over expansive regions from images collected from multiple dates.
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
## Example data sources
### Data of the Tallahatchie River & the Big Sunflower River
| Study area                | Setellite   | Date      | ID                                       |
|---------------------------|-------------|-----------|------------------------------------------|
| Little Tallahatchie River | PlanetScope | 10/7/2021 | 20211007_155117_27_245a                  |
|                           |             |           | 20211007_155119_58_245a                  |
|                           |             |           | 20211007_155121_88_245a                  |
|                           |             |           | 20211007_155124_19_245a                  |
|                           |             |           | 20211007_155126_49_245a                  |
|                           |             |           | 20211007_155128_79_245a                  |
|                           |             |           | 20211007_155131_10_245a                  |
|                           |             |           | 20211007_155133_40_245a                  |
|                           |             | 6/23/2022 | 20220623_155055_14_2465                  |
|                           | Landsat 8   | 6/25/2022 | LC08_L2SP_023036_20220625_20220706_02_T1 |
|                           |             |           | LC08_L2SP_023037_20220625_20220706_02_T1 |
| Big   Sunflower River     | PlanetScope | 8/13/2022 | 20220813_163511_16_2426                  |
|                           |             |           | 20220813_163513_42_2426                  |
|                           |             |           | 20220813_163515_68_2426                  |
|                           |             |           | 20220813_163517_93_2426                  |
|                           |             |           | 20220813_163520_19_2426                  |

### Data of the Arkansas River
Can be obtained from Google Earth Engine following the steps in the [example notebook](https://github.com/yiLuo374/geeCenterline/blob/main/example.ipynb).
