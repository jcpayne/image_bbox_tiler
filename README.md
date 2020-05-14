[![PyPI version](https://badge.fury.io/py/image-bbox-slicer.svg)](https://badge.fury.io/py/image-bbox-slicer) [![](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)


# image_bbox_tiler (IN DEVELOPMENT)
This is a fork of the image_bbox_slicer package, designed to make the tiling more accurate, to avoid losing pixels on the edges of images, and to allow the user to sample some proportion of 'empty' tiles (tiles that do not include any object of interest).  It also avoids creating tiles that will not be saved, to speed up the tiling operation.

**CAVEATS: I'm ignorant about proper etiquette for forking so maybe I shouldn't have changed the name or docs or license...I'm just trying to make the forked version importable.  Currently the `slice_by_size()` function is working (plus all of the functions that it depends on) but I haven't tested everything to make sure that I didn't break something else.**

The main differences are:
1. The original package discarded any pixels that fall outside an even multiple of the tile size. That wastes a lot of data if the tiles are large. Each image is now padded with zeros out to an even multiple of tile size _before_ tiling it so no data is lost, and the padding works correctly if the images are of different sizes. 
2. The tile overlap math, tile size calculations, and row and column indexes were fixed to make them precisely correct (instead of various rough approximations, truncations, etc.);
3. Fixed a problem with float values in annotations that caused a display error;
4. Built in the capability to sample a variable proportion of empty tiles;
5. Revamped tile naming so that tiles are named with row and column indexes to make future reassembly easier.
6. Modified the code so tiles that will not be saved are not created in the first place, to save memory and CPU cycles;
7. Made the tiled images display in the correct row and column relative to the original image, and to show padding (the placement of tiles in the original package was approximate, relative to the source image). 

The rest of this document is the original **image_bbox_slicer** document:
---------------------------------------------
This easy-to-use library is a data transformer sometimes useful in Object Detection tasks. It splits images and their bounding box annotations into tiles, both into specific sizes and into any arbitrary number of equal parts. It can also resize them, both by specific sizes and by a resizing/scaling factor. Read the docs [here](https://image-bbox-slicer.readthedocs.io/en/latest/).

<div align="center">
<img src="imgs/ibs_demo.jpg" alt="Partial Labels Example" />
</div>

Currently, this library only supports bounding box annotations in [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) format. And as of now, there is **no command line execution support**. Please raise an issue if needed. 

## UPDATE: This tool was only tested on Linux/Ubuntu. Please find a _potential fix_ to make it work on Windows [here](https://github.com/acl21/image_bbox_slicer/issues/2).  

## Installation
```bash
#Install this fork:
$ pip install git+https://github.com/jcpayne/image_bbox_tiler@master 

#The original package is on PyPI.  To install it instead of this fork: 
$ pip install image_bbox_slicer
```

Works with Python 3.4 and higher versions and requires:
```python
Pillow==5.4.1
numpy==1.16.2
pascal-voc-writer==0.1.4
matplotlib==3.0.3
```

## Usage - A Quick Demo
_Note: This usage demo can be found in `demo.ipynb` in the repo._

```python
import image_bbox_tiler as ibs
```

### Create And Configure `Slicer` Object

#### Setting Paths To Source And Destination Directories.
You must configure paths to source and destination directories like the following. 

```python
im_src = './src/images'
an_src = './src/annotations'
im_dst = './dst/images'
an_dst = './dst/annotations'

slicer = ibs.Slicer()
slicer.config_dirs(img_src=im_src, ann_src=an_src, 
                   img_dst=im_dst, ann_dst=an_dst)
```

#### Dealing With Partial Labels
<div align="center">
<img src="imgs/partial_labels.jpg" alt="Partial Labels Example" style="width: 850px;" />
</div>

The above images show the difference in slicing with and without partial labels. In the image on the left, all the box annotations masked in <span style="color:green">**green**</span> are called Partial Labels. 

Configure your slicer to either ignore or consider them by setting `Slicer` object's `keep_partial_labels` instance variable to `True` or `False` respectively. By default it is set to `False`.


```python
slicer.keep_partial_labels = True
```

#### Dealing With Empty Tiles
<img src="imgs/empty_tiles.png" alt="Empty Tiles Example" style="width: 850px;"/>

An empty tile is a tile with no "labels" in it. The definition of "labels" here is tightly coupled with the user's preference of partial labels. If you choose to keep the partial labels (i.e. `keep_partial_labels = True`), a tile with a partial label is not treated as empty. If you choose to not keep the partial labels (i.e. `keep_partial_labels = False`), a tile with one or more partial labels is considered empty. 

Configure your slicer to either ignore or consider empty tiles by setting `Slicer` object's `ignore_empty_tiles` instance variable to `True` or `False` respectively. By default it is set to `True`.  

**New in image_bbox_tiler**: you can sample a proportion of the empty tiles by setting 'empty_sample'= <a float in the range [0-1]>.


```python
slicer.ignore_empty_tiles = False
```

#### Before-After Mapping

You can choose to store the mapping between file names of the images before and after slicing by setting the `Slicer` object's `save_before_after_map` instance variable to `True`. By default it is set to `False`.

Typically, `mapper.csv` looks like the following:
```
| old_name   | new_names                       |
|------------|---------------------------------|
| 2102       | 000001, 000002, 000003, 000004  |
| 3931       | 000005, 000005, 000007, 000008  |
| test_image | 000009, 000010, 000011, 000012  |
| ...        | ...                             |
```


```python
slicer.save_before_after_map = True
```

### Slicing

#### Images and Bounding Box Annotations Simultaneously

#### By Number Of Tiles


```python
slicer.slice_by_number(number_tiles=4)
slicer.visualize_sliced_random()
```

<div align="center">
<img src="imgs/output_10_1.png" alt="Output1" style="width: 200px;" />

<img src="imgs/output_10_2.png" alt="Output2" style="width: 200px;" />
</div>


#### By Specific Size

```python
slicer.slice_by_size(tile_size=(418,279), tile_overlap=0)
slicer.visualize_sliced_random()
```


<div align="center">
<img src="imgs/output_12_1.png" alt="Output3" style="width: 200px;" />

<img src="imgs/output_12_2.png" alt="Output4" style="width: 200px;" />
</div>

*Note: `visualize_sliced_random()` randomly picks a recently sliced image from the directory for plotting.*

### Other Slicing Functions

#### By Number Of Tiles
```python
slicer.slice_images_by_number(number_tiles=4)
```

#### By Specific Size
```python
slicer.slice_images_by_size(tile_size=(418,279), tile_overlap=0)
```

#### Slicing Only Bounding Box Annotations
#### By Number Of Tiles
```python
slicer.slice_bboxes_by_number(number_tiles=4)
```

#### By Specifc Size
```python
slicer.slice_bboxes_by_size(tile_size=(418,279), tile_overlap=0)
```

### Resizing 
![png](imgs/resize_demo.png)

#### Images and Bounding Box Annotations Simultaneously
#### By Specific Size


```python
slicer.resize_by_size(new_size=(500,200))
slicer.visualize_resized_random()
```


![png](imgs/output_18_0.png)


![png](imgs/output_18_1.png)


#### By A Resize Factor


```python
slicer.resize_by_factor(resize_factor=0.05)
slicer.visualize_resized_random()
```

![png](imgs/output_20_0.png)


![png](imgs/output_20_1.png)

_Note:_

*`visualize_resized_random()` randomly picks a recently resized image from the destination directory for plotting.*


### Other Resizing Functions

#### Resizing Separately

#### Only Images

* #### By Specific Size

```python
slicer.resize_images_by_size(new_size=(500,200))
```

* #### By Resize Factor

```python
slicer.resize_images_by_factor(resize_factor=0.05)
```

####  Only Bounding Box Annotations

* #### By Specific Size
```python
slicer.resize_bboxes_by_size(new_size=(500,200))
```

* #### By Resize Factor
```python
slicer.resize_bboxes_by_factor(resize_factor=0.05)
```
