"""
Main functionality of ``image_bbox_slicer``.
"""
import os
import csv
import glob
import random
from PIL import Image
from pascal_voc_writer import Writer
from torchvision.transforms.functional import pad as tvpad #Used for padding
#from pathlib import Path  #I put this in helpers instead
from .helpers import *

class Slicer(object):
    """
    Slicer class.

    Attributes
    ----------
    IMG_SRC : str
        /path/to/images/source/directory.
        Default value is the current working directory. 
    IMG_DST : str
        /path/to/images/destination/directory.
        Default value is the `/sliced_images` in the current working directory.
    ANN_SRC : str
        /path/to/annotations/source/directory.
        Default value is the current working directory.
    ANN_DST : str
        /path/to/annotations/source/directory.
        Default value is the `/sliced_annotations` in the current working directory.
    keep_partial_labels : bool
        A boolean flag to denote if the slicer should keep partial labels or not.
        Partial labels are the labels that are partly in a tile post slicing.
        Default value is `False`.
    save_before_after_map : bool
        A boolean flag to denote if mapping between 
        original file names and post-slicing file names in a csv or not. 
        Default value is `False`.
    ignore_empty_tiles : bool
        A boolean flag to denote if tiles with no labels post-slicing
        should be ignored or not.
        Default value is `True`.
    empty_sample : float (0 to 1)
        Proportion of the 'empty' tiles (tiles without bounding boxes) to sample.  Default is 0.
    """

    def __init__(self):
        """
        Constructor. 

        Assigns default values to path attributes and other preference attributes. 

        Parameters
        ----------
        None
        """
        self.IMG_SRC = os.getcwd()
        self.IMG_DST = os.path.join(os.getcwd(), 'sliced_images')
        self.ANN_SRC = os.getcwd()
        self.ANN_DST = os.path.join(os.getcwd(), 'sliced_annotations')
        self.keep_partial_labels = False
        self.save_before_after_map = False
        self.ignore_empty_tiles = True
        self._ignored_files = [] #Files without objects of interest to be ignored (if ignore_empty_tiles=TRUE)
        self._mapper = {} #A dict of file+tile names.
        self._just_image_call = True
        self._tilematrix_dim = None
        self._tile_size = None
        self._tile_overlap = None
        
    def config_dirs(self, img_src, ann_src,
                    img_dst=os.path.join(os.getcwd(), 'sliced_images'),
                    ann_dst=os.path.join(os.getcwd(), 'sliced_annotations')):
        """Configures paths to source and destination directories after validating them. 

        Parameters
        ----------
        img_src : str
            /path/to/image/source/directory
        ann_src : str
            /path/to/annotation/source/directory
        img_dst : str, optional
            /path/to/image/destination/directory
            Default value is `/sliced_images`.
        ann_dst : str, optional
            /path/to/annotation/destination/directory
            Default value is `/sliced_annotations`.

        Returns
        ----------
        None
        """
        validate_dir(img_src)
        validate_dir(ann_src)
        validate_dir(img_dst, src=False)
        validate_dir(ann_dst, src=False)
        validate_file_names(img_src, ann_src)
        self.IMG_SRC = img_src
        self.IMG_DST = img_dst
        self.ANN_SRC = ann_src
        self.ANN_DST = ann_dst
          
    def pad_image(self,img,tile_size,tile_overlap):
        """
        Returns an image that is padded to an even multiple of the tile size, taking overlap 
        into account.  The padding (black by default) is added to the right and bottom edges of 
        the original image.
        Parameters
        ---------------
        img: a PIL image
        tile_size: tuple (width, height) in pixels
        tile_overlap: float. Tile overlap between consecutive strides as proportion of tile size 
        (the same proportion is used for height and width).
        """
        #Extract params
        im = Image.open(img)
        img_size = im.size
        
        #Calculate padding
        padding = calc_padding(img_size,tile_size,tile_overlap)
        
        #Pad image, using pad function from torchvision.transforms.functional
        padded_img = tvpad(im, padding) #By default, fill=0 (black), padding_mode='constant'
        return padded_img

    def __get_tiles(self,img_size, tile_size, tile_overlap):
        """Generates a list coordinates of all the tiles after validating the values. 
        Private Method.
        Parameters
        ----------
        img_size : tuple
            Size of the original image in pixels, as a 2-tuple: (width, height).
        tile_size : tuple
            Size of each tile in pixels, as a 2-tuple: (width, height).
        tile_overlap: float 
            Tile overlap between two consecutive strides as percentage of tile size.
        Returns
        ----------
        list
            A list of tuples.
            Each holding coordinates of possible tiles 
            in the format - `(xmin, ymin, xmax, ymax)` 
        """
        validate_tile_size(tile_size, img_size)
        tiles = []
        img_w, img_h = img_size
        tile_w, tile_h = tile_size
        #Convert overlap to pixels
        tile_w_overlap = int(tile_w * tile_overlap) 
        tile_h_overlap = int(tile_h * tile_overlap)
        #Calculate stride
        stride_w = tile_w - tile_w_overlap
        stride_h = tile_h - tile_h_overlap
        #Calculate number of rows and cols (and index)
        rows = range(0, img_h-tile_h+1, stride_h)
        nrows = len(rows)
        cols = range(0, img_w-tile_w+1, stride_w)
        ncols = len(cols)
        #Make list of tile coordinates
        for y in rows:
            for x in cols:
                x2 = x + tile_w
                y2 = y + tile_h
                tiles.append((x, y, x2, y2))
        self._tilematrix_dim = (nrows,ncols) 
        #breakpoint()
        return tiles
    
    def slice_by_size(self, tile_size, tile_overlap=0.0,empty_sample=0.0):
        """Slices both images and box annotations in source directories by specified size and overlap.

        Parameters
        ----------
        tile_size : tuple
            Size (width, height) of each tile.
        tile_overlap: float, optional  
            Percentage of tile overlap between two consecutive strides.
            Default value is `0`.

        Returns
        ----------
        None
        """
        self._just_image_call = False
        self.slice_bboxes_by_size(tile_size, tile_overlap,empty_sample)
        self.slice_images_by_size(tile_size, tile_overlap)
        #Reset params
        self._ignored_files = []
        self._just_image_call = True

    def slice_by_number(self, number_tiles):
        """Slices both images and box annotations in source directories into specified number of tiles.

        Parameters
        ----------
        number_tiles : int
            The number of tiles an image needs to be sliced into.

        Returns
        ----------
        None
        """
        self._just_image_call = False
        self.slice_bboxes_by_number(number_tiles)
        self.slice_images_by_number(number_tiles)
        self._ignored_files = []
        self._just_image_call = True

    def slice_images_by_size(self, tile_size, tile_overlap=0.0):
        """Slices each image in the source directory by specified size and overlap.

        Parameters
        ----------
        tile_size : tuple
            Size of each tile in pixels, as a 2-tuple: (width, height).
        tile_overlap: float, optional  
            Percentage of tile overlap between two consecutive strides.
            Default value is `0`.

        Returns
        ----------
        None
        """
        validate_tile_size(tile_size)
        validate_overlap(tile_overlap)
        if self._just_image_call:
            self.ignore_empty_tiles = []
        mapper = self.__slice_images(tile_size, tile_overlap, number_tiles=-1)
        if self.save_before_after_map:
            save_before_after_map_csv(mapper, self.IMG_DST)
        self._mapper = {} #reset mapper

    def slice_images_by_number(self, number_tiles):
        """Slices each image in the source directory into specified number of tiles.

        Parameters
        ----------
        number_tiles : int
            The number of tiles an image needs to be sliced into.

        Returns
        ----------
        None
        """
        validate_number_tiles(number_tiles)
        if self._just_image_call:
            self.ignore_empty_tiles = []
        mapper = self.__slice_images(None, None, number_tiles=number_tiles)
        if self.save_before_after_map:
            save_before_after_map_csv(mapper, self.IMG_DST)
        self._mapper = {} #reset mapper
        
    def __slice_images(self, tile_size, tile_overlap, number_tiles):
        """
        Private Method
        If a self._mapper dict has been created by slice_bboxes(), we use it to determine which tiles to save.
        Otherwise we follow our own logic.
        """
        mapper = {} #A dict
        img_no = 1
        self._tile_size = tile_size #set these in self for plotting
        self._tile_overlap = tile_overlap
            
        #for file in sorted(glob.glob(self.IMG_SRC + "/*")):
        image_files = [str(x) for x in sorted(list(Path(self.IMG_SRC).rglob('*')))]
        for file in image_files:
            file_fullname = file
            file_name = file.split('/')[-1].split('.')[0]
            file_type = file.split('/')[-1].split('.')[-1].lower()
            if file_type.lower() not in IMG_FORMAT_LIST:
                continue
            #Pad image (black added to right and bottom) so you don't lose edges
            im = self.pad_image(file,tile_size,tile_overlap) 

            if number_tiles > 0:
                n_cols, n_rows = calc_columns_rows(number_tiles)
                tile_w, tile_h = int(floor(im.size[0] / n_cols)), int(floor(im.size[1] / n_rows))
                tile_size = (tile_w, tile_h)
                tile_overlap = 0.0

            #Get a list of tile coordinates
            tiles = self.__get_tiles(im.size, tile_size, tile_overlap)
            
            #Note: in the top-level function slice_by_size(), slice_bboxes_by_size() 
            #is called *before* slice_images_by_size(); therefore, the 'ignore_tiles' 
            #list has already been modified in __slice_bboxes() when it is passed to this function.
            new_ids = []
            for tile in tiles:
                row,col = self.__get_rowcol_indexes(tiles,tile)
                tile_id_str = '{}{}{}'.format(file_name,row,col) #To match with images, extension is omitted
                #tile_id_str = str('{:06d}'.format(img_no))
                
                if self._mapper:
                    if tile_id_str in self._mapper[file_name]:
                        new_im = im.crop(tile) 
                        new_im.save('{}/{}.{}'.format(self.IMG_DST, tile_id_str, file_type))
                        new_ids.append(tile_id_str)
                        img_no += 1
                else:
                    #Skip files if they are in the ignore list
                    if len(self._ignored_files) != 0:
                        if tile_id_str in self._ignored_files:
                            #pop the name once it has been skipped so you don't keep finding it
                            self._ignored_files.remove(tile_id_str) 
                            continue
                    new_im = im.crop(tile) #moved down to avoid wasting the cropping operation on skipped files
                    new_im.save('{}/{}.{}'.format(self.IMG_DST, tile_id_str, file_type))
                    new_ids.append(tile_id_str)
                    img_no += 1
            mapper[file_fullname] = new_ids #Add the tiles to the dict (key=file_name, item = saved tiles (new_ids))    
        print('Obtained {} image slices!'.format(img_no-1))
        return mapper

    def slice_bboxes_by_size(self, tile_size, tile_overlap,empty_sample):
        """Slices each box annotation in the source directory by specified size and overlap.

        Parameters
        ----------
        tile_size : tuple
            Size of each tile in pixels, as a 2-tuple: (width, height).
        tile_overlap: float, optional  
            Percentage of tile overlap between two consecutive strides.
            Default value is `0`.

        Returns
        ----------
        None
        """
        validate_tile_size(tile_size)
        validate_overlap(tile_overlap)
        self._ignored_files = []
        mapper = self.__slice_bboxes(tile_size, tile_overlap, number_tiles=-1,empty_sample=empty_sample)
        self._mapper = mapper
        if self.save_before_after_map:
            save_before_after_map_csv(mapper, self.ANN_DST)

    def slice_bboxes_by_number(self, number_tiles,empty_sample):
        """Slices each box annotation in source directories into specified number of tiles.

        Parameters
        ----------
        number_tiles : int
            The number of tiles an image needs to be sliced into.

        Returns
        ----------
        None
        """
        validate_number_tiles(number_tiles)
        self._ignored_files = []
        mapper = self.__slice_bboxes(None, None, number_tiles=number_tiles,empty_sample=empty_sample)
        self._mapper = mapper
        if self.save_before_after_map:
            save_before_after_map_csv(mapper, self.ANN_DST)

    def __slice_bboxes(self, tile_size, tile_overlap, number_tiles, empty_sample):
        """
        Private Method.  Determines whether tiles contain bounding boxes, then saves tiles as requested.
        
        Writes a mapper (a dict of filenames/tiles) to self that may later be read by __slice_images()
        """        
        img_no = 1
        mapper = {}
        empty_count = 0

        for xml_file in sorted(glob.glob(self.ANN_SRC + '/*.xml')):
            root, objects = extract_from_xml(xml_file)
            #Get size of original image
            orig_w, orig_h = int(root.find('size')[0].text), int(
                root.find('size')[1].text)
            #Get original image filename
            im_filename = root.find('filename').text.split('.')[0]
            #Get size of padded image
            padding = calc_padding((orig_w,orig_h),tile_size,tile_overlap)
            im_size = (orig_w + padding[2],orig_h + padding[3])
            im_w,im_h = im_size
            if number_tiles > 0:
                n_cols, n_rows = calc_columns_rows(number_tiles)
                tile_w = int(floor(im_w / n_cols))
                tile_h = int(floor(im_h / n_rows))
                tile_size = (tile_w, tile_h)
                tile_overlap = 0.0
            else:
                tile_w, tile_h = tile_size
            tiles = self.__get_tiles(im_size, tile_size, tile_overlap)
            tile_ids = []

            for tile in tiles:
                #Get tile row and column
                row,col = self.__get_rowcol_indexes(tiles,tile)
                img_no_str = '{}{}{}'.format(im_filename,row,col)
                #initialize a new writer
                voc_writer = Writer('{}'.format(img_no_str), tile_w, tile_h)
                #Loop through all objects (bboxes) in the image to check if each falls in this tile
                empty_count = 0 #The number of bboxes that don't fall in the tile
                for obj in objects:
                    obj_lbl = obj[-4:]
                    points_info = which_points_lie(obj_lbl, tile)

                    if points_info == Points.NONE:
                        empty_count += 1 
                        continue

                    elif points_info == Points.ALL:       # All points lie inside the tile
                        new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                                   obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                    elif not self.keep_partial_labels:    # Ignore partial labels based on configuration
                        empty_count += 1
                        continue

                    elif points_info == Points.P1:
                        new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                                   tile_w, tile_h)

                    elif points_info == Points.P2:
                        new_lbl = (0, obj_lbl[1] - tile[1],
                                   obj_lbl[2] - tile[0], tile_h)

                    elif points_info == Points.P3:
                        new_lbl = (obj_lbl[0] - tile[0], 0,
                                   tile_w, obj_lbl[3] - tile[1])

                    elif points_info == Points.P4:
                        new_lbl = (0, 0, obj_lbl[2] - tile[0],
                                   obj_lbl[3] - tile[1])

                    elif points_info == Points.P1_P2:
                        new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                                   obj_lbl[2] - tile[0], tile_h)

                    elif points_info == Points.P1_P3:
                        new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                                   tile_w, obj_lbl[3] - tile[1])

                    elif points_info == Points.P2_P4:
                        new_lbl = (0, obj_lbl[1] - tile[1],
                                   obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                    elif points_info == Points.P3_P4:
                        new_lbl = (obj_lbl[0] - tile[0], 0,
                                   obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                    voc_writer.addObject(obj[0], new_lbl[0], new_lbl[1], new_lbl[2], new_lbl[3],
                                         obj[1], obj[2], obj[3])
                #Add filename to the "ignore" list if none of the bbox objects fall in it
                if self.ignore_empty_tiles and (empty_count == len(objects)):
                    self._ignored_files.append(img_no_str)
                    #However we may still sample it
                    rd = random.random()
                    if rd < empty_sample:
                        #Save the tile (it's empty but we sample it)
                        voc_writer.save('{}/{}.xml'.format(self.ANN_DST, img_no_str))
                        tile_ids.append(img_no_str)
                        img_no += 1
                else:
                    #Save the tile (it contains objects of interest)
                    voc_writer.save('{}/{}.xml'.format(self.ANN_DST, img_no_str))
                    tile_ids.append(img_no_str)
                    img_no += 1
            mapper[im_filename] = tile_ids #Add new item to mapper dict (key=filename,value=tile_ids)

        print('Obtained {} annotation slices!'.format(img_no-1))
        return mapper

    def resize_by_size(self, new_size, resample=0):
        """Resizes both images and box annotations in source directories to specified size `new_size`.

        Parameters
        ----------
        new_size : tuple
            The requested size in pixels, as a 2-tuple: (width, height)
        resample: int, optional  
            An optional resampling filter, same as the one used in PIL.Image.resize() function.
            Check it out at https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
            `0` (Default) for NEAREST (nearest neighbour)
            `1` for LANCZOS/ANTIALIAS (a high-quality downsampling filter)
            `2` for BILINEAR/LINEAR (linear interpolation)
            `3` for BICUBIC/CUBIC (cubic spline interpolation)

        Returns
        ----------
        None
        """
        self.resize_images_by_size(new_size, resample)
        self.resize_bboxes_by_size(new_size)

    def resize_images_by_size(self, new_size, resample=0):
        """Resizes images in the image source directory to specified size `new_size`.

        Parameters
        ----------
        new_size : tuple
            The requested size in pixels, as a 2-tuple: (width, height)
        resample: int, optional  
            An optional resampling filter, same as the one used in PIL.Image.resize() function.
            Check it out at https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
            `0` (Default) for NEAREST (nearest neighbour)
            `1` for LANCZOS/ANTIALIAS (a high-quality downsampling filter)
            `2` for BILINEAR/LINEAR (linear interpolation)
            `3` for BICUBIC/CUBIC (cubic spline interpolation)

        Returns
        ----------
        None
        """
        validate_new_size(new_size)
        self.__resize_images(new_size, resample, None)

    def resize_by_factor(self, resize_factor, resample=0):
        """Resizes both images and annotations in the source directories by a scaling/resizing factor.

        Parameters
        ----------
        resize_factor : float
            A factor by which the images and the annotations should be scaled.
        resample: int, optional  
            An optional resampling filter, same as the one used in PIL.Image.resize() function.
            Check it out at https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
            `0` (Default) for NEAREST (nearest neighbour)
            `1` for LANCZOS/ANTIALIAS (a high-quality downsampling filter)
            `2` for BILINEAR/LINEAR (linear interpolation)
            `3` for BICUBIC/CUBIC (cubic spline interpolation)

        Returns
        ----------
        None
        """
        validate_resize_factor(resize_factor)
        self.resize_images_by_factor(resize_factor, resample)
        self.resize_bboxes_by_factor(resize_factor)

    def resize_images_by_factor(self, resize_factor, resample=0):
        """Resizes images in the image source directory by a scaling/resizing factor.

        Parameters
        ----------
        resize_factor : float
            A factor by which the images should be scaled.
        resample: int, optional  
            An optional resampling filter, same as the one used in PIL.Image.resize() function.
            Check it out at https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
            `0` (Default) for NEAREST (nearest neighbour)
            `1` for LANCZOS/ANTIALIAS (a high-quality downsampling filter)
            `2` for BILINEAR/LINEAR (linear interpolation)
            `3` for BICUBIC/CUBIC (cubic spline interpolation)

        Returns
        ----------
        None
        """
        validate_resize_factor(resize_factor)
        self.__resize_images(None, resample, resize_factor)

    def __resize_images(self, new_size, resample, resize_factor):
        """Private Method
        """
        #for file in sorted(glob.glob(self.IMG_SRC + "/*")):
        image_files = [str(x) for x in sorted(list(Path(self.IMG_SRC).rglob('*')))]
        for file in image_files:
            file_name = file.split('/')[-1].split('.')[0]
            file_type = file.split('/')[-1].split('.')[-1].lower()
            if file_type not in IMG_FORMAT_LIST:
                continue
            im = Image.open(file)
            if resize_factor is not None:
                new_size = [0, 0]
                new_size[0] = int(im.size[0] * resize_factor)
                new_size[1] = int(im.size[1] * resize_factor)
                new_size = tuple(new_size)

            new_im = im.resize(size=new_size, resample=resample)
            new_im.save('{}/{}.{}'.format(self.IMG_DST, file_name, file_type))

    def resize_bboxes_by_size(self, new_size):
        """Resizes bounding box annotations in the source directory to specified size `new_size`.

        Parameters
        ----------
        new_size : tuple
            The requested size in pixels, as a 2-tuple: (width, height)

        Returns
        ----------
        None
        """
        validate_new_size(new_size)
        self.__resize_bboxes(new_size, None)

    def resize_bboxes_by_factor(self, resize_factor):
        """Resizes bounding box annotations in the source directory by a scaling/resizing factor.

        Parameters
        ----------
        resize_factor : float
            A factor by which the bounding box annotations should be scaled.

        Returns
        ----------
        None
        """
        validate_resize_factor(resize_factor)
        self.__resize_bboxes(None, resize_factor)

    def __resize_bboxes(self, new_size, resize_factor):
        """Private Method
        """
        for xml_file in sorted(glob.glob(self.ANN_SRC + '/*.xml')):
            root, objects = extract_from_xml(xml_file)
            im_w, im_h = int(root.find('size')[0].text), int(
                root.find('size')[1].text)
            im_filename = root.find('filename').text.split('.')[0]
            an_filename = xml_file.split('/')[-1].split('.')[0]
            if resize_factor is None:
                w_scale, h_scale = new_size[0]/im_w, new_size[1]/im_h
            else:
                w_scale, h_scale = resize_factor, resize_factor
                new_size = [0, 0]
                new_size[0], new_size[1] = int(
                    im_w * w_scale), int(im_h * h_scale)
                new_size = tuple(new_size)

            voc_writer = Writer(
                '{}'.format(im_filename), new_size[0], new_size[1])

            for obj in objects:
                obj_lbl = list(obj[-4:])
                obj_lbl[0] = int(obj_lbl[0] * w_scale)
                obj_lbl[1] = int(obj_lbl[1] * h_scale)
                obj_lbl[2] = int(obj_lbl[2] * w_scale)
                obj_lbl[3] = int(obj_lbl[3] * h_scale)

                voc_writer.addObject(obj[0], obj_lbl[0], obj_lbl[1], obj_lbl[2], obj_lbl[3],
                                     obj[1], obj[2], obj[3])
            voc_writer.save('{}/{}.xml'.format(self.ANN_DST, an_filename))

    def visualize_sliced_random(self, map_dir=None):
        """Picks an image randomly and visualizes unsliced and sliced images using `matplotlib`.

        Parameters:
        ----------
        map_dir : str, optional
            /path/to/mapper/directory.
            By default, looks for `mapper.csv` in image destination folder. 

        Returns:
        ----------
        None
            However, displays the final plots.
        """
        if not self.save_before_after_map and map_dir is None:
            print('No argument passed to `map_dir` and save_before_after_map is set False. \
                Looking for `mapper.csv` in image destination folder.')
        mapping = ''

        if map_dir is None:
            map_path = self.IMG_DST + '/mapper.csv'
        else:
            map_path = map_dir + '/mapper.csv'

        #Extract one record (orig_filename, list of tile_names) from the mapper.csv file at random
        with open(map_path) as src_map:
            read_csv = csv.reader(src_map, delimiter=',')
            # Skip the header
            next(read_csv, None)
            mapping = random.choice(list(read_csv))
            src_fullpath = mapping[0] #the full path to the jpg image
            src_name = src_fullpath.split('/')[-1].split('.')[-2] #just the filename without extension
            tile_files = mapping[1:]
            tsize = self._tile_size
            toverlap = self._tile_overlap
            
            #Plot the original image, then the tiles
            self.plot_image_boxes(self.IMG_SRC, self.ANN_SRC, src_fullpath, src_name)
            self.plot_tile_boxes(self.IMG_SRC,self.IMG_DST, self.ANN_DST, src_fullpath, src_name,tile_files,tsize,toverlap)

    def visualize_resized_random(self):
        """Picks an image randomly and visualizes original and resized images using `matplotlib`

        Parameters:
        ----------
        None 

        Returns:
        ----------
        None
            However, displays the final plots.
        """
        #im_file = random.choice(list(glob.glob('{}/*'.format(self.IMG_SRC))))
        image_files = [str(x) for x in list(Path(self.IMG_SRC).rglob('*'))]
        im_file = random.choice(image_files)
        file_name = im_file.split('/')[-1].split('.')[0]

        self.plot_image_boxes(self.IMG_SRC, self.ANN_SRC, file_name)
        self.plot_image_boxes(self.IMG_DST, self.ANN_DST, file_name)

    def __get_rowcol_indexes(self,tiles,tile):
        """
        Private method.
        Finds the row and column index for a given tile by searching for the tile's
        coordinates in a list of all tile coordinates.  

        Parameters
        ----------
        tiles : a list of tuples (xmin,ymin,xmax,ymax) that define a tile
        tile : a tuple for one particular tile
        Returns:
        ----------
        (rownum,colnum): tuple.  The row and column index of the tile passed in

        """
        #Get list of unique row and col coordinates
        x1 = sorted(set([i[0] for i in tiles]))
        x2 = sorted(set([i[1] for i in tiles]))
        x3 = sorted(set([i[2] for i in tiles]))
        x4 = sorted(set([i[3] for i in tiles]))
        cols = list(zip(x1,x3)) #enclose iterable zip in list() to make reusable
        rows = list(zip(x2,x4))

        #Find a particular tile's coordinates in the list
        colnum  = [n for (n,tpl) in enumerate(cols) if (tile[0],tile[2]) == tpl]
        rownum = [n for (n,tpl) in enumerate(rows) if (tile[1],tile[3]) == tpl]
        return (rownum,colnum)

    def plot_image_boxes(self,img_path, ann_path, src_fullpath,src_name):
        """Plots bounding boxes on images using `matplotlib`.
        Parameters
        ----------
        img_path : str
            /path/to/image/source/directory
        ann_path : str
            /path/to/annotation/source/directory
        src_fullpath: str 
            /full/path/to/image
        src_name: str 
            image name without extension

        Returns
        ----------
        None
        """    
        #Plot original image
        #Find the original xml file with bbox annotations for this image:
        tree = ET.parse(ann_path + '/' + src_name + '.xml')
        root = tree.getroot()
        #Find the image and convert to Numpy array
        #im = Image.open(img_path + '/' + file_name + '.jpg')
        im = Image.open(src_fullpath)
        im = np.array(im, dtype=np.uint8)

        #list the original (un-tiled) bounding boxes
        rois = []
        for member in root.findall('object'):
            rois.append((int(float(member[4][0].text)), int(float(member[4][1].text)),
                             int(float(member[4][2].text)), int(float(member[4][3].text))))

        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(10, 10))

        # Display the image
        ax.imshow(im)

        #Display the bounding boxes on top of it
        for roi in rois:
            # Create a Rectangle patch
            rect = patches.Rectangle((roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1],
                                     linewidth=3, edgecolor='b', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()

    def plot_tile_boxes(self,src_img_path,tileimg_path, ann_path, src_fullpath,src_name,tile_names,tile_size,tile_overlap):
        """
        Plots a matrix of tiles from a tiled image in correct row/column locations.
        Parameters
        ----------
        src_img_path: /path/to/image/source/directory
        tileimg_path: /path/to/tile_image/destination/directory
        ann_path: /path/to/annotation/source/directory
        src_fullpath: full path to original (un-tiled) file
        src_name: filename only of original (un-tiled) file, without extension
        tile_names: list of filenames for the tiles corresponding to the src_img_name file
        tile_size: tuple(int,int).  Tile size in pixels.  Set in __slice_images().
        tile_overlap: float.  Proportion of overlap between consecutive tiles.  
        
        Returns
        ----------
        None
        """
        #Get tile matrix dimensions for this particular image
        #Image must be padded and then tiles calculated.
        fn = src_fullpath
        orig_size = Image.open(fn).size
        padding = calc_padding(orig_size,tile_size,tile_overlap)
        img_size = (orig_size[0] + padding[2],orig_size[1] + padding[3])
        #We don't care about the tiles, just the side-effect of setting tilematrix_dim
        _ = self.__get_tiles(img_size, tile_size, tile_overlap)
        rows,cols = self._tilematrix_dim
        #Create a matrix of empty subplots (n_rows x n_cols)
        pos = []
        for i in range(0, rows):
            for j in range(0, cols):
                pos.append((i, j))
        fig, ax = plt.subplots(rows, cols, sharex='col',
                               sharey='row', figsize=(10, 7))
        for file in tile_names:
            #Get the tile annotation
            tree = ET.parse(ann_path + '/' + file + '.xml')
            root = tree.getroot()
            #Get the tile image
            im = Image.open(tileimg_path + '/' + file + '.jpg')
            #Extract the tile row & col coordinates from the name        
            clist = re.findall(r"\[([0-9]+)\]", file)
            coords = tuple([int(x) for x in clist])
            #Convert the image to an Numpy array
            im = np.array(im, dtype=np.uint8)

            #Make a list of bboxes
            rois = []
            for member in root.findall('object'):
                rois.append((int(float(member[4][0].text)), int(float(member[4][1].text)),
                             int(float(member[4][2].text)), int(float(member[4][3].text))))

            # Display the tile at the right position
            ax[coords[0], coords[1]].imshow(im)
            #ax[pos[idx][0], pos[idx][1]].imshow(im)

            #Show the bounding boxes on the tile
            for roi in rois:
                # Create a Rectangle patch
                rect = patches.Rectangle((roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1],
                                         linewidth=3, edgecolor='b', facecolor='none')
                # Add the patch to the Axes
                ax[coords[0], coords[1]].add_patch(rect)
        plt.show()

class Points(Enum):
    """An Enum to hold info of points of a bounding box or a tile.
    Used by the method `which_points_lie` and a private method in `Slicer` class. 
    See `which_points_lie` method for more details.

    Example
    ----------
    A box and its points
    P1- - - - - - -P2
    |               |
    |               |
    |               |
    |               |
    P3- - - - - - -P4
    """

    P1, P2, P3, P4 = 1, 2, 3, 4
    P1_P2 = 5
    P1_P3 = 6
    P2_P4 = 7
    P3_P4 = 8
    ALL, NONE = 9, 10


def which_points_lie(label, tile):
    """Method to check if/which points of a label lie inside/on the tile.

    Parameters
    ----------
    label: tuple
        A tuple with label coordinates in `(xmin, ymin, xmax, ymax)` format.
    tile: tuple
        A tuple with tile coordinates in `(xmin, ymin, xmax, ymax)` format.  

    Note
    ----------
    Ignoring the cases where either all 4 points of the `label` or none of them lie on the `tile`, 
    at most only 2 points can lie on the `tile`. 

    Returns
    ----------
    Point (Enum)
        Specifies which point(s) of the `label` lie on the `tile`.
    """
    # 0,1 -- 2,1
    # |        |
    # 0,3 -- 2,3
    points = [False, False, False, False]

    if (tile[0] <= label[0] and tile[2] >= label[0]):
        if (tile[1] <= label[1] and tile[3] >= label[1]):
            points[0] = True
        if (tile[1] <= label[3] and tile[3] >= label[3]):
            points[2] = True

    if (tile[0] <= label[2] and tile[2] >= label[2]):
        if (tile[1] <= label[1] and tile[3] >= label[1]):
            points[1] = True
        if (tile[1] <= label[3] and tile[3] >= label[3]):
            points[3] = True

    if sum(points) == 0:
        return Points.NONE
    elif sum(points) == 4:
        return Points.ALL

    elif points[0]:
        if points[1]:
            return Points.P1_P2
        elif points[2]:
            return Points.P1_P3
        else:
            return Points.P1

    elif points[1]:
        if points[3]:
            return Points.P2_P4
        else:
            return Points.P2

    elif points[2]:
        if points[3]:
            return Points.P3_P4
        else:
            return Points.P3

    else:
        return Points.P4
