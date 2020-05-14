from distutils.core import setup
setup(
  name = 'image_bbox_tiler',
  packages = ['image_bbox_slicer'],   
  version = '0.2',      
  license='MIT',        
  long_description = 'This is a fork of the image_box_slicer package.  This easy-to-use library is a data transformer useful in Object Detection tasks. It splits images and their bounding box annotations into tiles, both into specific sizes and into any arbitrary number of equal parts. It can also resize them, both by specific sizes and by a resizing/scaling factor. \n\nRead the docs at https://image-bbox-slicer.readthedocs.io/en/latest/.',   
  author = 'John Payne; this is a fork of image_bbox_slicer, by AKSHAY CHANDRA LAGANDULA',
  author_email = '22941316+jcpayne@users.noreply.github.com',
  url = 'https://github.com/jcpayne/image_bbox_tiler/',
  download_url = '',
  keywords = ['Image Tiler','Image Slicer', 'Tiling','Bounding Box Annotations', 'PASCAL VOC', 'Object Detection'],  
  install_requires=[            
		'Pillow',
		'numpy',
		'matplotlib',
		'pascal-voc-writer'
		],
  classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
