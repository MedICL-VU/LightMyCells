Welcome to the Light My Cells Challenge training database !

This Challenge is empowered by the national infrastructure France BioImaging (https://france-bioimaging.org/) 
and will be at ISBI2024 (https://biomedicalimaging.org/2024/).

If you want further informations we invite you to visit the challenge website 
https://lightmycells.grand-challenge.org

Don't hesitate if you have any question to post it on the forum of : 
https://grand-challenge.org/forums/forum/light-my-cells-bright-field-to-fluorescence-imaging-challenge-715/


In this ZIP, you  have :

 - README.txt : this file
 - Light My Cells Challenge REMBI.xlsx: 
 	A REMBI file with metadata to assume FAIR data before the BioImage Archive release.
 - Filelist directoty : 
 	tsv files with lists of images by studies 
 	(= biosample preparations, or, in other words by 'type' of acquisitions)
 - ReadImages.py : 
 	a example code to read each individual images
 - download_train.py
 	code to download the whole database
 
 
If you prefer downloading each study, 
in addition, you have .tar.gz files for each study (30) at:
 https://seafile.lirmm.fr/d/123f71e12bf24db59d84/

Each .tar.gz which contains a study with images.
(e.g. for extracting one .tar.gz into the current working directory: tar -xf filename.tar.gz )


Thus whole training database has :
	- 57 022images (with 52 382 z inputs) over 30 studies
	- all in ome.tiff format
	- the dimensions are TCZYX
			with T,C,Z equal to 1, and various X, Y shapes. 
			(for exemple you can have one image with shape (1,1,1,512,512) and another with (1,1,1,2048,2048))
	
	- you can easily read the images through the python library aiscimageio (bioformats)
		- with AISCImage :

			from aicsimageio import AISCImage

			img   = AISCImage(image_path) 		# you can add 'readers ='
			array = img.data 					# get your image as ndarray
			metadata = img.metadata				# get the metadata as ome_types
			xml_metadata = img.to_xml() 		# get the metadata as xml	


		- with BioFile :
			from aicsimageio.readers.bioformats_reader import BioFile

			img   = BioFile(image_path) 		
			array = img.to_numpy() 				# get your image as ndarray
			metadata = img.ome_metadata			# get the metadata as ome_types


		-> to navigates through metadata you can check 
			https://ome-types.readthedocs.io/en/latest/API/base_type/#ome_types._mixins._base_type.OMEType
			for example you can try:
				metadata 				# ome_type object (OME element is a container for all information objects accessible by OME)
				metadata.images 			# list
				metadata.images[0].pixels		# ome_type pixels object
				metadata.images[0].pixels.channels 	# list
				metadata.images[0].pixels.physical_size_x # float
		
