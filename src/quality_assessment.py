import os
import csv
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyhdf.SD import SD, SDC
from tqdm import tqdm


def init_qc_table():
	"""Create a df structure to map 8-bit integer quality assessment reference"""
	
	QC_Data = []

	# Iterate through the list of 8-bit integers and populate QC table with bit definitions 
	for integer in range(0, 256, 1):
		bits = list(map(int, list("{0:b}".format(integer).zfill(8))))

		# Describe each of the bits. Remember bits are big endian so bits[7] == bit 0
		# Mandatory_QA bits description
		if (bits[6] == 0 and bits[7] == 0):
			Mandatory_QA = 'LST GOOD'
		elif (bits[6] == 0 and bits[7] == 1):
			Mandatory_QA = 'LST Produced,Other Quality'
		elif (bits[6] == 1 and bits[7] == 0):
			Mandatory_QA = 'No Pixel,clouds'
		elif (bits[6] == 1 and bits[7] == 1):
			Mandatory_QA = 'No Pixel, Other QA'

		# Data_Quality bits description
		if (bits[4] == 0 and bits[5] == 0):
			Data_Quality = 'Good Data'
		elif (bits[4] == 0 and bits[5] == 1):
			Data_Quality = 'Other Quality'
		elif (bits[4] == 1 and bits[5] == 0):
			Data_Quality = 'TBD'
		elif (bits[4] == 1 and bits[5] == 1):
			Data_Quality = 'TBD'

		# Emiss_Err bits description
		if (bits[2] == 0 and bits[3] == 0):
			Emiss_Err = 'Emiss Err <= .01'
		elif (bits[2] == 0 and bits[3] == 1):
			Emiss_Err = 'Emiss Err <= .02'
		elif (bits[2] == 1 and bits[3] == 0):
			Emiss_Err = 'Emiss Err <= .04'
		elif (bits[2] == 1 and bits[3] == 1):
			Emiss_Err = 'Emiss Err > .04'

		# LST_Err bits description
		if (bits[0] == 0 and bits[1] == 0):
			LST_Err = 'LST Err <= 1K'
		elif (bits[0] == 0 and bits[1] == 1):
			LST_Err = 'LST Err <= 3K'
		elif (bits[0] == 1 and bits[1] == 0):
			LST_Err = 'LST Err <= 2K'
		elif (bits[0] == 1 and bits[1] == 1):
			LST_Err = 'LST Err > 3K' 

		# Append this integers bit values and descriptions to list
		QC_Data.append([integer] + bits + [Mandatory_QA, Data_Quality, Emiss_Err, LST_Err])

	QC_Data = pd.DataFrame(QC_Data, columns=['Integer_Value', 'Bit7', 'Bit6', 'Bit5', 'Bit4', 'Bit3', 'Bit2', 'Bit1', 'Bit0', 'Mandatory_QA', 'Data_Quality', 'Emiss_Err', 'LST_Err'])
	return QC_Data


def estimate_tile_pixel_quality(tile, QC_Data):
	"""Return cloud coverage and above 3K error for a given tile"""

	# define quality values to be excluded
	nopixels = ("No Pixel,clouds", "No Pixel, Other QA")

	# define Temperature error to be excluded from no-cloud filtered QA_Data
	noerrors = ("LST Err > 3K",)

	# filter Full 8-bit QC_Data according to current tile
	tile = tile.flatten()
	current_values = pd.unique(tile)
	QC_Data_current = QC_Data[QC_Data.Integer_Value.isin(current_values)].sort_values("Integer_Value")
	QC_Data_current["percentage"] = [(tile == value).sum() / tile.shape[0] for value in sorted(QC_Data_current["Integer_Value"])]
	
	#QC_Data_current.sort_values("percentage", ascending=False, inplace=True)
	
	#QC_Data_current["cum_percentage"] = QC_Data_current["percentage"].cumsum()

	# Estimate the amount of cloud / other non-LST coverage percentage
	cloud_perc = 0.
	for nopixel in nopixels:
		cloud_perc += QC_Data_current[QC_Data_current.Mandatory_QA==nopixel]["percentage"].values[0]\
			if nopixel in QC_Data_current.Mandatory_QA.unique() \
			else 0.

	# Filter according to Valid Variable data (no-cloud)
	filtered = QC_Data_current[~QC_Data_current.Mandatory_QA.isin(nopixels)]

	# Estimate percentage Temperature errors of non clouds data
	error_perc = 0.
	for noerror in noerrors:
		#try: print((filtered.groupby("LST_Err")["Integer_Value"].count()/filtered.shape[0])[noerror])
		#except: print("notfound")
		error_perc += (filtered.groupby("LST_Err")["Integer_Value"].count() / filtered.shape[0])[noerror]\
			if noerror in filtered.LST_Err.unique() \
			else 0.

	return round(cloud_perc,4), round(error_perc,4)


def resize_data(data, outdim):
	""" Resize input tensor """
	data = torch.Tensor(data).unsqueeze(0).unsqueeze(0)
	interp = F.interpolate(data, outdim)
	return interp.squeeze().numpy()


def scan_tiles(data, QC_Data, tile_size):
	"""Scan tiles and provide useful metrics
	Args:
		data: (np.array) 2D variable grid
		QC_data: (pd.DataFrame) of QC table as for MODIS product
		tile_size: (int) 
	"""
	tile_results = dict()

	h, w = data.shape

	# generate tile coordinate 
	tiles_h = [(i, i+tile_size) for i in range(0, h, tile_size)]
	tiles_w = [(i, i+tile_size) for i in range(0, w, tile_size)]

	tile_number = 1
	for i, coord_h in enumerate(tiles_h):
		for j, coord_w in enumerate(tiles_w):

			# collect tile coordinates
			tile = data[coord_h[0] : coord_h[1], coord_w[0] : coord_w[1]]

			# estimate tile stats
			cloud_perc, error_perc = estimate_tile_pixel_quality(tile, QC_Data)
			
			# store results
			tile_results[f"t{tile_number}_fcloud"] = cloud_perc
			tile_results[f"t{tile_number}_ferror"] = error_perc
			tile_number += 1

	return tile_results


def scan_single(filename, data, variable_name, layer, QC_Data, tile_size):
	"""Return a single df row of a given file"""

	result = dict()

	data = resize_data(data, outdim=tile_size * 4)

	# collect values
	fnamelist = filename.strip().split(".")
	product, date, tiles, version = fnamelist[0], fnamelist[1], fnamelist[2], fnamelist[3]
	
	# store params
	result["product"] =  product
	result["version"] =  version
	result["variable"] = variable_name
	result["layer"] = layer
	result["date"] =  date
	result["year"] =  int(date[1:5])
	result["fullname"] =  filename
	result["vtile"] =  int(tiles[4:6])
	result["htile"] =  int(tiles[1:3])
	result["tiles"] =  "v" + str(tiles[4:6]) + "h" + str(tiles[1:3])
	result["tile_date_layer"] = result["tiles"] + "_" + result["date"] + "_" + result["layer"]

	# Overall acquisition coverage
	qa_fcloud, qa_ferror = estimate_tile_pixel_quality(data, QC_Data)
	result["qa_fcloud"] = qa_fcloud
	result["qa_ferror"] = qa_ferror

	# update with tile results
	result.update(scan_tiles(data, QC_Data, tile_size))

	return result
	

def scan_all(folder, variable_name, tile_size, outfile, year_filter=None):
	"""Perform for all files over folder filtered according to year_filter
	Args:
		folder: (str) folder name.
		variable_name: (str) Satellite Variable of Interest.
		tile_size: (int) squared tile size.
		outfile: (str) output filename
		year_filter: (str) filtering over year on which to perform the whole scanning procedure.
	"""
	qa_list = []

	QC_Data = init_qc_table()

	file_list = sorted(os.listdir(folder))

	# filder according to year
	if year_filter:
		file_list = [item for item in file_list if item.strip().split(".")[1][1:5] == year_filter]

	# Main loop
	for i, filename in tqdm(enumerate(file_list), total = len(file_list)):

		if not filename.endswith(".hdf"): continue

		# open with rasterIO
		#hdf_file = rio.open_rasterio(os.path.join(folder,filename))

		# open with pyhdf
		hdf_file = SD(os.path.join(folder,filename), SDC.READ)

		# loop over QC day and night
		for layer in ["QC_Day", "QC_Night"]:

			# read current QC layer
			data = hdf_file.select(layer)[:,:]

			# complete fullname Day or Night
			res = variable_name.split("_")[-1]
			variable_name = f"LST_{layer.split('_')[-1]}_{res}"

			# perform single scan
			current_scan = scan_single(filename, data, variable_name, layer, QC_Data, tile_size)
			
			# save into pd.DataFrame
			current_file_values = pd.DataFrame(current_scan, index = [f"{i}_{layer.split('_')[-1]}"])

			# concat QA results over current hdr_files in a list
			qa_list.append(current_file_values)


	# condition for which no file has been previously computed
	if os.path.exists(outfile):
	
		# read past file
		qa_past = pd.read_csv(outfile)

		# merging past QA dataframe into the current
		qa_list.insert(0, qa_past)

	qa = pd.concat(qa_list)
	qa.to_csv(outfile, index=False)


if __name__ == "__main__":

	# example usage
	parser = argparse.ArgumentParser(description='Scan through folder files and save QA over csv.')
	parser.add_argument('-f','--folder', type=str, help='Folder to be scan', required=True)
	parser.add_argument('-v','--variable', type=str, help='Variable name', required=True)
	parser.add_argument('-t','--tile-size', type=int, help='Tile size', required=True)
	parser.add_argument('-o','--output', type=str, help="Output path", required=True)
	parser.add_argument('-y','--year-filter',type=str, help="Filter according to year", required=False)
	args = vars(parser.parse_args())

	print(args)

	folder = args["folder"]
	variable = args["variable"]
	tile_size = args["tile_size"] 
	outfile = args["output"]
	year_filter = args["year_filter"]

	scan_all(folder, variable, tile_size, outfile, year_filter)

	