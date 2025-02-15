import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as tt
import torchvision.transforms.functional as ttf

from thop import profile
from pyhdf.SD import SD, SDC
 
from quality_assessment import init_qc_table

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SF = 0.02



class Mask_AvgPool2d(torch.nn.Module):
	""" Custom AvgPool2D layer meant to be used while imputing missing data:
	if zeros are involved over the considered spatial kernels, the averaged
	output will neglect them """
	def __init__(self, kernel_size, stride, padding):
		super(Mask_AvgPool2d, self).__init__()
		self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
		self.stride      = torch.nn.modules.utils._pair(stride)
		self.padding     = torch.nn.modules.utils._quadruple(padding)

	def _masked_avg(self, x):
		""" Custom avg """
		# Masked mean (excluding zeros)

		x = torch.where(x==0, x[x!=0].mean(), x)
		m = x.mean(-1)
		if m.isnan().any().item():
		    m = torch.where(m.isnan(), 0, m).mean(-1)
		return m

	def forward(self, x):
		#x = x.to(DEVICE) # check this that may not work while using cuda11.6
		h, w = x.shape[-2], x.shape[-1]

		#if len(x.shape) == 3:
		#    _, h, w = x.shape
		#elif len(x.shape) == 4:
		#    _, _, h, w = x.shape
		#else:
		#    raise ValueError(f" Shape must be [b,c,h,w] or [c,h,w]: found n.dim ({len(x.shape)}")

		x = x.reshape(1, 1, h, w) if len(x.shape)!=4 else x

		x = ttf.pad(x, tuple([(self.kernel_size[0]-1)//2]*4), padding_mode='reflect')
		x = x.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
		x = x.contiguous().view(x.size()[:4] + (-1,))
		pool = self._masked_avg(x).squeeze(0)
		return pool.to("cpu")
    
class DataFilter():
	"""
	Example usage
	datafilter =  DataFilter("lr-file-to-path", "hr-file-to-path", 0.1)
	datafilter.scan_all()
	"""
	def __init__(self, lr_filepath:str, hr_filepath:str, cloud_coverage:float):
		
		self.lr_filepath = lr_filepath
		self.hr_filepath = hr_filepath
		self.lr_list = [item for item in sorted(os.listdir(self.lr_filepath))\
			 if item.endswith(".hdf")]
		self.hr_list = [item for item in sorted(os.listdir(self.hr_filepath))\
			 if item.endswith(".hdf")]
		self.lr_layers = ["LST_Day_6km", "LST_Night_6km"]
		self.hr_layers = ["LST_Day_1km", "LST_Night_1km"]
		self.cloud_coverage = cloud_coverage
		self.merged_df = None
		self.qc_table = init_qc_table()

	@staticmethod
	def _read_rescale_data(filepath, layername, sf):
		""" General utils function to open a hdf file and rescale it
		Args:
			filepath: (str) full path
			layername: (str) of the layer name to be open
			sf: (float) scale factor
		"""
		data = SD(filepath, SDC.READ)
		data = torch.Tensor(data.select(layername)[:,:].astype(np.float32))
		return data * sf

	@staticmethod
	def _get_qc_data(filepath, layername):
		""" Return QC layer
		"""
		data = SD(filepath, SDC.READ)
		data = torch.Tensor(data.select(layername)[:,:].astype(np.int64))
		return data

	@staticmethod
	def _estimate_tile_pixel_quality(tile, QC_Data):
		"""Return cloud coverage and above 3K error for a given tile
		Args:
			tile: (array) object of QC data
			QC_Data (pd.DataFrame) reference QC 8-bit table
		"""
		cloud_perc, error_perc = 0., 0.
	 
		# Define quality values to be excluded and Temperature Error
		nopixels = ("No Pixel,clouds", "No Pixel, Other QA")
		noerrors = ("LST Err > 3K",)

		# Filter Full 8-bit QC_Data according to current tile
		tile = tile.flatten()
		current_values = pd.unique(tile)
		QC_Data_current = QC_Data[QC_Data.Integer_Value.isin(current_values)]\
			.sort_values("Integer_Value")
		QC_Data_current["percentage"] = [(tile == value).sum() / tile.shape[0]\
			for value in sorted(QC_Data_current["Integer_Value"])]
		
		# Estimate the amount of cloud / other non-LST coverage percentage
		for nopixel in nopixels:
			cloud_perc += QC_Data_current[QC_Data_current.Mandatory_QA==nopixel]["percentage"].values[0]\
				if nopixel in QC_Data_current.Mandatory_QA.unique() \
				else 0.

		# Filter according to Valid Variable data (no-cloud)
		filtered = QC_Data_current[~QC_Data_current.Mandatory_QA.isin(nopixels)]

		# Estimate percentage Temperature errors of non clouds data
		for noerror in noerrors:
			error_perc += (filtered.groupby("LST_Err")["Integer_Value"].count() / filtered.shape[0])[noerror]\
				if noerror in filtered.LST_Err.unique() \
				else 0.

		return cloud_perc, error_perc

	@staticmethod
	def _estimate_zerofilled(tile):
		return (tile==0).float().mean().item()


	def merge_datasets(self):
		""" Combine scanning datasets from LR and HR sets
		"""
		dfs = []
		for res in ["lr", "hr"]:
			file_list = self.lr_list if res == "lr" else self.hr_list
			df = pd.DataFrame({f"{res}_files":file_list})
			df[f"{res}_date"] = df[f"{res}_files"].apply(lambda x: x.split(".")[1])
			df[f"{res}_tiles"] = df[f"{res}_files"].apply(lambda x: x.split(".")[2])
			df["date_tiles"] = df[f"{res}_date"] + "_" + df[f"{res}_tiles"]
			dfs.append(df)

		return dfs[0].merge(dfs[1], how="inner", on="date_tiles")


	def scan_all(self):

		if os.path.exists("./scanning_paired_dataset.csv"):
			return

		self.merged_df = self.merge_datasets()

		re = []

		for i, line in tqdm(self.merged_df.iterrows(), total = self.merged_df.shape[0]):

			date_tile = line["date_tiles"]
			lr_file = os.path.join(self.lr_filepath, line["lr_files"])
			hr_file = os.path.join(self.hr_filepath, line["hr_files"])

			tmps = []
			for layer in ["Day", "Night"]:

				lr_qc_data = self._get_qc_data(lr_file, f"QC_{layer}")
				hr_qc_data = self._get_qc_data(hr_file, f"QC_{layer}")

				lr_cp, lr_ep = self._estimate_tile_pixel_quality(lr_qc_data, self.qc_table)
				hr_cp, hr_ep = self._estimate_tile_pixel_quality(hr_qc_data, self.qc_table)

				tmps.append({
					f"lr_{layer}_zero_filled":self._estimate_zerofilled(self._read_rescale_data(lr_file, f"LST_{layer}_6km", SF)),
					f"lr_{layer}_cloud_perc":lr_cp,
					f"lr_{layer}_error_perc":lr_ep,
					f"hr_{layer}_zero_filled":self._estimate_zerofilled(self._read_rescale_data(hr_file, f"LST_{layer}_1km", SF)),
					f"hr_{layer}_cloud_perc":hr_cp,
					f"hr_{layer}_error_perc":hr_ep,
					})

			tmps[0].update(tmps[1]) 

			re.append(pd.DataFrame(tmps[0], index=[date_tile]))

		re = pd.concat(re).reset_index().rename(columns={"index":"date_tiles"})

		self.merged_df = self.merged_df.merge(re, how="inner", on="date_tiles")

		self.merged_df.to_csv("./scanning_paired_dataset.csv")


# Actually no more useful!
class SubTilesCoords():
	def __init__(self, lr_data_shape, hr_data_shape, 
		lr_subtile_size, hr_subtile_size):

		self._subtiles_coords = dict()

		# generate sub-tile coordinate 
		for ii, data in enumerate([lr_data_shape, hr_data_shape]):

			k = "lr" if ii == 0 else "hr"

			tile_size = lr_subtile_size if k == "lr" else hr_subtile_size

			self._subtiles_coords[k] = dict()

			_, h, w = data

			# h sub-tiles coords
			tiles_h = [(i, i+tile_size) for i in range(0, h, tile_size)] 

			# w sub-tiles coords
			tiles_w = [(i, i+tile_size) for i in range(0, w, tile_size)] 

			# draw sub-tiles from t1-t16
			subtiles = []
			for _i, coord_h in enumerate(tiles_h):
				for _j, coord_w in enumerate(tiles_w):
					subtiles.append((coord_h[0], coord_h[1], coord_w[0], coord_w[1]))

			# store info
			self._subtiles_coords[k]["tiles_h"] = tiles_h # consider to remove 
			self._subtiles_coords[k]["tiles_w"] = tiles_w
			self._subtiles_coords[k]["tiles_n"] = subtiles

	def get_coords(self):
		return self._subtiles_coords



class MODIS_dataset(Dataset):

	def __init__(self,
		lr_filepath,
		hr_filepath,
		qa,
		zerof_thr,
		cloud_thr,
		error_thr,
		impute_thr,
		up_scale,
		kernel_size,
		train_params,
		data_set = "train",
		seed = 8609):
		"""
		Args:
		...
		qa: (str) path to file to qa scanning (LR-HR matching)
		...
		"""
		self.lr_filepath = lr_filepath
		self.hr_filepath = hr_filepath

		self.zerof_thr = zerof_thr
		self.cloud_thr = cloud_thr
		self.error_thr = error_thr
		self.impute_thr = impute_thr

		self.up_scale = up_scale
		self.seed = seed

		# Subpatches sizes
		self.patch_size = 64

		#self.lr_data_shape = (1, self.lr_subtile_size*4, self.lr_subtile_size*4)
		#self.hr_data_shape = (1, self.hr_subtile_size*4, self.hr_subtile_size*4)

		# Collect subtiles-coordinates (not needed since it would be random)
		#stc = SubTilesCoords(self.lr_data_shape, self.hr_data_shape, self.lr_subtile_size, self.hr_subtile_size)
		#self.subtiles_coords = stc.get_coords()

		# Flag to allow augmentation (enable only while training)
		self.transforms = True if data_set == "train" else False
		self.kernel_size = kernel_size
		self.train_params = pd.read_csv(train_params)

		self.lr_layer_spec_name = "6km"
		self.hr_layer_spec_name = "1km"

		# Initialize padding layer:
		self.average_pooling = Mask_AvgPool2d(self.kernel_size, stride=1, padding=self.kernel_size//2).to(DEVICE)

		# Filtering QA table according to thresholds
		self.qa = pd.read_csv(qa)
		self.qa = self.qa.loc[
			(self.qa.lr_Day_zero_filled < self.zerof_thr) & \
			(self.qa.lr_Day_cloud_perc < self.cloud_thr) & \
			(self.qa.lr_Day_error_perc < self.error_thr) & \
			(self.qa.lr_Night_zero_filled < self.zerof_thr) & \
			(self.qa.lr_Night_cloud_perc < self.cloud_thr) & \
			(self.qa.lr_Night_error_perc < self.error_thr) & \
			(self.qa.hr_Day_zero_filled < self.zerof_thr) & \
			(self.qa.hr_Day_cloud_perc < self.cloud_thr) & \
			(self.qa.hr_Day_error_perc < self.error_thr) & \
			(self.qa.hr_Night_zero_filled < self.zerof_thr) & \
			(self.qa.hr_Night_cloud_perc < self.cloud_thr) & \
			(self.qa.hr_Night_error_perc < self.error_thr)
		]

		print("Collected # paired LR-HR data: ", self.qa.shape[0])

		# Year/tiles based train/validation/test set
		if data_set == "test":
			self.qa = self.qa.loc[
				(self.qa.lr_date.apply(lambda x: str(x)[1:5]).isin(["2023"])) &\
				(self.qa.lr_tiles.isin(["h11v04", "h26v05"])) ]

		elif data_set == "val":
			self.qa = self.qa.loc[
				(self.qa.lr_date.apply(lambda x: str(x)[1:5]).isin(["2023"])) &\
				(self.qa.lr_tiles.isin(["h10v05", "h19v04"])) ]

		elif data_set == "train":
			self.qa = self.qa.loc[
				self.qa.lr_date.apply(lambda x: str(x)[1:5])\
				.isin(["2019", "2020", "2021", "2022"]) ]

		self.qa = self.qa.reset_index().drop(columns="index")

		# Set seeds 
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		# Reshufle
		qa_idx = list(self.qa.index)

		if data_set == "train":
			np.random.shuffle(qa_idx)

		self.qa_reshufle = self.qa.iloc[qa_idx]

		# Zero mean norm:
		# True: to zerocentered input data (final output \in [-1, 1])
		#   z = ({LR-HR} - mean({LR-HR})) / std({LR-HR})
		#   z = z / max(z)
		# False: to rescale to [0, 1]
		#   {LR-HR} = {LR-HR} / max({LR-HR})
		self.zeromean_norm = False


	@staticmethod
	def _read_rescale_data(filepath, layername, sf):
		""" General utils function to open a hdf file and rescale it
		Args:
			filepath: (str) full path
			layername: (str) of the layer name to be open
			sf: (float) scale factor
		Return torch tensor of size (1, 1, H, W)
		"""
		data = SD(filepath, SDC.READ)
		data = torch.Tensor(data.select(layername)[:,:].astype(np.float32)).unsqueeze(0).unsqueeze(0)
		return data * sf

	@staticmethod
	def _get_outlier_map(data, nstd):
		""" Return map of outliers values according to a given nstd
		Args:
		nstd: (int) number of standard dev outlier threshold
		"""
		condition = (data < (data.mean() - nstd*data.std())) | (data > (data.mean() + nstd*data.std()))
		outlier_map = torch.where(condition, 1, 0)
        
		# Check dimension
		outlier_map = outlier_map.unsqueeze(0) if len(outlier_map.shape) < 3 else outlier_map
                    
		return outlier_map


	def get_train_params(self, res, tiles):
		""" Return the mean and variance of the corresponding train subset
		filtered by the given res and tiles
		"""
		# Convert tile format vXXhXX to hXXvXX
		params = self.train_params[(self.train_params.res == res) &\
			(self.train_params.tiles == tiles)][["train_mean", "train_var"]].values[0]
		train_mean, train_var = params[0], params[1]

		return train_mean, train_var


	def crop_subpatch(self, lr_data, hr_data):
		""" Return Random Cropped LR interpolated to match HR size, HR and invalid mask
		Args:
			lr_data: (torch.Tensor) lr data already interpolated to match HR size
			hr_data: (torch.Tensor) hr data
		"""
		# Merge data and perform random cropping
               
		data = torch.cat([lr_data, hr_data], 1)
        
		subpatch = tt.Compose([tt.RandomCrop((self.patch_size, self.patch_size))])(data)
		lr_subpatch, hr_subpatch = subpatch[:,0,:,:], subpatch[:,1,:,:]
        
		# Estimate binary mask for valide / invalid values
		valid_mask = torch.cat([lr_subpatch, hr_subpatch], 0).all(0).int().unsqueeze(0)
		invalid_mask = torch.abs(valid_mask - 1).int()

		# Check dimension!!!
		return lr_subpatch, hr_subpatch, invalid_mask


	def yield_subpatch(self, lr_data, hr_data):
		# Interpolate with NN to match HR dimension

		# check whether lr-hr present the same dimension to test last attempt (see below)
		# if degrade_old (different sizes),
		if lr_data.shape != hr_data.shape:
			lr_data = F.interpolate(lr_data, scale_factor=self.up_scale, mode="nearest-exact")

		lr_subpatch, hr_subpatch, invalid_mask = self.crop_subpatch(lr_data, hr_data)

		while invalid_mask.float().mean().item() > self.impute_thr:
			lr_subpatch, hr_subpatch, invalid_mask = self.crop_subpatch(lr_data, hr_data)

		return lr_subpatch, hr_subpatch, invalid_mask


	def impute_missing(self, data, invalid_mask, filled_thr=10e-5, max_patience=5):
		""" Kernel based interpolation (via Torch.nn.Module)
		Iterative update by imputing data over a given kernel_size
		"""
		# If no missing value
		if data.all().item():
			return data

		# Estimate percentage of zero-filled pixels
		zerofilled_perc = invalid_mask.float().mean().item()
		patience = 0

		# Loop until all values had been imputed
		while zerofilled_perc > filled_thr and patience < max_patience:

			# Estimate current binary masks
			tmp_valid_mask = data.all(0).int()
			tmp_invalid_mask = torch.abs(tmp_valid_mask - 1)

			# Fill missing coordinates
			avg_map = self.average_pooling(data)
			data = torch.add(data, torch.mul(tmp_invalid_mask, avg_map))

			# Update criteria
			zerofilled_perc = tmp_invalid_mask.float().mean().item()
			patience += 1

		return data


	def impute_outliers(self, data, nstd=5, max_patience=5):
		""" Kernel based interpolation (via Torch.nn.Module)
		Iterative update by imputing data over a given kernel_size
		"""

		init_outlier_map = self._get_outlier_map(data, nstd)
		outlier_map = init_outlier_map.clone()

		# If no missing value
		if torch.abs(outlier_map - 1).all().item():
			return data, init_outlier_map.int()
        
		avg_map = self.average_pooling(data)
		patience = 0

		# Estimate percentage of outliers pixels
		outlier_perc = outlier_map.float().mean().item()

		# Loop until all values had been imputed
		while not torch.abs(outlier_map - 1).all() and patience < max_patience:

			# Impute outlier with spatial mean
			data = torch.where(outlier_map.bool(), avg_map, data)

			# Update criteria
			outlier_map  = self._get_outlier_map(data, nstd)
			outlier_perc = outlier_map.float().mean().item()
			patience += 1

		return data, init_outlier_map.int()


	def normalise_tilebased(self, data, res, tiles):
		""" Standardize image value according to intra-tiles mean
		"""
		train_mean, train_var = self.get_train_params(res, tiles)
		data = (data - train_mean) / np.sqrt(train_var)
		return data, train_mean, train_var


	def normalise_subpatchbased(self, lr_data, hr_data):
		""" Standardize image value according to the current {LR-HR} sub-patch mean/var
		"""
		
		# solve individual pair normalization
		# rescale to [-1; +1] interval
       
		concat = torch.cat([lr_data, hr_data], 0)
		concat = concat[concat!=0].flatten()
		datamean, datastd = concat.mean(), concat.std()
        
        if self.zeromean_norm:
	        # Zero mean centering
			lr_data = (lr_data - datamean) / datastd
			hr_data = (hr_data - datamean) / datastd

		# Re-scaling to [-1, +1]
		lr_data = lr_data / lr_data.max()
		hr_data = hr_data / hr_data.max()

		return lr_data, hr_data, datamean, datastd  

	def normalise_datasetbase(self, lr_data, hr_data, datamax):
		""" Return the [0,1] data normalisation according to the 
		whole Dataset max value, regardless tiles acquisition
		"""
		raise NotImplementedError


	def __len__(self):
		return self.qa.shape[0]


	def __getitem__(self, idx):
		"""Yield {LR, HR, invalid_pixels_mask} 
		"""
		pair = dict()
		curr_line = self.qa_reshufle.iloc[idx]
		curr_tiles = curr_line["lr_tiles"]
		layer = np.random.choice(["Day", "Night"])

		for res in ["lr", "hr"]:

			# Define params
			curr_path = self.lr_filepath if res == "lr" else self.hr_filepath
			curr_filename = curr_line[f"{res}_files"]
			spec_name = self.lr_layer_spec_name if res == "lr" else self.hr_layer_spec_name
			curr_layer = f"LST_{layer}_{spec_name}"

			# Load data
			pair[res] = self._read_rescale_data(os.path.join(curr_path, curr_filename), curr_layer, SF)

		# Iterate until a valid mask is given
		lr_subpatch, hr_subpatch, invalid_mask = self.yield_subpatch(pair["lr"], pair["hr"])
        
		# Impute missing for both LR and HR
		lr_subpatch = self.impute_missing(lr_subpatch, invalid_mask)
		hr_subpatch = self.impute_missing(hr_subpatch, invalid_mask)

		# Impute outliers (tile-based approach)
		lr_subpatch, lr_outlier_mask = self.impute_outliers(lr_subpatch)
		hr_subpatch, hr_outlier_mask = self.impute_outliers(hr_subpatch)

		# Merge invalid masks
		#invalid_mask = torch.cat([invalid_mask, lr_outlier_mask, hr_outlier_mask], 0).any(0).unsqueeze(0)
		invalid_mask = ((invalid_mask + lr_outlier_mask + hr_outlier_mask) >= 1).int()

		# Normalise input (tile-based approach)
		#lr_subpatch = self.normalise(lr_subpatch, "lr", curr_tiles)
		#hr_subpatch = self.normalise(hr_subpatch, "hr", curr_tiles)
        
		# Normalise input (subpatch-based approach)
		lr_subpatch, hr_subpatch, datamean, datastd = self.normalise_subpatchbased(lr_subpatch, hr_subpatch)

		# Perform data augmentation
		if self.transforms: 
			outcomes = torch.rand((2))

			if outcomes[0].item() < 0.5:
				lr_subpatch = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(lr_subpatch)
				hr_subpatch = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(hr_subpatch)
				invalid_mask = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(invalid_mask)

			if outcomes[1].item() < 0.5:
				lr_subpatch = tt.Compose([tt.RandomVerticalFlip(p=1.)])(lr_subpatch)
				hr_subpatch = tt.Compose([tt.RandomVerticalFlip(p=1.)])(hr_subpatch)
				invalid_mask = tt.Compose([tt.RandomVerticalFlip(p=1.)])(invalid_mask)

			angle = np.random.choice([0., 90., 180., 270.])
			lr_subpatch = ttf.rotate(lr_subpatch, angle)
			hr_subpatch = ttf.rotate(hr_subpatch, angle)
			invalid_mask = ttf.rotate(invalid_mask, angle)

		return lr_subpatch, hr_subpatch, invalid_mask, datamean.reshape(1,1,1), datastd.reshape(1,1,1)


class MODIS_dataset_single(MODIS_dataset):

	def __init__(self,
		hr_filepath,
		qa,
		zerof_thr,
		cloud_thr,
		error_thr,
		impute_thr,
		up_scale,
		kernel_size,
		train_params,
		data_set = "train",
		seed = 8609):
		"""
		Args:
		...
		qa: (str) path to file to qa scanning (LR-HR matching)
		...
		"""
		self.hr_filepath = hr_filepath

		self.zerof_thr = zerof_thr
		self.cloud_thr = cloud_thr
		self.error_thr = error_thr
		self.impute_thr = impute_thr

		self.up_scale = up_scale
		self.seed = seed

		# Subpatches sizes
		self.patch_size = 64

		#self.lr_data_shape = (1, self.lr_subtile_size*4, self.lr_subtile_size*4)
		#self.hr_data_shape = (1, self.hr_subtile_size*4, self.hr_subtile_size*4)

		# Collect subtiles-coordinates (not needed since it would be random)
		#stc = SubTilesCoords(self.lr_data_shape, self.hr_data_shape, self.lr_subtile_size, self.hr_subtile_size)
		#self.subtiles_coords = stc.get_coords()

		# Flag to allow augmentation (enable only while training)
		self.transforms = True if data_set == "train" else False
		self.kernel_size = kernel_size
		self.train_params = pd.read_csv(train_params)

		self.hr_layer_spec_name = "1km"

		# Initialize padding layer:
		self.average_pooling = Mask_AvgPool2d(self.kernel_size, stride=1, padding=self.kernel_size//2).to(DEVICE)

		# Filtering QA table according to thresholds
		self.qa = pd.read_csv(qa)
		self.qa = self.qa.loc[
			(self.qa.hr_Day_zero_filled < self.zerof_thr) & \
			(self.qa.hr_Day_cloud_perc < self.cloud_thr) & \
			(self.qa.hr_Day_error_perc < self.error_thr) & \
			(self.qa.hr_Night_zero_filled < self.zerof_thr) & \
			(self.qa.hr_Night_cloud_perc < self.cloud_thr) & \
			(self.qa.hr_Night_error_perc < self.error_thr)
		]

		print("Collected # paired HR data: ", self.qa.shape[0])

		# Year/tiles based train/validation/test set
		if data_set == "test":
			self.qa = self.qa.loc[
				(self.qa.lr_date.apply(lambda x: str(x)[1:5]).isin(["2023"])) &\
				(self.qa.lr_tiles.isin(["h11v04", "h26v05"])) ]

		elif data_set == "val":
			self.qa = self.qa.loc[
				(self.qa.lr_date.apply(lambda x: str(x)[1:5]).isin(["2023"])) &\
				(self.qa.lr_tiles.isin(["h10v05", "h19v04"])) ]

		elif data_set == "train":
			self.qa = self.qa.loc[
				self.qa.lr_date.apply(lambda x: str(x)[1:5])\
				.isin(["2019", "2020", "2021", "2022"]) ]

		self.qa = self.qa.reset_index().drop(columns="index")

		# Set seeds 
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		# Reshufle
		qa_idx = list(self.qa.index)

		if data_set == "train":
			np.random.shuffle(qa_idx)

		self.qa_reshufle = self.qa.iloc[qa_idx]	

		# Zero mean norm:
		# True: to zerocentered input data (final output \in [-1, 1])
		#   z = ({LR-HR} - mean({LR-HR})) / std({LR-HR})
		#   z = z / max(z)
		# False: to rescale to [0, 1]
		#   {LR-HR} = {LR-HR} / max({LR-HR})
		self.zeromean_norm = False



	def degrade_old(self, data):
		""" Image degradation according to the initial scale factor
		"""
		interp = F.interpolate(data, scale_factor=1/self.up_scale, mode="bilinear")
		#interp = F.interpolate(interp, scale_factor=self.up_scale, mode="nearest-exact")
		return interp

	def degrade(self, data):
		down = F.interpolate(data, scale_factor=1/self.up_scale, mode="bicubic")
		up   = F.interpolate(down, scale_factor=self.up_scale, mode="bicubic")
		return up


	def __getitem__(self, idx):
		"""Yield {LR, HR, invalid_pixels_mask} 
		"""
		# Define params
		curr_line  = self.qa_reshufle.iloc[idx]
		curr_tiles = curr_line["hr_tiles"]
		curr_path  = self.hr_filepath
		curr_filename = curr_line[f"hr_files"]
		spec_name  = self.hr_layer_spec_name
		
		layer = np.random.choice(["Day", "Night"])
		curr_layer = f"LST_{layer}_{spec_name}"
		
		hr_data = self._read_rescale_data(os.path.join(curr_path, curr_filename),
					curr_layer, SF)
		lr_data = self.degrade(hr_data)

		# Iterate until a valid mask is given
		lr_subpatch, hr_subpatch, invalid_mask = self.yield_subpatch(lr_data, hr_data)
        
		# Impute missing for both LR and HR
		lr_subpatch = self.impute_missing(lr_subpatch, invalid_mask)
		hr_subpatch = self.impute_missing(hr_subpatch, invalid_mask)

		# Impute outliers (tile-based approach)
		lr_subpatch, lr_outlier_mask = self.impute_outliers(lr_subpatch)
		hr_subpatch, hr_outlier_mask = self.impute_outliers(hr_subpatch)

		# Merge invalid masks
		#invalid_mask = torch.cat([invalid_mask, lr_outlier_mask, hr_outlier_mask], 0).any(0).unsqueeze(0)
		invalid_mask = ((invalid_mask + lr_outlier_mask + hr_outlier_mask) >= 1).int()

		# Normalise input (tile-based approach)
		#lr_subpatch = self.normalise(lr_subpatch, "lr", curr_tiles)
		#hr_subpatch = self.normalise(hr_subpatch, "hr", curr_tiles)
        
		# Normalise input (subpatch-based approach)
		lr_subpatch, hr_subpatch, datamean, datastd = self.normalise_subpatchbased(lr_subpatch, hr_subpatch)

		# Perform data augmentation
		if self.transforms: 
			outcomes = torch.rand((2))

			if outcomes[0].item() < 0.5:
				lr_subpatch = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(lr_subpatch)
				hr_subpatch = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(hr_subpatch)
				invalid_mask = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(invalid_mask)

			if outcomes[1].item() < 0.5:
				lr_subpatch = tt.Compose([tt.RandomVerticalFlip(p=1.)])(lr_subpatch)
				hr_subpatch = tt.Compose([tt.RandomVerticalFlip(p=1.)])(hr_subpatch)
				invalid_mask = tt.Compose([tt.RandomVerticalFlip(p=1.)])(invalid_mask)

			angle = np.random.choice([0., 90., 180., 270.])
			lr_subpatch = ttf.rotate(lr_subpatch, angle)
			hr_subpatch = ttf.rotate(hr_subpatch, angle)
			invalid_mask = ttf.rotate(invalid_mask, angle)

		return lr_subpatch, hr_subpatch, invalid_mask, datamean.reshape(1,1,1), datastd.reshape(1,1,1)





"""
if __name__ == "__main__":

	
	# Example usage
	PATH_LR = ""
	PATH_HR = ""
	datafilter =  DataFilter(PATH_LR, PATH_HR, 0.1)
	datafilter.scan_all()
	
	
	PATH_LR = ""
	PATH_HR = ""

	mds = MODIS_dataset(
		lr_filepath=PATH_LR,
		hr_filepath=PATH_HR,
		qa="",
		zerof_thr=0.8,
		cloud_thr=0.8,
		error_thr=0.8,
		impute_thr=0.01,
		up_scale=6,
		kernel_size=5,
		train_params="",
		data_set = "train",
		seed = 8609)
"""
