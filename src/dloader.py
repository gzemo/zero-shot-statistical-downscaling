import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

#import torch
#import torch.nn.functional as F
#from torch.utils.data import Dataset

#import torchvision
#import torchvision.transforms as tt
#import torchvision.transforms.functional as ttf

#from thop import profile
from pyhdf.SD import SD, SDC
 
from quality_assessment import init_qc_table


#DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SF = 0.02
THRESHOLD =?????

class DataFilter():
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
		print("Hardcoded!")
		data = np.array(data.select(layername)[:,:].astype(np.float32))
		return data * sf

	@staticmethod
	def _get_qc_data(filepath, layername):
		""" Return QC layer
		"""
		data = SD(filepath, SDC.READ)
		data = np.array(data.select(layername)[:,:].astype(np.int64))
		return data

	@staticmethod
	def _estimate_tile_pixel_quality(tile, QC_Data):
		"""Return cloud coverage and above 3K error for a given tile
		Args:
			tile: (array) object of QC data
			QC_Data (pd.DataFrame) reference QC 8-bit table
		"""
		cloud_perc, error_perc = 0., 0.
	 
		# define quality values to be excluded
		nopixels = ("No Pixel,clouds", "No Pixel, Other QA")

		# define Temperature error to be excluded from no-cloud filtered QA_Data
		noerrors = ("LST Err > 3K",)

		# filter Full 8-bit QC_Data according to current tile
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
			#try: print((filtered.groupby("LST_Err")["Integer_Value"].count()/filtered.shape[0])[noerror])
			#except: print("notfound")
			error_perc += (filtered.groupby("LST_Err")["Integer_Value"].count() / filtered.shape[0])[noerror]\
				if noerror in filtered.LST_Err.unique() \
				else 0.

		return round(cloud_perc, 6), round(error_perc, 6)

	@staticmethod
	def _estimate_zerofilled(tile):
		return round((tile==0).mean(), 6)


	def merge_datasets(self):
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
			self._subtiles_coords[k]["tiles_h"] = tiles_h # consider to remove this 
			self._subtiles_coords[k]["tiles_w"] = tiles_w
			self._subtiles_coords[k]["tiles_n"] = subtiles

	def get_coords(self):
		return self._subtiles_coords


print("Hardcoded!")
#class MODIS_dataset(Dataset):
class MODIS_dataset():

	def __init__(self,
		lr_filepath,
		hr_filepath,
		qa,
		zerof_thr,
		cloud_thr,
		error_thr,
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
		self.train_params = train_params

		self.lr_layer_spec_name = "6km"
		self.hr_layer_spec_name = "1km"

		# Initialize padding layer:
		print("Hardcoded!") # should be enable to be a valid torch instance
		#self.average_pooling = Mask_AvgPool2d(self.kernel_size, stride=1, padding=self.kernel_size//2).to(DEVICE)

		# Filtering according to thresholds
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

		print("Collected # paired LR-HR data: ", self.qa.shape)

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

		# Reshufle
		np.random.seed(self.seed)
		qa_idx = list(self.qa.index)
		np.random.shuffle(qa_idx)
		self.qa_reshufle = self.qa.iloc[qa_idx]

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
        return torch.where(condition, 1, 0)


	def get_train_params(self, res, tiles):
		""" Return the mean and variance of the corresponding train subset
		filtered by the given res and tiles
		"""
		tiles = tiles[3:6]+tiles[0:3]
		train_mean, train_var = self.train_params[(self.train_params.res == res) &\
			(self.train_params.tiles == tiles)]\
			[["train_mean", "train_var"]]

		return train_mean, train_var


	def crop_subpatch(self, lr_data, hr_data):
		""" Return LR interpolated to match HR size, HR and invalid mask
		Args:
			lr_data: (torch.Tensor) lr data already interpolated to match HR size
			hr_data: (torch.Tensor) hr data
		"""
		# Merge data and perform random cropping
		data = torch.cat([lr_data, hr_data], 1)
		subpatch = tt.Compose([tt.RandomCrop((self.patch_size, self.patch_size))])(data)
		lr_subpatch, hr_subpatch = subpatch[0,0,:,:], subpatch[0,1,:,:]
		
		# Estimate binary mask for invalid values
		valid_mask = subpatch.all(1).int()
		invalid_mask = torch.abs(valid_mask - 1).int()

		# Check dimension!!!
		return lr_subpatch, hr_subpatch, invalid_mask


	def yield_subpatch(self, lr_data, hr_data):

		# Interpolate with NN to match HR dimension
		lr_data = F.interpolate(lr_data, up_scale=self.up_scale, mode="nearest-exact")

		lr_subpatch, hr_subpatch, invalid_mask = self.crop_subpatch(lr_data, hr_data)

		while invalid_mask.float().mean().item() > THRESHOLD:
			lr_subpatch, hr_subpatch, invalid_mask = self.crop_subpatch(lr_data, hr_data)

		return lr_subpatch, hr_subpatch, invalid_mask


	def impute_missing(self, data, invalid_mask, filled_thr=10e-5, max_patience=5):
		""" Kernel based interpolation (via Torch.nn.Module)
		Iterative update by imputing data over a given kernel_size
		"""
		# If no missing value
		if torch.all(data).item():
			return data

		# Estimate percentage of zero-filled pixels
		zerofilled_perc = invalid_mask.float().mean().item()
		patience = 0

		# Loop until all values had been imputed
		while zerofilled_perc > filled_thr and patience < max_patience:

			# Estimate current binary masks
			tmp_valid_mask = data.all().int()
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
		avg_map = self.average_pooling(data)
		#avg_map = data.mean()

		init_outlier_map = self._get_outlier_map(data, nstd)
		outlier_map = init_outlier_map.clone()

		# If no missing value
		if torch.abs(outlier_map - 1).all().item():
			return data, init_outlier_map.int()

		patience = 0

		# Estimate percentage of outliers pixels
		outlier_perc = outlier_map.float().mean().item()

		# Loop until all values had been imputed
		while not torch.abs(outlier_map - 1).all() and patience < max_patience:

			data = torch.where(outlier_map.bool(), avg_map, data)

			# Update criteria
			outlier_map  = self._get_outlier_map(data, nstd)
			outlier_perc = outlier_map.float().mean().item()
			patience += 1

		return data, init_outlier_map.int()


	def normalise(self, data, res, tiles):
		""" Standardize image value according to intra-tiles mean
		"""
		train_mean, train_var = self._get_train_params(res, tiles)
		data = torch.divide((data-train_mean), torch.sqrt(train_var))
		return data



	def __getitem__(self, idx):

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

		# Impute outliers
		lr_subpatch, lr_outlier_mask = self.impute_outliers(lr_subpatch)
		hr_subpatch, hr_outlier_mask = self.impute_outliers(hr_subpatch)

		# Merge invalid masks
		invalid_mask = torch.cat([invalid_mask, lr_outlier_mask, hr_outlier_mask], 1).all(1)

		# Normalise imput
		lr_subpatch = self.normalise(lr_subpatch, "lr", curr_tiles)
		hr_subpatch = self.normalise(hr_subpatch, "hr", curr_tiles)

		
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

		# Check dimension!!!	
		return lr_subpatch, hr_subpatch, invalid_mask


if __name__ == "__main__":

	# Example usage
	#datafilter =  DataFilter("./../data/MOD11B1/", "./../data/MOD11A1/", 0.1)
	#datafilter.scan_all()
	
	mds = MODIS_dataset(
		lr_filepath="./../data/MOD11B1/",
		hr_filepath="./../data/MOD11A1/",
		qa="./scanning_paired_dataset.csv",
		zerof_thr=0.8,
		cloud_thr=0.8,
		error_thr=0.8,
		up_scale=6,
		kernel_size=7,
		train_params="./train_params_tiles_2019_2020_2021_2022.csv",
		data_set = "train",
		seed = 8609)
