import os
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

 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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



def data_param_estimation(filepath, layernames, train_years, outname):
    """ Estimate Train dataset distribution parameters
    """
    def read_rescale(filename, layername):
        hdrfile = SD(filename, SDC.READ)
        hdrfile = hdrfile.select(layername)
        sf = 0.02 # hardcoded
        data = torch.Tensor(hdrfile[:,:].astype(np.float32)).flatten()
        data = torch.mul(data, sf)
        data = data[data != 0]
        return data
    
    re = list()

    # Complete filelist
    filelist = os.listdir(filepath)

    # Complete tileslist
    tiles_list = set([item.strip().split(".")[2] for item in filelist])

    for tiles in sorted(list(tiles_list)):

        print("Now on tile:", tiles)
        # subfilter according to train_years
        train_list = [item for item in filelist if\
            item.strip().split(".")[2] == tiles] 

        partial, n = 0.0, 0.0
        for item in tqdm(train_list, total = len(train_list)):
            for layername in layernames:
                data = read_rescale(os.path.join(filepath, item), layername)
                partial += data.sum().item()
                n += data.shape[0]

        if n != 0:
            mean = partial/n
            partial, n = 0.0, 0.0
            for item in tqdm(train_list, total = len(train_list)):
                for layername in layernames:
                    data = read_rescale(os.path.join(filepath, item), layername)
                    partial += ((data - mean)**2).sum().item()
                    n += data.shape[0]
            variance = partial/n

        else: 
            mean = np.nan
            variance = np.nan
            
        re.append(pd.DataFrame({"tiles":tiles, "train_mean":mean, "train_var":variance}, index=[0]))

    re = pd.concat(re).to_csv(outname)



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



class MODIS_datafilter():
    def __init__(self, lr_filepath, hr_filepath, 
        cloud_coverage, error_thr):
        """
        Args:
            lr_filepath: (str) path-to-file to qa scanning (LR)
            hr_filepath: (str) path-to-file to qa scanning (HR)
            cloud_coverage: (float)
            error_thr: (float)
        """
        self.lr_filepath = lr_filepath
        self.hr_filepath = hr_filepath

        self.cloud_coverage = cloud_coverage
        self.error_thr = error_thr

    @staticmethod
    def melt_dataset(df):
        """ From wide to long version
        """
        toretain = ["product", "version", "variable", "layer", "date", "fullname", 
            "tiles", "tile_date_layer", "qa_fcloud", "qa_ferror"]
            
        # Expanding cloud coverage dimension over tiles
        df_long = pd.melt(df,
            id_vars= toretain,
            value_vars= [f"t{n}_fcloud" for n in range(1,16+1)], 
            var_name = "tile_n")\
        .rename(columns={"value":"cloud_val"})

        # Expanding temperature error dimension over tiles
        df_long0 = pd.melt(df,
            id_vars= toretain,
            value_vars= [f"t{n}_ferror" for n in range(1,16+1)], 
            var_name = "tile_ferror")\
        .rename(columns={"value":"error_val"})
        
        df_long["tile_n"] = df_long["tile_n"].apply(lambda x: x.split("_")[0].strip("t"))
        df_long["error_val"] = df_long0.error_val

        # tile_date_layer will be useful while filtering for unique tiles combination
        df_long["tile_date_layer"] = df_long["tile_date_layer"] + "_" + df_long["tile_n"]
        return df_long

    @staticmethod
    def filter_dataset(df, cloud_coverage, error_thr):
        """ Return the resulting dataframe filtered by cloud coverage and T error """
        filtered = df[(df.cloud_val <= cloud_coverage) & (df.error_val <= error_thr)]
        return filtered

    def perform_filtering(self):
        """ Merging datasets """
        # Load
        qa_lr = pd.read_csv(self.lr_filepath).sort_values(by=["product","version","date"])
        qa_hr = pd.read_csv(self.hr_filepath).sort_values(by=["product","version","date"])

        # Filter common days/tiles/layer
        lr_unique = set(qa_lr.tile_date_layer.unique())
        fqa_lr, fqa_hr = qa_lr, qa_hr[qa_hr.tile_date_layer.isin(lr_unique)]

        # Perform conversion into long
        fqa_lr = MODIS_datafilter.melt_dataset(fqa_lr)
        fqa_hr = MODIS_datafilter.melt_dataset(fqa_hr)

        # Filter according to cloud coverage and T error
        fqa_lr = MODIS_datafilter.filter_dataset(fqa_lr, self.cloud_coverage, self.error_thr)
        fqa_hr = MODIS_datafilter.filter_dataset(fqa_hr, self.cloud_coverage, self.error_thr)

        merged_df = fqa_lr.merge(fqa_hr, how="inner", on="tile_date_layer")
        newcols = dict()
        for c in merged_df.columns:
            if c.endswith("x"):
                newcols[c] = "_".join(c.split("_")[0:-1])+"_lr"
            elif c.endswith("y"):
                newcols[c] = "_".join(c.split("_")[0:-1])+"_hr"
            else:
                newcols[c] = c
                
        merged_df = merged_df.rename(columns = newcols)
        merged_df["tile_n_lr"] = merged_df["tile_n_lr"].astype(int)
        merged_df["tile_n_hr"] = merged_df["tile_n_hr"].astype(int)

        return merged_df



class MODIS_dataset(Dataset):

    def __init__(self, path_lr, path_hr, qa_lr, qa_hr, cloud_coverage,
        error_thr, up_scale, kernel_size, train_params,
        data_set = "train", seed = 8609):

        self.up_scale = up_scale

        # Original param
        self.path_lr = path_lr
        self.path_hr = path_hr

        # Thresholds
        self.cloud_coverage = cloud_coverage
        self.error_thr = error_thr

        # Peform Datafiltering given the QA tables of LR, HR datasets.
        # sub-tiles found to have percentage of clouds/errors < thresholds are excluded
        self.mdf = MODIS_datafilter(qa_lr, qa_hr, self.cloud_coverage, self.error_thr)
        self.merged_df = self.mdf.perform_filtering()
        
        # Generate subtiles (consider to map the img size instead)
        self.lr_subtile_size = 50
        self.hr_subtile_size = 300
        self.lr_data_shape = (1, self.lr_subtile_size*4, self.lr_subtile_size*4)
        self.hr_data_shape = (1, self.hr_subtile_size*4, self.hr_subtile_size*4)

        # Collect subtiles-coordinates
        stc = SubTilesCoords(self.lr_data_shape, self.hr_data_shape, 
            self.lr_subtile_size, self.hr_subtile_size)
        self.subtiles_coords = stc.get_coords()

        # Performing zero-filled values check (full-pass over filtered data):
        # Additional check to address un-specified zero-filled values
        if os.path.exists(f"zerofilled_valid_ct{self.cloud_coverage}.csv"):
            print("Reading sub-patches filtering ...")
            zerofilled = pd.read_csv(f"zerofilled_valid_ct{cloud_coverage}.csv")
            
        # Case if no pre-filtering has been done yet over the current cloud threshold
        else:
            print("Performing sub-patches filtering ...")
            unique_ids, zerofilled = [], []
            for i, curr_line in tqdm(self.merged_df.iterrows(), total=self.merged_df.shape[0]):
               
                tmp = []
                unique_id = curr_line["tile_date_layer"]
                
                for res in ["lr", "hr"]:

                    # collect the subtile number, layer and filename
                    curr_path = self.path_lr if res == "lr" else self.path_hr
                    curr_subtile_size = self.lr_subtile_size if res == "lr" else self.hr_subtile_size
                    layer     = curr_line[f"variable_{res}"]
                    filename  = curr_line[f"fullname_{res}"]
                    subtile_n = curr_line[f"tile_n_{res}"]
                    
                    # Read HDR file and rescale it
                    data = self._read_rescale_data(curr_path, filename, layer, curr_subtile_size)

                    # Extract the corresponding sub-tile boundaries 
                    h_min, h_max, w_min, w_max = tuple(self.subtiles_coords[res]["tiles_n"][subtile_n-1])
                    subtile = data[:,  h_min : h_max,  w_min : w_max]
                    
                    # Estimate percentage of missing / filled values
                    subtile_f = subtile.flatten()
                    percentage = (torch.where(subtile_f==0, 1, 0).sum() / subtile_f.shape[-1]).item()
                    tmp.append(percentage <= self.cloud_coverage)
                    
                unique_ids.append(unique_id)
                zerofilled.append(tmp)

            # (zerofilled_valid: True if percentage of zero filled points is lower than 
            # cloud threshold on both lr-hr pair.
            zerofilled = np.array(zerofilled).all(-1)
            zerofilled = pd.DataFrame({
                "tile_date_layer":unique_ids,
                "zerofilled_valid":zerofilled,
            })
            zerofilled.to_csv(f"zerofilled_valid_ct{self.cloud_coverage}.csv", index=False)

        self.merged_df = self.merged_df.merge(zerofilled, how="inner", on="tile_date_layer")
        self.merged_df = self.merged_df[self.merged_df["zerofilled_valid"] == True]

        print("Done!", end="\n")

        # Random drawning a validation set from training data:
        # Modify merged_df by retaining randomly chosen sub-patches
        #np.random.seed(seed)
        #val_n      = round(self.merged_df.index.shape[0] * val_ratio)
        #all_idxs   = set(self.merged_df.index)
        #val_idxs   = set(np.random.choice(self.merged_df.index, val_n, replace=False))
        #train_idxs = all_idxs.difference(val_idxs)
        # Filter dataset accordingly
        #val_set   = self.merged_df.loc[sorted(list(val_idxs))]
        #train_set = self.merged_df.loc[sorted(list(train_idxs))]
        #self.merged_df = val_set if is_validation_set else train_set

        # Year based validation set drawning
        if data_set == "test":
            self.merged_df = self.merged_df.loc[
                (self.merged_df.date_lr.apply(lambda x: str(x)[1:5]).isin(["2023"])) &\
                (self.merged_df.tiles_lr.isin(["v04h11", "v05h26"]))]

        elif data_set == "val":
            self.merged_df = self.merged_df.loc[
                (self.merged_df.date_lr.apply(lambda x: str(x)[1:5]).isin(["2023"])) &\
                (self.merged_df.tiles_lr.isin(["v05h10", "v04h19"]))]

        elif data_set == "train":
            self.merged_df = self.merged_df.loc[
                self.merged_df.date_lr.apply(lambda x: str(x)[1:5])\
                .isin(["2019", "2020", "2021", "2022"]) ]
        
        # Flag to allow augmentation (enable only while training)
        self.transforms = True if data_set == "train" else False

        self.kernel_size = kernel_size

        # Initialize padding layer:
        self.average_pooling = Mask_AvgPool2d(
            self.kernel_size, stride=1, padding=self.kernel_size//2,
            ).to(DEVICE)
        
        # Load csv of rescaling params
        self.train_params = pd.read_csv(train_params)


    @staticmethod
    def _read_rescale_data(curr_path, filename, layer, subtile_size):
        """ Return LST array rescaled by scale factor in torch Tensor shape (C, H, W)
        and the realtive binary mask (1=zero-filled values, 0 otherwise)
        Now reading only (no rescaling to match HauNet requirements).
        """
        sf = 0.02

        hdrfile = SD(os.path.join(curr_path, filename), SDC.READ)
        hdrfile = hdrfile.select(layer)
        array_obj = hdrfile[:,:].astype(np.float32)
        data = torch.Tensor(array_obj * sf).unsqueeze(0).unsqueeze(0)

        # Interpolate data to target dimension 
        #data = F.interpolate(data * sf, size=subtile_size*4, mode="nearest-exact").squeeze(0)
        return data


    @staticmethod
    def _get_outlier_map(data, nstd):
        """ Return map of outliers values according to a given nstd
        Args:
            nstd: (int) number of standard dev outlier threshold
        """
        condition = (data < (data.mean() - nstd*data.std())) | (data > (data.mean() + nstd*data.std()))
        return torch.where(condition, 1, 0)


    def _get_train_params(self, res, tiles):
        """ Return the mean and variance of the corresponding train subset
        filtered by the given res and tiles
        """
        tiles = tiles[3:6]+tiles[0:3]
        train_mean, train_var = self.train_params[(self.train_params.res == res) &\
                                    (self.train_params.tiles == tiles)]\
                                    [["train_mean", "train_var"]]
        return train_mean, train_var


    def _normalise(self, data, res, tiles):
        """ Standardize image value according to intra-tiles mean
        """
        train_mean, train_var = self._get_train_params(res, tiles)
        data = torch.divide((data-train_mean), torch.sqrt(train_var))
        return data


    def _impute_missing(self, data, filled_thr=0.001, max_patience=5):
        """ Kernel based interpolation (via Torch.nn.Module)
        Iterative update by imputing data over a given kernel_size
        """
        # If no missing value
        if torch.all(data).item():
            return data
    
        patience = 0
        
        # Estimate percentage of zero-filled pixels
        zerofilled_perc = np.where(data==0, 1, 0).mean().item()
        
        # Loop until all values had been imputed
        while not torch.all(data).item() \
            and zerofilled_perc > filled_thr \
            and patience < max_patience:
            
            # Fill missing coordinates
            missing_mask = torch.where(data==0, 1, 0)
            avg_map = self.average_pooling(data)
            data = torch.add(data, torch.mul(missing_mask, avg_map))
            
            # Update criteria
            zerofilled_perc = np.where(data==0, 1, 0).mean().item()
            patience += 1

        return data


    def _impute_outliers(self, data, nstd=5, max_patience=5):
        """ Kernel based interpolation (via Torch.nn.Module)
        Iterative update by imputing data over a given kernel_size
        """
        avg_map = self.average_pooling(data)
        #avg_map = data.mean()

        init_outlier_map = self._get_outlier_map(data, nstd)
        outlier_map = init_outlier_map.clone()
        
        # If no missing value
        if torch.abs(outlier_map - 1).all():
            return data, init_outlier_map.float()

        patience = 0

        # Estimate percentage of outliers pixels
        outlier_perc = outlier_map.float().mean().item()

        # Loop until all values had been imputed
        while not torch.abs(outlier_map - 1).all() \
            and patience < max_patience:

            data = torch.where(outlier_map.bool(), avg_map, data)

            # Update criteria
            outlier_map  = self._get_outlier_map(data, nstd)
            outlier_perc = outlier_map.float().mean().item()
            patience += 1

        return data, init_outlier_map.float()

    #def _interpolate(self, data, up_scale):
        """ Eventually interpolate LR image to match HR sizes
        !!! This method can be omitted or improved !!!
        """
        #data = data.unsqueeze(0) if not len(data.shape)==4 else data
        #_, c, h, w = data.shape
        #data = F.interpolate(data, size = h * up_scale , mode="nearest-exact").squeeze(0)
        #return data


    def __len__(self):
        return self.merged_df.shape[0]


    def __getitem__(self, idx, transforms=True):

        pair = dict()
        curr_line = self.merged_df.iloc[idx]

        for res in ["lr", "hr"]:

            # collect the subtile number, layer and filename
            curr_path = self.path_lr if res == "lr" else self.path_hr
            curr_subtile_size = self.lr_subtile_size if res == "lr" else self.hr_subtile_size
            layer     = curr_line[f"variable_{res}"]
            filename  = curr_line[f"fullname_{res}"]
            subtile_n = curr_line[f"tile_n_{res}"]
            tiles     = curr_line[f"tiles_{res}"]
            
            # Read file and rescale by the corresponding scale factor
            data = self._read_rescale_data(curr_path, filename, layer, curr_subtile_size)
            
            # Extract the corresponding sub-tile boundaries 
            h_min, h_max, w_min, w_max = tuple(self.subtiles_coords[res]["tiles_n"][subtile_n-1])
            data = data[:,  h_min : h_max,  w_min : w_max]

            # Perform invalid pixels binary mask
            # Interpolate at target size
            data = F.interpolate(data.unsqueeze(0), scale_factor=self.up_scale, mode="nearest-exact") \
                if res == "lr" else data

            # Impute missing and outliers
            impute_bmask = torch.where(data == 0, 1, 0).squeeze(0)
            data = self._impute_missing(data)
            data, outlier_bmask = self._impute_outliers(data)

            # Merge binary masks
            bmask = torch.cat([impute_bmask, outlier_bmask]).any(0)
            #bmask = torch.where((impute_bmask + outlier_bmask) > 0, 1, 0)

            # Normalise input data (!!!)
            data = self._normalise(data, res, tiles)

            # Save subtiles along with bmask
            pair[res] = (data, bmask)

            # DEPRECATED If LR output shape needs to match HR
            #    data = data.unsqueeze(0) if len(data.shape) < 4 else data
            #    data = self._interpolate(data, up_scale=self.up_scale)
            # outlier_map = self._interpolate(outlier_map, up_scale=self.up_scale)

        # Collect tensors
        lr_data, lr_bmask = pair["lr"]
        hr_data, hr_bmask = pair["hr"]

        # Merge binary masks and check for outdim
        bmask = torch.cat([lr_bmask, hr_bmask]).any(0) # !!!  check the torch conversion issue from bool to float
        bmask = bmask.unsqueeze(0) if len(bmask.shape) < 3 else bmask # !!!
             
        # Perform data augmentation
        if self.transforms: 
            outcomes = torch.rand((2))

            if outcomes[0].item() < 0.5:
                lr_data = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(lr_data)
                hr_data = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(hr_data)
                bmask   = tt.Compose([tt.RandomHorizontalFlip(p=1.)])(bmask)

            if outcomes[1].item() < 0.5:
                lr_data = tt.Compose([tt.RandomVerticalFlip(p=1.)])(lr_data)
                hr_data = tt.Compose([tt.RandomVerticalFlip(p=1.)])(hr_data)
                bmask   = tt.Compose([tt.RandomVerticalFlip(p=1.)])(bmask)

            angle   = np.random.choice([0., 90., 180., 270.])
            lr_data = ttf.rotate(lr_data, angle)
            hr_data = ttf.rotate(hr_data, angle)
            bmask   = ttf.rotate(bmask, angle)

        print("SHAPE: ", bmask.shape)
        return lr_data, hr_data, bmask




def get_MODIS_dataloader(path_lr, path_hr, qa_lr, qa_hr, cloud_coverage, error_thr,
                       train_mean, train_var, up_scale, kernel_size,
                       batch_size=4, shuffle=False):
    """ Return MODIS iterable dataloader 
    """
    mds = MODIS_dataset(path_lr, path_hr, qa_lr, qa_hr, cloud_coverage, error_thr,
                       train_mean, train_var, up_scale, kernel_size)
    dl  = torch.utils.data.DataLoader(mds, batch_size, shuffle, num_workers=16)
    return dl                        


### ------------------------------------------------------
### Deprecated (general utils functions)
### ------------------------------------------------------

class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


def eval_profile(net, img):
    """ Eval model profile (flops and nparams)
    Args:
        net: torch.nn.Module (network to eval)
        img: torch.tensor (1, C, H, W)
    """ 
    flops, params = profile(haunet_s, (x,))
    print('Net evaluation:\nflops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1000000.0))
    



if __name__ == "__main__":

    """
    mds = MODIS_dataset("/mnt/LOCALDATA/STUDENTS/SATELLITE/MODIS/MOD11B1",
                "/mnt/LOCALDATA/STUDENTS/SATELLITE/MODIS/MOD11A1",
                "/home/giacomo.t/src/resulting_scanning_MOD11B1.csv",
                "/home/giacomo.t/src/resulting_scanning_MOD11A1.csv",
                cloud_coverage=0.0001, 
                error_thr=0.1,
                up_scale=6, 
                kernel_size=7,
                train_params="./train_params_tiles_2019_2020_2021_2022.csv",
                data_set = "train", 
                )

    FILEPATH = "/mnt/LOCALDATA/STUDENTS/SATELLITE/MODIS/MOD11B1"
    #FILEPATH = "/mnt/LOCALDATA/STUDENTS/SATELLITE/MODIS/MOD11A1"
    LAYERNAMES = ["LST_Day_6km", "LST_Night_6km"]
    #LAYERNAMES = ["LST_Day_1km", "LST_Night_1km"]
    TRAINYEARS = ["2019", "2020", "2021", "2022"]
    OUTNAME = "train_params_tiles_2019_2020_2021_2022.csv"


    #data_param_estimation(FILEPATH, LAYERNAMES, TRAINYEARS, OUTNAME)
    """