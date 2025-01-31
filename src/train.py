import os
import argparse

from piqa import SSIM

from tqdm import tqdm
from models.srcnn import *
from models.unet import *
from models.resunet import *
from models.haunet import *
from dloader import *

import matplotlib.pyplot as plt


# go on with others
DEVICE  = "cuda:0" if torch.cuda.is_available() else "cpu"
PATH_LR = ""
PATH_HR = ""
QA      = ""
PARAMS  = ""
UP_SCALE = 6
KERNEL_SIZE = 5
SEED = 8609

REAL_MATCH = False # whether to train with {LR-HR} or {f(HR)-HR}

# initialize similarity
ssim = SSIM().to(DEVICE)


class Masked_MSE(torch.nn.Module):

    def __init__(self, imputed_weight:float = 10e-8):
        super(Masked_MSE, self).__init__()
        self.imputed_weight = imputed_weight
        
    def forward(self, x, y, invalid_mask, datamean, datastd):

        # Compute valid mask
        valid_mask = torch.abs(invalid_mask.int() - 1)
        n_valid, n_imputed = valid_mask.sum(), invalid_mask.sum()
        
        # Perform mse over valid and imputed imputs separately
        curr_diff   = torch.subtract(x,y)
        mse_valid   = torch.div(torch.multiply(torch.square(curr_diff), valid_mask).sum(), n_valid)
        mse_imputed = torch.div(torch.multiply(torch.square(curr_diff), invalid_mask).sum(), n_imputed)\
            if n_imputed > 0 else 0

        # Concat over channel dim
        x = torch.cat(tuple([x.squeeze(0) for _ in range(3)]), 1)
        y = torch.cat(tuple([y.squeeze(0) for _ in range(3)]), 1)
        b, c, h, w = x.shape

        # De-normalize to estimate SSIM
        x = x * datastd + datamean
        y = y * datastd + datamean
        
        # Rescale to [0,1] range
        xmax = x.reshape(b, c*h*w).max(1).values.reshape(b, 1, 1, 1)
        ymax = y.reshape(b, c*h*w).max(1).values.reshape(b, 1, 1, 1)
        x = x / xmax
        y = y / ymax

        # Compute ssim over valid data
        simm_loss = 1 - ssim(torch.multiply(x, valid_mask), torch.multiply(y, valid_mask))

        return mse_valid + self.imputed_weight*mse_imputed + simm_loss


class Masked_MEA(torch.nn.Module):

    def __init__(self, imputed_weight:float = 10e-8):
        super(Masked_MSE, self).__init__()
        self.imputed_weight = imputed_weight

    def forward(self, x, y, invalid_mask, datamean, datastd):

    	# Compute valid mask
        valid_mask = torch.abs(invalid_mask.int() - 1)
        n_valid, n_imputed = valid_mask.sum(), invalid_mask.sum()

        # Perform mse over valid and imputed imputs separately
        curr_diff   = torch.subtract(x,y)
        mae_valid   = torch.div(torch.multiply(torch.abs(curr_diff), valid_mask).sum(), n_valid)
        mae_imputed = torch.div(torch.multiply(torch.abs(curr_diff), invalid_mask).sum(), n_imputed)\
            if n_imputed > 0 else 0

       # Concat over channel dim
        x = torch.cat(tuple([x.squeeze(0) for _ in range(3)]), 1)
        y = torch.cat(tuple([y.squeeze(0) for _ in range(3)]), 1)
        b, c, h, w = x.shape

        # De-normalize to estimate SSIM
        x = x * datastd + datamean 
        y = y * datastd + datamean

        # Rescale to [0,1] range
        xmax = x.reshape(b, c*h*w).max(1).values.reshape(b, 1, 1, 1)
        ymax = y.reshape(b, c*h*w).max(1).values.reshape(b, 1, 1, 1)
        x = x / xmax
        y = y / ymax

        # Compute ssim over valid data
        simm_loss = 1 - ssim(torch.multiply(x, valid_mask), torch.multiply(y, valid_mask))

        return mae_valid + self.imputed_weight*mae_imputed + simm_loss



def get_model(model_name, scale_factor=SF, nchannels=1, multi_image=False):
	if model_name == "srcnn":
		model = build_srcnn(nchannels, multi_image=False)
	elif model_name == "unet":
		model = build_unet(nchannels, multi_image=False)
	elif model_name == "resunet":
		model = build_resunet(nchannels, multi_image=False)
	elif model_name == "haunet":
		model = build_haunet(scale_factor, nchannels, multi_image=False)
	else:
		raise ValueError ("Model name not found!")
	return model.to(DEVICE)


def get_loss_function():
	loss_function = Masked_MSE()
	#loss_function = Masked_MAE()
	return loss_function


def get_optimizer(model, lr, weight_decay=0.0001):
	optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	return optim


def get_scheduler(optimizer, gamma=0.99):
	#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
		factor=0.1, patience=3, threshold=0.0001, threshold_mode='abs')
	return scheduler


def get_loaders(thresholds, batch_size):
	""" Return Train / Validation dataloader
	"""
	dls = []

	zerof_thr, cloud_thr, error_thr, impute_thr = thresholds

	for dataset in ["train", "val"]:

		toshufle = False if dataset == "train" else False 

		# Custom cloud thr for validation
		#cloud_thr = cloud_thr if dataset == "train" else 0.01

		if REAL_MATCH:
			mds = MODIS_dataset(
				lr_filepath=PATH_LR,
				hr_filepath=PATH_HR,
				qa=QA,
				zerof_thr=zerof_thr,
				cloud_thr=cloud_thr,
				error_thr=error_thr,
				impute_thr=impute_thr,
				up_scale=UP_SCALE,
				kernel_size=KERNEL_SIZE,
				train_params=PARAMS,
				data_set=dataset,
				seed=SEED)
		else:
			mds = MODIS_dataset_single(
				hr_filepath=PATH_HR,
				qa=QA,
				zerof_thr=zerof_thr,
				cloud_thr=cloud_thr,
				error_thr=error_thr,
				impute_thr=impute_thr,
				up_scale=UP_SCALE,
				kernel_size=KERNEL_SIZE,
				train_params=PARAMS,
				data_set=dataset,
				seed=SEED)

		dls.append( torch.utils.data.DataLoader(mds, 
					batch_size=batch_size, shuffle=toshufle,
					num_workers=16) )

	return dls[0], dls[1]


def display_training(cache, spec):
	""" Save final evaluation metrics plot
	"""
	cache = pd.DataFrame(cache,
			columns = ["train_loss", "train_mae", "train_bias", "train_rs", "train_pcc",
				       "val_loss", "val_mae", "val_bias", "val_rs", "val_pcc"])

	fig, axs = plt.subplots(1, 4, figsize=(15,5))
	axs = axs.flatten()
	for i in range(axs.shape[0]):
		axs[i].plot(cache[cache.columns[i]].values, label="Train")
		axs[i].plot(cache[cache.columns[i+5]].values, label="Val")
		axs[i].set_title(cache.columns[i].split("_")[-1], fontsize=12)
		axs[i].set_xlabel("Epochs")
		axs[i].grid(0.4)

	plt.suptitle(spec, fontsize=16)
	plt.legend()
	plt.tight_layout()
	plt.savefig(f"./progress/{spec}.png")


def collect_metrics(sr, hr):
	""" Perform on both Train and Val set
	Args:
		sr: super-resolved data (b, c, h, w)
		hr: ground-truth (b, c, h, w)
	"""
	#mse    = torch.square(sr-hr).mean().item()
	mae    = torch.abs(sr-hr).mean().item()
	bias   = (sr-hr).mean().item()
	ss_tot = torch.square(sr-sr.mean()).mean().item()
	ss_res = torch.square(sr-hr).mean().item()
	rs     = 1 - (ss_res/ss_tot)
	tmp = torch.zeros(2, sr.flatten().shape[0])
	tmp[0,:] = sr.flatten()
	tmp[1,:] = hr.flatten()
	pcc = torch.corrcoef(tmp)[0,1].item()
	return torch.Tensor([mae, bias, rs, pcc])



def run_n_epoch(num_epochs, run_name, model, dataloader_train,
	dataloader_val, loss_function, optimizer, scheduler,
	batch_size, checkpoint_path=None, is_fine_tune=False,
	verbose=True):
	""" Perform num_epochs training run
	"""
	cache = np.zeros((num_epochs, 10))

	if checkpoint_path:

		# Load previous state if any
		print("Loading past ckpt:")
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])

		# If not fine-tuning: load previous opt and scheduler states
		if not is_fine_tune:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			scheduler.load_state_dict(checkpoint['scheduler'])

		print("Done!")

	for e in range(1, num_epochs+1):

		cum_loss_t, cum_loss_v,  n_sample_t, n_sample_v = 0., 0., 0., 0.
		cum_train_metrics,  cum_val_metrics = torch.zeros((4)), torch.zeros((4))
		
		# Allow for model's params update
		model.train()

		# Init train progression bar and loop over train batches
		progression_bar = tqdm(dataloader_train, total=dataloader_train.__len__())
		for batch in progression_bar:
			
			lr, hr, invalid_mask, datamean, datastd = batch

			lr = lr.to(DEVICE)
			hr = hr.to(DEVICE)
			invalid_mask = invalid_mask.to(DEVICE)
			datamean = datamean.to(DEVICE)
			datastd = datastd.to(DEVICE)

			# Compute Feed-forward and current batch cost
			output = model(lr).to(DEVICE)
			loss = loss_function(output, hr, invalid_mask, datamean, datastd)

			# Update models params
			loss.backward()

			# Parameters update
			optimizer.step()

			# Gradients reset
			optimizer.zero_grad()

			# Update Loss
			cum_loss_t += loss.item()
			n_sample_t += batch[0].shape[0]

			# Collect val metrics
			cum_train_metrics += collect_metrics(output, hr)

			# Print progression
			progression_bar.set_postfix_str(f"Epoch: {e}/{num_epochs} Training Loss « {round(cum_loss_t/n_sample_t,6):.6f} »")

		# Freeze model's params:
		model.eval()

		# Init Validation progression bar and loop over Validation batches
		progression_bar = tqdm(dataloader_val, total=dataloader_val.__len__())
		with torch.no_grad():
			for batch in progression_bar:
				
				lr, hr, invalid_mask, datamean, datastd = batch
			
				lr = lr.to(DEVICE)
				hr = hr.to(DEVICE)
				invalid_mask = invalid_mask.to(DEVICE)
				datamean = datamean.to(DEVICE)
				datastd = datastd.to(DEVICE)

				# Compute Feed-forward and current batch cost
				output = model(lr).to(DEVICE)
				loss = loss_function(output, hr, invalid_mask, datamean, datastd)

				# Update Loss
				cum_loss_v += loss.item()
				n_sample_v += batch[0].shape[0]

				# Collect and update validation set metrics
				cum_val_metrics = collect_metrics(output, hr)

				# Print progression
				progression_bar.set_postfix_str(f"Epoch: {e}/{num_epochs} Validation Loss « {round(cum_loss_v/n_sample_v,6):.6f} »")

		# Collect epochs params
		train_loss = cum_loss_t/n_sample_t
		val_loss   = cum_loss_v/n_sample_v
		cum_train_metrics = [item/n_sample_t for item in cum_train_metrics.tolist()]
		cum_val_metrics   = [item/n_sample_v for item in cum_val_metrics.tolist()]

		# Save in cache
		cache[e-1, :] = [train_loss] + cum_train_metrics + [val_loss] + cum_val_metrics
		
		# Update learning rate schedule
		#scheduler.step()
		scheduler.step(train_loss)

		if verbose:
			print(f"Epoch: {e}/{num_epochs} - Train Cost: {train_loss:.7f}  Validation Cost: {val_loss:.7f}")

		# Save current epoch's checkpoint after num_epochs
		if e % 10 == 0:
			torch.save({
				'epoch': e,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'loss': val_loss,
			}, f"./checkpoint/{run_name}_ce{e}.pt")

	# Save Training performance
	pd.DataFrame(
		cache,
		columns = ["train_loss", "train_mae", "train_bias", "train_rs", "train_pcc",
   				"val_loss", "val_mae", "val_bias", "val_rs", "val_pcc"]
		)\
	.to_csv(f"./checkpoint/{run_name}_performance.csv", index=False)

	display_training(cache, run_name)

	return cache



if __name__=="__main__":

	print("!!!! check error on hr NIGHT acqusition !!!")
	# Parse arguments
	parser = argparse.ArgumentParser(description='Training SR model')
	parser.add_argument('-m' ,'--model-name', type=str, help='Model name: allowed (srcnn, unet, resunet, haunet, ...)', required=True)
	parser.add_argument('-n' ,'--run-name', type=str, help='Current Run name to save', required=True)
	parser.add_argument('-e' ,'--epochs', type=int, help='Amount of epochs to train the model', required=True)
	parser.add_argument('-b' ,'--batch-size', type=int, help='Batch-size', required=True)
	parser.add_argument('-lr','--learning-rate', type=float, help='Learning rate', required=True)
	parser.add_argument('-zt','--zerofilled-threshold', type=float, help='Zero-filled values threshold', required=True)
	parser.add_argument('-ct','--cloud-threshold', type=float, help='Cloud covereage threshold', required=True)
	parser.add_argument('-et','--error-threshold', type=float, help='Temperature error threshold', required=True)
	parser.add_argument('-it','--impute-threshold', type=float, help='Zero-filling imputing threshold', required=True)
	parser.add_argument('-c' ,'--checkpoint-path', type=str, help='Past model checkpoint', required=False)
	parser.add_argument('-f' ,'--fine-tune', action='store_true', help='Fine-tune resetting optimizer and scheduler', required=False)
	args = vars(parser.parse_args())
	print(args)

	# Initialise checkpoints and plot dir
	if not os.path.exists("./checkpoint"):
		os.system("mkdir checkpoint")

	if not os.path.exists("./progress"):
		os.system("mkdir progress")

	# Collect params
	model_name = args["model_name"]
	run_name   = args["run_name"]
	epochs     = args["epochs"]
	batch_size = args["batch_size"]
	lr         = args["learning_rate"]
	cpk_path   = args["checkpoint_path"]
	is_fine_tune = args["fine_tune"]
	thresholds = (args["zerofilled_threshold"], args["cloud_threshold"],
				 args["error_threshold"], args["impute_threshold"])

	print("Thresholds:\n", thresholds)

	# Initialise model 
	model         = get_model(model_name)
	loss_function = get_loss_function()
	optimizer     = get_optimizer(model, lr)
	scheduler     = get_scheduler(optimizer)

	# Collect MODIS dataset
	dataloader_train, dataloader_val = get_loaders(thresholds, batch_size)

	# Perfom training
	cache = run_n_epoch(
		num_epochs= epochs, 
		run_name= run_name, 
		model= model, 
		dataloader_train= dataloader_train,
		dataloader_val= dataloader_val, 
		loss_function= loss_function,
		optimizer= optimizer,
		scheduler= scheduler,
		batch_size= batch_size,
		checkpoint_path= cpk_path,
		is_fine_tune= is_fine_tune,
		verbose=True)

