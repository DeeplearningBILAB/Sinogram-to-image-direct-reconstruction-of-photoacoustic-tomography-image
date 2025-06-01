import os
import gc
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage import img_as_float
from scipy.io import savemat, loadmat
import numpy as np

from CustomData import create_dataloader
from Network_1 import FC_SWIN
from Loss_function import Combined_loss, Combined_SSIM_MAE
from utils import get_lr, checkpoint
from Loss_function import Mae, ssim
from utils import tensor_display, psnr

from test_model import test_model

# ======================= Global Parameters =======================
torch.cuda.empty_cache()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device_idx = 0
device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
num_workers = 0
LR_change = 1
# ======================= Hyperparameters =======================
num_epochs = 150
num_batch = 2

# Learning rate and loss weighting
lr = 0.0001
alpha = 0.5
# Network parameters
input_dim = 256 * 1024
hidden_dim = 8 * 256
output_dim = 256 * 256
img_size = (256, 256)
activation = 'relu'
# training loop parameter
min_val_loss = float('inf')
best_model_state = None
best_epoch = 0
patience = 5 # number of epochs to wait before early stopping
no_improvement_count = 0
# ======================= Define Loss Function =======================
criterion = Combined_loss(alpha=alpha) 
# ======================= Paths =======================
base_dir ='G:\\PhD\\Third_year\\Sinogram_recon\\2Simulation_data\\20250407Data\\noisy_npz'

dataset_files = {
    "train": "Training.npz",
    "val": "Validation.npz",
    "test": "Test.npz",
    "Exptest": "Exptest.npz"
}
model_dir = "G:\\PhD\\Third_year\\Sinogram_recon\\hyper\\test_5_py"
epoch_dir = os.path.join(model_dir, "Epoch_in_progress")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(epoch_dir, exist_ok=True)
model_name = "best_metric_model_test.pth"
im_save_dir_prediction = os.path.join(model_dir, 'prediction')
im_save_dir_GT = os.path.join(model_dir, 'groundtruth')
im_save_fc = os.path.join(model_dir, 'fc_output')
im_save_act1 = os.path.join(model_dir, 'act1')
im_save_act2 = os.path.join(model_dir, 'act2')

unseenpath = "G:\\PhD\\Third_year\\Sinogram_recon\\2Simulation_data\\Unseen_data\\New_v1\\npz"
noisy_recon_path = r'G:\PhD\Third_year\Sinogram_recon\2Simulation_data\20250407Data\mat\noisy_recon'
noisy_test_path = r'G:\PhD\Third_year\Sinogram_recon\hyper\test_5_py\recon'
os.makedirs(noisy_test_path, exist_ok=True)
# ======================= Data Loading =======================
train_loader = create_dataloader(base_dir, dataset_files, 'train', num_batch, num_workers)
val_loader = create_dataloader(base_dir, dataset_files, 'val', num_batch, num_workers)
test_loader = create_dataloader(base_dir, dataset_files, 'test', 1, num_workers)
# ======================= Model Setup =======================
model = FC_SWIN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, img_size=img_size, activation=activation).to(device)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience=patience//2)

# ======================= Training Loop =======================
train_losses = []
val_losses = []
Learning_rate = []
for epoch in tqdm(range(num_epochs),desc='Epochs'):
    epoch_iterator = tqdm(enumerate(train_loader), desc="Training Batches", total=len(train_loader))
    total_loss = 0.0
    model.train() # set the model to the training mode
    for step, batch in epoch_iterator:
        x_img = batch["image"].type(torch.FloatTensor)
        y_img = batch["label"].type(torch.FloatTensor)
        x, y = (x_img.to(device), y_img.to(device))
        
        predicted_x, predicted_fc, _,_ = model(x)

        loss = criterion(predicted_x, y) 
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 ==0:
            epoch_iterator.set_description(f"Epoch {epoch+1}, Step {step+1}/{len(train_loader)}")
    print(f"Total loss for epoch {epoch+1}: {total_loss}")
    train_losses.append(total_loss / len(train_loader)) # average loss on each batch
    
    checkpoint(model.state_dict(), os.path.join(epoch_dir, f"epoch-{epoch}.pth"))
    
    val_loss = 0.0   
    with torch.no_grad():
        model.eval() # set the model to evaluation mode
        for batch in tqdm(val_loader, desc="Validation Batcher", total=len(val_loader)):
            val_inp = batch["image"].type(torch.FloatTensor)
            val_lbl = batch["label"].type(torch.FloatTensor)
            val_inputs, val_labels = (val_inp.to(device), val_lbl.to(device))
            predicted_vx, predicted_vfc, _,_ = model(val_inputs)
            val_loss += criterion(predicted_vx, val_labels).item() 
            
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Average validation Loss for Epoch {epoch+1}: {avg_val_loss}")  
    
    if LR_change==1:
        scheduler.step(avg_val_loss)    
    current_lr = get_lr(optimizer)
    Learning_rate.append(current_lr)
    if avg_val_loss<min_val_loss:
        min_val_loss = avg_val_loss
        no_improvement_count = 0
        best_model_state = model.state_dict()# save the model if validation loss is minimum    
        best_epoch = epoch+1
    else:
        no_improvement_count+=1
        if no_improvement_count>=patience:
            print("Early stopping triggered. Training stopped at epoch {epoch+1}.")
            break

if best_model_state is not None:
    checkpoint(best_model_state, os.path.join(model_dir, model_name))        
    print(f"Best model saved at epoch {best_epoch}")

plt.figure()
plt.plot(range(1, epoch+2), train_losses, label='Training Loss')
plt.plot(range(1, epoch+2), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses vs Epoch')
plt.legend()
plt.grid(True)
plt.show()    

plt.figure()
plt.plot(range(1, epoch+2), Learning_rate, label='Learning_rate')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning_rate')
plt.legend()
plt.grid(True)
plt.show()  
# ======================= Test Loop =======================
test_model(
    test_loader, 
    model,  
    os.path.join(model_dir, model_name), 
    im_save_dir_prediction, 
    im_save_dir_GT, 
    im_save_fc,
    activation=activation,
    activation_display = None, 
    im_save_act1 = im_save_act1, 
    im_save_act2 = im_save_act2,
)

# ======================= Exp Test Loop =======================
exptest_loader = create_dataloader(base_dir, dataset_files, 'Exptest', 1, num_workers)
test_model(
    exptest_loader, 
    model,  
    os.path.join(model_dir, model_name), 
    os.path.join(model_dir, 'EXPPRE_output'), 
    os.path.join(model_dir, 'EXPGT_output'), 
    os.path.join(model_dir, 'EXPfc_output'),
    activation=activation,
)

# ======================= Unseen Test Loop =======================
unseentest_loader = create_dataloader(unseenpath, dataset_files, 'Exptest', 1, num_workers)
test_model(
    unseentest_loader, 
    model,  
    os.path.join(model_dir, model_name), 
    os.path.join(model_dir, 'unseenPRE_output'), 
    os.path.join(model_dir, 'unseenGT_output'), 
    os.path.join(model_dir, 'unseenfc_output'),
    activation=activation,
)

# ======================= noisy recon showing =======================
test_path = os.path.join(base_dir, 'Test_index.mat')
test_idx = loadmat(test_path)['test_array'].flatten()
# Loop over test indices
for step_id, idx in enumerate(test_idx):
    if step_id<80:
        # Load .mat file
        matname = os.path.join(noisy_recon_path, f'{idx}.mat')
        noisy_recon = loadmat(matname)['reconstructed_image']
        
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))

        vmin, vmax = np.min(noisy_recon), np.max(noisy_recon)
        im = axes.imshow(noisy_recon, cmap='gray', vmin=vmin, vmax=vmax)
        axes.axis('off')
        cbar = fig.colorbar(im, ax=axes)
        cbar.remove()  # optional, can be deleted if you want to keep colorbar

        filepath = os.path.join(noisy_test_path, f'{step_id}.png')
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        print(f"Image saved as {filepath}")
        plt.close(fig)

    