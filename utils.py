from typing import Tuple
import numpy as np
import os
from scipy.io import savemat
import matplotlib.pyplot as plt
import random
import torch
import matplotlib.pyplot as plt

def image_individual_norm(image):
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / std
    return image

def image_batch_norm(image_batch):
    mean = np.mean(image_batch)
    std = np.std(image_batch)
    image_batch = (image_batch - mean) / std
    return image_batch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def checkpoint(model_state, filename):
    torch.save(model_state, filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel = 1.0  # Assuming images are normalized to [0, 1] range
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

# Assuming the model's output and ground truth are both tensors of shape (1, 1, 256, 256)
def draw_comparison(predicted, ground_truth):
    # Convert tensors to numpy arrays for plotting
    predicted_np = predicted.squeeze().cpu().numpy()
    ground_truth_np = ground_truth.squeeze().cpu().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(predicted_np, cmap='gray')
    axs[0].set_title('Model Prediction')
    axs[1].imshow(ground_truth_np, cmap='gray')
    axs[1].set_title('Ground Truth')
    plt.show()

def save_mat(file, i, dir, folder_name):
    folder = os.path.join(dir, folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    savemat(os.path.join(folder , f'{i}.mat'), {'data': file})
    return None

# Assuming 'data' is your array with the shape (1050, 256, 256, 1)
# Let's select 10 random samples from the data
def draw_npz(data_list):
    selected_samples = data_list[:6]
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        # Assuming each sample in data_list is a numpy array, we use [:,:,0] to select the 2D array for imshow
        ax.imshow(selected_samples[i][:, :, 0], cmap='gray')
        ax.axis('off')
    plt.show()

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def tensor_display(tensor_data, ifsave, save_path, name, vmin=None, vmax=None, fontsize=14):
    fig, axes = plt.subplots(1, tensor_data.size(0), figsize=(20, 20))  # Adjust the subplot layout dynamically based on A's size
    if tensor_data.size(0) == 1:  # If there's only one image
        axes = [axes]  # Wrap it in a list to make iterable
    for i, img in enumerate(tensor_data):
        img = img.squeeze()  # Remove unnecessary dimensions for display
        im = axes[i].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)  # Display the image in grayscale
        axes[i].axis('off')  # Turn off the axis
        cbar = fig.colorbar(im, ax=axes[i])  # Add a colorbar for each image with padding
        cbar.remove()  # Set font size of colorbar labels
        
    plt.subplots_adjust(wspace=0, hspace=0)
    if ifsave == 1:
        filepath = os.path.join(save_path, f'{name}.png')
        plt.savefig(filepath)
        #print(f"Image saved as {filepath}")
        plt.close(fig)
    #plt.show()

def img_display(tensor_data, ifsave, save_path, name, vmin=None, vmax=None, fontsize=14):
    fig, axes = plt.subplots(1, tensor_data, figsize=(20, 20))  # Adjust the subplot layout dynamically based on A's size
    if tensor_data.size(0) == 1:  # If there's only one image
        axes = [axes]  # Wrap it in a list to make iterable
    for i, img in enumerate(tensor_data):
        img = img.squeeze()  # Remove unnecessary dimensions for display
        im = axes[i].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)  # Display the image in grayscale
        axes[i].axis('off')  # Turn off the axis
        cbar = fig.colorbar(im, ax=axes[i])  # Add a colorbar for each image with padding
        cbar.remove()  # Set font size of colorbar labels
        
    plt.subplots_adjust(wspace=0, hspace=0)
    if ifsave == 1:
        filepath = os.path.join(save_path, f'{name}.png')
        plt.savefig(filepath)
        print(f"Image saved as {filepath}")
        plt.close(fig)

def displaydataset(train_dataset, num_images=10, ifimage=1):
   # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])
    if ifimage==1:
        first_10_images = train_dataset.images[:10]  # This slices the first 10 images
    elif ifimage==0:
        first_10_images = train_dataset.labels[:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # Setting up the plot for 10 images in a 2x5 grid
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration
    for i, img in enumerate(first_10_images):
        # Since the images are stored with a channel dimension, we use squeeze to remove it for display
        img_display = axes[i].imshow(img.squeeze(), cmap='gray')  # Assuming the images are grayscale
        axes[i].axis('off')  # Turn off axis to make it look cleaner
        axes[i].set_title(f'Image {i+1}')  # Optional: Set title for each subplot
        fig.colorbar(img_display, ax=axes[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()    
    
def colorbar_plot(vmin, vmax):
    fig, ax = plt.subplots()

    # Create dummy data
    data = np.random.rand(10, 10)

    # Plot the data with a colormap
    im = ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)

    # Add colorbar to the figure
    cbar = fig.colorbar(im, ax=ax)

    # Show the plot
    plt.show()
    