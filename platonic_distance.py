import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from carvekit.api.high import HiInterface
from scipy.signal import argrelextrema, gaussian, convolve


class PlatonicDistanceModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') # 'dinov2_vitb14')
        self.foreground_classifier = torch.nn.Linear(1536, 1)

        linear_probe_weights_path = os.path.join(Path(__file__).parent.absolute(), "../scripts/linear_probe.pt")

        self.foreground_classifier.load_state_dict(torch.load(linear_probe_weights_path, map_location=torch.device('cpu')))
        self.encoder.to(self.device)
        self.foreground_classifier.to(self.device)

        self.interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                batch_size_seg=5,
                batch_size_matting=1,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                matting_mask_size=2048,
                trimap_prob_threshold=231,
                trimap_dilation=30,
                trimap_erosion_iters=5,
                fp16=False)


    def preprocess(self, x):

        # x = self.interface([x])[0]

        width, height = x.size
        new_width = 336#(width // 14) * 14
        new_height = 336#(height // 14) * 14

        def _to_rgb(x):
            if x.mode != "RGB":
                x = x.convert("RGB")
            return x

        return torchvision.transforms.Compose([
            _to_rgb,
            torchvision.transforms.Resize((new_height, new_width), interpolation=Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(x)

    def preprocess_crop(self, x):

        def _to_rgb(x):
            if x.mode != "RGB":
                x = x.convert("RGB")
            return x

        return torchvision.transforms.Compose([
            _to_rgb,
            torchvision.transforms.Resize((800, 800), interpolation=Image.BICUBIC),
        ])(x)

    def get_foreground_mask(self, tensor_img):
        # Ensure tensor_img is detached and moved to CPU
        tensor_img = tensor_img.detach().cpu()

        # Sum across the channel dimension
        numpy_img_sum = tensor_img.sum(dim=1).squeeze(0).numpy()

        # Get the minimum value across the summed image.
        min_value = np.min(numpy_img_sum)

        # Create a 2D mask based on whether the pixel value is equal to the minimum value
        mask = ~(numpy_img_sum == min_value)

        # Convert the boolean mask to uint8 (values 0 and 1), which is acceptable for PIL
        mask = mask.astype(np.uint8)

        # Convert the mask to a PIL Image
        mask = Image.fromarray(mask * 255)  # PIL needs the image to be in the range [0, 255]

        # Downsample the mask to 57x57
        resized_mask = mask.resize((24, 24), Image.BILINEAR)  # You may choose a different resampling filter

        # Convert the resized mask back to a NumPy array
        resized_mask_numpy = np.array(resized_mask)

        # Normalize the mask to be in [0, 1] range and convert it to a float
        resized_mask_numpy = resized_mask_numpy / 255.0

        # Convert the mask to a PyTorch tensor
        tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))

        # Make all non-zero values equal to 1
        tensor_mask[tensor_mask > 0.5] = 1.0

        # Add an extra dimension for the channel
        tensor_mask = tensor_mask.unsqueeze(0).long().to(tensor_img.device)

        # If the mask is all zeros, return a tensor of ones
        if tensor_mask.sum() == 0:
            return torch.ones_like(tensor_mask)

        return tensor_mask

    def forward(self, x):

        with torch.no_grad():

            original_width, original_height = x.size

            img = np.array(self.interface([x])[0])
            img[img[..., 3] == 0] = [0, 0, 0, 0]
            img = Image.fromarray(img)
            preprocessed_img = self.preprocess(img).unsqueeze(0).to(self.device)

            mask = self.get_foreground_mask(preprocessed_img).to(self.device)

            emb = self.encoder.forward_features(preprocessed_img)
            h, w = original_height // 14, original_width // 14
            grid = emb["x_norm_patchtokens"].view(1, 24, 24, -1)


            return (grid * mask.unsqueeze(-1)).mean(dim=(1, 2)) / mask.sum()

            # Compute features for first image
            features1, masks1 = self.get_features_from_image(x)

        return features1[0]

    def forward_all_features(self, x):
        with torch.no_grad():
            # Compute features for first image
            features1, masks1 = self.get_features_from_image(x)

        features1 = torch.stack(features1)
        return features1

    def forward_compare(self, im1, im2):

        with torch.no_grad():

            # Compute features for first image
            features1, masks1 = self.get_features_from_image(im1)
            features2, masks2 = self.get_features_from_image(im2)

        # Initialize the maximum similarity to a small value
        max_similarity = -1.0

        # Compute the cosine similarity between each pair of features
        for feature1 in features1:
            for feature2 in features2:
                similarity = torch.nn.functional.cosine_similarity(feature1, feature2, dim=0)
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def forward_plot(self, im1, im2):
        with torch.no_grad():
            # Compute features and masks for the images
            features1, masks1 = self.get_features_from_image(im1)
            features2, masks2 = self.get_features_from_image(im2)

        max_similarity =  F.cosine_similarity(features1[0], features2[0], dim=0)
        max_index1, max_index2 = 0, 0

        to_tenosor = torchvision.transforms.ToTensor()
        im1 = to_tenosor(im1)
        im2 = to_tenosor(im2)

        # Upsample the masks and overlay them on the images
        mask1 = F.interpolate(masks1[max_index1].cpu().unsqueeze(0).unsqueeze(0).float(), size=im1.shape[-2:], mode='bilinear').squeeze()
        mask2 = F.interpolate(masks2[max_index2].cpu().unsqueeze(0).unsqueeze(0).float(), size=im2.shape[-2:], mode='bilinear').squeeze()

        overlay1 = im1 + (mask1.unsqueeze(0) * torch.tensor([0.5, 0.0, 0.0]).unsqueeze(1).unsqueeze(1))  # Red overlay for Image 1
        overlay2 = im2 + (mask2.unsqueeze(0) * torch.tensor([0.5, 0.0, 0.0]).unsqueeze(1).unsqueeze(1))   # Red overlay for Image 2

        # Plot the images side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(overlay1.permute(1, 2, 0))
        axs[1].imshow(overlay2.permute(1, 2, 0))
        axs[0].axis('off')
        axs[1].axis('off')

        # Set the title to the maximum similarity
        plt.suptitle(f'Maximum Similarity: {max_similarity:.2f}')
        plt.show()

        return max_similarity

    def plot_matching(self, im1, im2):

        with torch.no_grad():
            # Compute features for first image
            features1, masks1 = self.get_features_from_image(im1)
            features2, masks2 = self.get_features_from_image(im2)

        print("length of features1", len(features1), [f.shape for f in features1], "length of features2", len(features2), [f.shape for f in features2])

        # Initialize the similarity matrix
        similarity_matrix = torch.empty((len(features1), len(features2)))

        # Compute the cosine similarity between each pair of features
        for i, feature1 in enumerate(features1):
            for j, feature2 in enumerate(features2):
                similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(feature1, feature2, dim=0)

        # Create a heatmap of the similarity matrix
        plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Cosine Similarity')
        plt.xlabel('Features from Image 2')
        plt.ylabel('Features from Image 1')
        plt.title('Cosine Similarity between Image Features')
        plt.show()

    def get_features_from_image(self, x):
        original_width, original_height = x.size
        x = self.preprocess(x).unsqueeze(0).to(self.device)

        x = self.encoder.forward_features(x)
        h, w = original_height // 14, original_width // 14
        grid = x["x_norm_patchtokens"].view(1, h, w, -1)

        # Compute the foreground mask
        foreground_mask = self.foreground_classifier(grid)

        # Find the foreground object
        foreground_object = self.find_foreground_object(foreground_mask[0, :, :, 0])

        # Compute the average features for the foreground object
        blob_features = self.compute_features_for_blobs(grid[0], foreground_object)
        # blob_features = self.compute_masked_features(grid[0], foreground_object)

        return blob_features, foreground_object

    def find_largest_island(self, tensor):
        # Get the height and width of the tensor
        H, W = tensor.shape

        # Define the four possible directions to move in the grid (up, down, left, right)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Initialize the visited mask
        visited = torch.zeros((H, W), dtype=torch.bool, device=tensor.device)

        def bfs(i, j):
            # Initialize the island mask
            island_mask = torch.zeros((H, W), dtype=torch.bool, device=tensor.device)

            # Initialize the queue with the starting position
            queue = deque([(i, j)])

            while queue:
                x, y = queue.popleft()

                # Add the cell to the island mask
                island_mask[x, y] = True

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    # Check if the neighbor is inside the grid and is a land cell
                    if 0 <= nx < H and 0 <= ny < W and tensor[nx, ny] == 1 and not visited[nx, ny]:
                        # Mark the cell as visited as soon as it's added to the queue
                        visited[nx, ny] = True
                        queue.append((nx, ny))

            return island_mask

        # Initialize the list of islands
        islands = []

        for i in range(H):
            for j in range(W):
                # Check if the cell is a land cell and has not been visited yet
                if tensor[i, j] == 1 and not visited[i, j]:
                    # Mark the cell as visited as soon as it's discovered
                    visited[i, j] = True
                    islands.append(bfs(i, j))

        # Sort the islands by size
        sorted_islands = sorted(islands, key=lambda x: x.sum().item(), reverse=True)
        return sorted_islands

    def find_foreground_object(self, grid, min_percentage=0.06):

        # Find the maximum and minimum values in the grid
        max_val = grid.max().item()
        min_val = grid.min().item()

        # Compute the step size
        step = (max_val - min_val) / 100

        # Compute the minimum island size
        min_size = grid.numel() * min_percentage

        for cutoff in torch.arange(max_val, min_val, -step):

            # Apply the cutoff to create the binary mask
            binary_mask = (grid >= cutoff).float()

            # Find the largest island
            islands = self.find_largest_island(binary_mask)

            # Check if the largest island is larger than the minimum size
            if len(islands) > 0 and islands[0].sum().item() > min_size:
                # Return the original grid multiplied by the mask of the largest island
                plt.subplot(1, 2, 1)
                plt.imshow(islands[0].cpu().detach().numpy())
                plt.subplot(1, 2, 2)
                plt.hist(grid.flatten().cpu().detach().numpy(), bins='auto')
                plt.show()
                return islands
                # return torch.ones_like(grid)

        # If no suitable island was found, return an empty tensor
        return [torch.zeros_like(grid)]


    def compute_features_for_blobs(self, feature_grid, masks):
        # Initialize the list of average features
        avg_features = []

        for mask in masks:
            # Expand the dimensions of the mask to match the feature grid
            expanded_mask = mask.unsqueeze(-1)

            # Multiply the feature grid by the mask
            masked_grid = feature_grid * expanded_mask

            # Compute the average feature for the positive region
            avg_feature = masked_grid.sum(dim=(0, 1)) / mask.sum()

            # Add the average feature to the list
            avg_features.append(avg_feature)

        return avg_features

    def compute_masked_features(self, feature_grid, masks):
        # Initialize the list of masked features
        masked_features = []

        for mask in masks:
            # Expand the dimensions of the mask to match the feature grid
            expanded_mask = mask.unsqueeze(-1)

            # Multiply the feature grid by the mask
            masked_grid = feature_grid * expanded_mask

            # Add the average feature to the list
            masked_features.append(masked_grid)

        return masked_features


