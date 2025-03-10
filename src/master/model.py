import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF

class MultiLayerFeatureKNN:
    def __init__(self, n_neighbors=5, metric='euclidean', layers=[2, 4, 6, 8], pca_variance=0.95):
        """
        Initialize the KNN model with a ResNet18 feature extractor that fuses features
        from multiple network layers
        
        Args:
            n_neighbors: Number of neighbors for KNN algorithm
            metric: Distance metric for KNN ('euclidean', 'cosine', 'manhattan', etc.)
            layers: List of ResNet layer indices to extract features from
            pca_variance: Target explained variance ratio for PCA dimensionality reduction
        """
        # Initialize ResNet18 as base model
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.eval()
        
        # Store which layers to extract features from
        self.layers = layers
        
        # Create hooks for feature extraction
        self.feature_maps = {}
        self.hooks = []
        
        # Set up hooks to capture outputs from specified layers
        for layer_idx in self.layers:
            if layer_idx < len(list(self.base_model.children())):
                layer = list(self.base_model.children())[layer_idx]
                hook = layer.register_forward_hook(self._get_hook(layer_idx))
                self.hooks.append(hook)
        
        # Initialize KNN classifier with specified metric
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        
        # Set up PCA for dimensionality reduction
        self.pca = None
        self.pca_variance = pca_variance
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # For storing features and labels during training
        self.features = []
        self.labels = []
    
    def _get_hook(self, layer_idx):
        """Create a hook function that stores output of a specific layer"""
        def hook(module, input, output):
            self.feature_maps[layer_idx] = output
        return hook
    
    def extract_features(self, image, mask):
        """
        Extract and fuse features from multiple layers of ResNet18
        for an image region defined by a mask
        
        Args:
            image: PIL Image or numpy array
            mask: Binary mask indicating the region of interest
        
        Returns:
            fused_features: Dimensionality-reduced feature vector from multiple layers
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert mask to numpy if it's not already
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Find bounding box of the mask to crop the region of interest
        if mask.sum() > 0:  # Only if mask has positive pixels
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Add padding to ensure the object is fully captured
            padding = 10
            y_min = max(0, y_min - padding)
            y_max = min(mask.shape[0], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(mask.shape[1], x_max + padding)
            
            # Crop the image to the bounding box
            cropped_image = np.array(image)[y_min:y_max, x_min:x_max]
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            
            # Convert back to PIL for processing
            cropped_image = Image.fromarray(cropped_image.astype(np.uint8))
            
            # Apply mask to the cropped image
            masked_image = np.array(cropped_image)
            if len(masked_image.shape) == 3:  # RGB image
                for c in range(3):  # Apply mask to each channel
                    masked_image[:, :, c] = masked_image[:, :, c] * cropped_mask
            else:  # Grayscale image
                masked_image = masked_image * cropped_mask
                
            masked_image = Image.fromarray(masked_image.astype(np.uint8))
        else:
            # If mask is empty, use original image (though this should be rare)
            masked_image = image
        
        # Transform the masked image for ResNet
        input_tensor = self.transform(masked_image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Clear previous feature maps
        self.feature_maps = {}
        
        # Forward pass through the network to trigger hooks
        with torch.no_grad():
            _ = self.base_model(input_batch)
        
        # Process and concatenate features from different layers
        all_features = []
        
        for layer_idx in self.layers:
            if layer_idx in self.feature_maps:
                # Get features for this layer
                layer_features = self.feature_maps[layer_idx]
                
                # Global average pooling if the features are spatial
                if len(layer_features.shape) > 2:
                    layer_features = torch.mean(layer_features, dim=(2, 3))
                
                # Flatten and convert to numpy
                flat_features = layer_features.squeeze().flatten().numpy()
                
                # L2 normalize this layer's features
                if np.linalg.norm(flat_features) > 0:
                    flat_features = flat_features / np.linalg.norm(flat_features)
                
                all_features.append(flat_features)
        
        # Concatenate all layer features
        if all_features:
            fused_features = np.concatenate(all_features)
        else:
            # Fallback if no features were extracted (should not happen)
            return np.zeros(1)
        
        return fused_features
    
    def fit(self, images, masks, labels=None):
        """
        Train the KNN classifier with features extracted from masked regions
        
        Args:
            images: List of images
            masks: List of masks where masks[i] can be either a single mask or a list of masks for image[i]
            labels: List of labels corresponding to each mask (if None, assumes all are positive class)
                   Should match the structure of masks: if masks[i] is a list, labels[i] should be a list of same length
        """
        self.features = []
        self.labels = []
        
        # Process each image and its corresponding mask(s)
        for i in range(len(images)):
            image = images[i]
            
            # Handle case where masks[i] is a single mask or a list of masks
            if isinstance(masks[i], list):
                # Multiple masks for this image
                image_masks = masks[i]
                
                # Get corresponding labels if provided
                if labels is not None:
                    if isinstance(labels[i], list):
                        image_labels = labels[i]
                    else:
                        # Replicate single label for all masks
                        image_labels = [labels[i]] * len(image_masks)
                else:
                    # Default all to positive class
                    image_labels = [1] * len(image_masks)
                
                # Process each mask for this image
                for j, mask in enumerate(image_masks):
                    feature = self.extract_features(image, mask)
                    self.features.append(feature)
                    self.labels.append(image_labels[j])
            else:
                # Single mask for this image
                mask = masks[i]
                
                # Get corresponding label
                if labels is not None:
                    label = labels[i]
                else:
                    label = 1  # Default to positive class
                
                # Extract features and store
                feature = self.extract_features(image, mask)
                self.features.append(feature)
                self.labels.append(label)
        
        # Apply PCA for dimensionality reduction
        if len(self.features) > 1:  # Need at least 2 samples for PCA
            self.pca = PCA(n_components=self.pca_variance, svd_solver='full')
            reduced_features = self.pca.fit_transform(self.features)
            print(f"Reduced feature dimension from {self.features[0].shape[0]} to {reduced_features.shape[1]} with {self.pca_variance*100:.1f}% variance preserved")
            
            # Train KNN with reduced features
            self.knn.fit(reduced_features, self.labels)
        else:
            # If only one sample, skip PCA
            self.knn.fit(self.features, self.labels)
    
    def predict(self, image, candidate_masks, threshold=None, return_scores=True):
        """
        Predict whether candidate masks are similar to the positive examples
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            threshold: Distance threshold (if None, use statistical threshold)
            return_scores: Whether to return similarity scores
        
        Returns:
            filtered_masks: List of masks that are similar to positive examples
            similarity_scores: (Optional) Similarity scores for each filtered mask
        """
        if not candidate_masks or len(candidate_masks) == 0:
            if return_scores:
                return [], []
            else:
                return []
            
        # Extract features for all candidate masks
        features = []
        for mask in candidate_masks:
            feature = self.extract_features(image, mask)
            features.append(feature)
        
        # Apply PCA transformation if available
        if self.pca is not None:
            features = self.pca.transform(features)
        
        # Get distances to nearest neighbors
        distances, indices = self.knn.kneighbors(features)
        
        # Convert distances to similarity scores (inverse distance)
        similarity_scores = 1.0 / (1.0 + np.mean(distances, axis=1))
        
        # Determine threshold for filtering
        if threshold is None:
            # Statistical threshold based on distances
            threshold = np.mean(distances) + np.std(distances)
        
        # Filter masks based on distance threshold
        filtered_indices = [i for i, d in enumerate(distances) if np.mean(d) <= threshold]
        filtered_masks = [candidate_masks[i] for i in filtered_indices]
        filtered_scores = [similarity_scores[i] for i in filtered_indices]
        
        if return_scores:
            return filtered_masks, filtered_scores
        else:
            return filtered_masks
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()

if __name__ == "__main__":

    # Initialize with specific layers and PCA variance threshold
    knn = MultiLayerFeatureKNN(layers=[5, 6, 7], pca_variance=0.95)

    # Train with just positive examples (no need for labels)
    knn.fit(positive_images, positive_masks)

    # Predict returns masks that are similar to training examples
    filtered_masks, similarity_scores = knn.predict(test_image, candidate_masks)

    # Always call cleanup when done to prevent memory leaks
    knn.cleanup()
