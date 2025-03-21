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
            
            # Check if the bounding box is valid (has positive width and height)
            if x_max <= x_min or y_max <= y_min:
                # Invalid bounding box, use the entire image instead
                masked_image = np.array(image)
            else:
                # Crop the image to the bounding box
                cropped_image = np.array(image)[y_min:y_max, x_min:x_max]
                cropped_mask = mask[y_min:y_max, x_min:x_max]
                
                # Check cropped image and mask dimensions
                if cropped_image.size == 0 or cropped_mask.size == 0:
                    # Empty crop, use the original image
                    masked_image = np.array(image)
                else:
                    # Convert back to PIL for processing
                    cropped_image = Image.fromarray(cropped_image.astype(np.uint8))
                    
                    # Apply mask to the cropped image
                    masked_image = np.array(cropped_image)
                    
                    # Handle dimension mismatch between image and mask
                    if len(masked_image.shape) == 3:  # RGB image
                        for c in range(3):  # Apply mask to each channel
                            # Ensure mask shape matches the image channel
                            if masked_image[:, :, c].shape == cropped_mask.shape:
                                masked_image[:, :, c] = masked_image[:, :, c] * cropped_mask
                    else:  # Grayscale image
                        # Ensure mask shape matches the image
                        if masked_image.shape == cropped_mask.shape:
                            masked_image = masked_image * cropped_mask
                        else:
                            # Reshape mask if dimensions don't match
                            if cropped_mask.ndim == 2 and masked_image.ndim == 2:
                                # Resize mask to match image dimensions
                                resized_mask = np.zeros(masked_image.shape, dtype=cropped_mask.dtype)
                                # Copy the mask data that fits within the image dimensions
                                h, w = min(masked_image.shape[0], cropped_mask.shape[0]), min(masked_image.shape[1], cropped_mask.shape[1])
                                resized_mask[:h, :w] = cropped_mask[:h, :w]
                                masked_image = masked_image * resized_mask
                    
                    # Convert back to uint8 for PIL
                    masked_image = masked_image.astype(np.uint8)
            
            # Convert back to PIL Image
            masked_image = Image.fromarray(masked_image)
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
            images: List of images or a single image
            masks: List of masks or a single mask where masks[i] can be either a single mask 
                  or a list of masks for image[i]
            labels: List of labels corresponding to each mask (if None, assumes all are positive class)
                   Should match the structure of masks: if masks[i] is a list, labels[i] should be 
                   a list of same length
        """
        self.features = []
        self.labels = []
        
        # Convert single image to list for consistent processing
        if not isinstance(images, list):
            images = [images]
        
        # Convert single mask to list for consistent processing
        if not isinstance(masks, list):
            masks = [masks]
            
        # Process each image and its corresponding mask(s)
        for i in range(len(images)):
            image = images[i]
            
            # Handle case where we only have one image but multiple masks
            if len(masks) > len(images) and i == 0:
                # Assume all masks are for the first image
                image_masks = masks
                
                # Get corresponding labels if provided
                if labels is not None:
                    if isinstance(labels, list) and len(labels) == len(masks):
                        image_labels = labels
                    else:
                        # Replicate single label for all masks
                        image_labels = [1] * len(image_masks)
                else:
                    # Default all to positive class
                    image_labels = [1] * len(image_masks)
                
                # Process each mask for this image
                for j, mask in enumerate(image_masks):
                    try:
                        feature = self.extract_features(image, mask)
                        self.features.append(feature)
                        self.labels.append(image_labels[j])
                    except Exception as e:
                        print(f"Error processing mask {j}: {e}")
                        continue
            # Handle case where masks[i] is a list of masks for image[i]
            elif i < len(masks) and isinstance(masks[i], list):
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
                    try:
                        feature = self.extract_features(image, mask)
                        self.features.append(feature)
                        self.labels.append(image_labels[j])
                    except Exception as e:
                        print(f"Error processing mask {j}: {e}")
                        continue
            elif i < len(masks):
                # Single mask for this image
                mask = masks[i]
                
                # Get corresponding label
                if labels is not None:
                    label = labels[i]
                else:
                    label = 1  # Default to positive class
                
                # Extract features and store
                try:
                    feature = self.extract_features(image, mask)
                    self.features.append(feature)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error processing mask for image {i}: {e}")
                    continue
        
        # Apply PCA for dimensionality reduction
        if len(self.features) > 1:  # Need at least 2 samples for PCA
            self.pca = PCA(n_components=self.pca_variance, svd_solver='full')
            reduced_features = self.pca.fit_transform(self.features)
            print(f"Reduced feature dimension from {self.features[0].shape[0]} to {reduced_features.shape[1]} with {self.pca_variance*100:.1f}% variance preserved")
            
            # Train KNN with reduced features
            self.knn.fit(reduced_features, self.labels)
        elif len(self.features) == 1:
            # If only one sample, skip PCA
            print("Warning: Only one feature extracted. Skipping PCA and using original feature.")
            self.knn.fit(self.features, self.labels)
        else:
            print("Error: No features were extracted. Cannot train KNN model.")
            return False
        
        return True
    
    def predict(self, image, candidate_masks, return_probabilities=True):
        """
        Predict whether candidate masks are of the positive class
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            return_probabilities: Whether to return class probabilities
        
        Returns:
            filtered_masks: List of masks classified as positive
            probabilities: (Optional) Probability scores for each filtered mask
        """
        if not candidate_masks or len(candidate_masks) == 0:
            if return_probabilities:
                return [], []
            else:
                return []
                
        # Extract features for all candidate masks
        features = []
        valid_masks = []
        
        for i, mask in enumerate(candidate_masks):
            try:
                feature = self.extract_features(image, mask)
                features.append(feature)
                valid_masks.append(mask)
            except Exception as e:
                print(f"Error extracting features for mask {i}: {e}")
                continue
        
        if not features:
            if return_probabilities:
                return [], []
            else:
                return []
        
        # Apply PCA transformation if available
        if self.pca is not None:
            features = self.pca.transform(features)
        
        # Get class predictions and probabilities
        predictions = self.knn.predict(features)
        if return_probabilities:
            probabilities = self.knn.predict_proba(features)
            # Get the probability of positive class (assumes binary classification)
            positive_probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        # Filter masks based on predictions
        filtered_indices = [i for i, pred in enumerate(predictions) if pred == 1]  # Keep positive predictions
        filtered_masks = [valid_masks[i] for i in filtered_indices]
        
        if return_probabilities:
            filtered_probs = [positive_probabilities[i] for i in filtered_indices]
            return filtered_masks, filtered_probs
        else:
            return filtered_masks
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()

import clip
import torch.nn.functional as F

class CLIPFeatureSimilarity:
    def __init__(self, clip_model="ViT-B/32", similarity_threshold=0.8, device=None):
        """
        Initialize a CLIP-based feature extractor for similarity-based auto-labeling
        
        Args:
            clip_model: CLIP model variant to use ("ViT-B/32", "ViT-B/16", etc.)
            similarity_threshold: Threshold for considering a region as matching the reference
            device: Device to run inference on (will use CUDA if available by default)
        """
        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load the CLIP model
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # We don't need to train the model
        self.model.eval()
        
        # Set the similarity threshold
        self.similarity_threshold = similarity_threshold
        
        # Store reference embeddings
        self.reference_embeddings = []
        self.reference_centroid = None
        
        # Define a basic transform to use when we're extracting with masks
        # This will be used after applying the mask, then we'll apply CLIP's preprocess
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def extract_features(self, image, mask):
        """
        Extract CLIP features from a masked region of an image
        
        Args:
            image: PIL Image or numpy array
            mask: Binary mask indicating the region of interest (numpy array)
            
        Returns:
            features: Normalized feature vector from CLIP
        """
        # Convert image to PIL if it's numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Ensure mask is numpy
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Find bounding box of the mask
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
            
            # Check if the bounding box is valid
            if x_max <= x_min or y_max <= y_min:
                # Invalid bounding box, use the entire image
                image_array = np.array(image)
                # Apply mask to the image if dimensions allow
                if len(image_array.shape) == 3 and image_array.shape[:2] == mask.shape:
                    mask_3d = np.expand_dims(mask, axis=2).repeat(3, axis=2)
                    masked_image = image_array * mask_3d
                else:
                    # If dimensions don't match, just use the original image
                    masked_image = image_array
            else:
                # Crop the image and mask to the bounding box
                image_array = np.array(image)
                
                # Handle different image formats
                if len(image_array.shape) == 3:  # RGB image
                    cropped_image = image_array[y_min:y_max, x_min:x_max, :]
                    cropped_mask = mask[y_min:y_max, x_min:x_max]
                    # Expand mask to match image channels
                    mask_3d = np.expand_dims(cropped_mask, axis=2).repeat(3, axis=2)
                    masked_image = cropped_image * mask_3d
                else:  # Grayscale image
                    cropped_image = image_array[y_min:y_max, x_min:x_max]
                    cropped_mask = mask[y_min:y_max, x_min:x_max]
                    masked_image = cropped_image * cropped_mask
                
                # Check if mask actually removed content - if it's all zeros, use the cropped image
                if np.sum(masked_image) < 10:  # Arbitrary small threshold
                    masked_image = cropped_image
        else:
            # Empty mask, use original image
            masked_image = np.array(image)
        
        # Convert back to PIL Image
        masked_image = Image.fromarray(masked_image.astype(np.uint8))
        
        # Apply CLIP's preprocessing
        input_tensor = self.preprocess(masked_image).unsqueeze(0).to(self.device)
        
        # Extract features using CLIP
        with torch.no_grad():
            features = self.model.encode_image(input_tensor)
            
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy()[0]  # Return as numpy array
    
    def fit(self, images, masks):
        """
        Extract and store reference features from positive examples
        
        Args:
            images: List of images or a single image
            masks: List of masks or a single mask
            
        Returns:
            bool: Success or failure
        """
        self.reference_embeddings = []
        
        # Convert single image to list for consistent processing
        if not isinstance(images, list):
            images = [images]
        
        # Convert single mask to list for consistent processing
        if not isinstance(masks, list):
            masks = [masks]
        
        # Process each image and its corresponding mask(s)
        for i in range(len(images)):
            image = images[i]
            
            # Handle case where we have one image but multiple masks
            if len(masks) > len(images) and i == 0:
                image_masks = masks
                
                # Process each mask for this image
                for mask in image_masks:
                    try:
                        feature = self.extract_features(image, mask)
                        self.reference_embeddings.append(feature)
                    except Exception as e:
                        print(f"Error processing mask: {e}")
            # Handle case where masks[i] is a list of masks for image[i]
            elif i < len(masks) and isinstance(masks[i], list):
                image_masks = masks[i]
                
                # Process each mask for this image
                for mask in image_masks:
                    try:
                        feature = self.extract_features(image, mask)
                        self.reference_embeddings.append(feature)
                    except Exception as e:
                        print(f"Error processing mask: {e}")
            elif i < len(masks):
                # Single mask for this image
                mask = masks[i]
                
                # Extract features
                try:
                    feature = self.extract_features(image, mask)
                    self.reference_embeddings.append(feature)
                except Exception as e:
                    print(f"Error processing mask: {e}")
        
        # Compute centroid of reference embeddings (if any)
        if self.reference_embeddings:
            self.reference_embeddings = np.array(self.reference_embeddings)
            self.reference_centroid = np.mean(self.reference_embeddings, axis=0)
            self.reference_centroid = self.reference_centroid / np.linalg.norm(self.reference_centroid)
            print(f"Created reference with {len(self.reference_embeddings)} examples")
            return True
        else:
            print("No valid reference features extracted")
            return False
    
    def predict(self, image, candidate_masks, threshold=None, return_similarities=True):
        """
        Predict which candidate masks match the reference features
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            threshold: Optional override for the similarity threshold
            return_similarities: Whether to return similarity scores
            
        Returns:
            filtered_masks: List of masks classified as matching
            similarities: (Optional) Similarity scores for each filtered mask
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        if not candidate_masks or len(candidate_masks) == 0:
            if return_similarities:
                return [], []
            else:
                return []
        
        # Check if we have reference features
        if self.reference_centroid is None:
            print("No reference features available. Call fit() first.")
            if return_similarities:
                return [], []
            else:
                return []
        
        # Extract features for all candidate masks
        candidate_features = []
        valid_masks = []
        
        for mask in candidate_masks:
            try:
                feature = self.extract_features(image, mask)
                candidate_features.append(feature)
                valid_masks.append(mask)
            except Exception as e:
                print(f"Error extracting features: {e}")
        
        if not candidate_features:
            if return_similarities:
                return [], []
            else:
                return []
        
        # Compare each candidate to the reference centroid
        similarities = []
        filtered_masks = []
        
        for i, feature in enumerate(candidate_features):
            # Compute cosine similarity
            similarity = np.dot(feature, self.reference_centroid)
            
            # Apply threshold
            if similarity >= threshold:
                filtered_masks.append(valid_masks[i])
                similarities.append(float(similarity))  # Convert to Python float for JSON serialization
        
        # Sort by similarity (highest first)
        if filtered_masks:
            sorted_indices = np.argsort(similarities)[::-1]
            filtered_masks = [filtered_masks[i] for i in sorted_indices]
            similarities = [similarities[i] for i in sorted_indices]
        
        if return_similarities:
            return filtered_masks, similarities
        else:
            return filtered_masks



import clip
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class CLIPClassifier:
    def __init__(self, clip_model="ViT-B/32", device=None):
        """
        Initialize a CLIP-based classifier that uses logistic regression on CLIP embeddings
        
        Args:
            clip_model: CLIP model variant to use ("ViT-B/32", "ViT-B/16", etc.)
            device: Device to run inference on (will use CUDA if available by default)
        """
        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load the CLIP model
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # We don't need to train the CLIP model
        self.model.eval()
        
        # Define a basic transform to use when we're extracting with masks
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Initialize logistic regression classifier
        self.classifier = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Flag to track if the model has been trained
        self.is_trained = False
    
    def extract_features(self, image, mask):
        """
        Extract CLIP features from a masked region of an image
        
        Args:
            image: PIL Image or numpy array
            mask: Binary mask indicating the region of interest (numpy array)
            
        Returns:
            features: Normalized feature vector from CLIP
        """
        # Convert image to PIL if it's numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Ensure mask is numpy
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Find bounding box of the mask
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
            
            # Check if the bounding box is valid
            if x_max <= x_min or y_max <= y_min:
                # Invalid bounding box, use the entire image
                image_array = np.array(image)
                # Apply mask to the image if dimensions allow
                if len(image_array.shape) == 3 and image_array.shape[:2] == mask.shape:
                    mask_3d = np.expand_dims(mask, axis=2).repeat(3, axis=2)
                    masked_image = image_array * mask_3d
                else:
                    # If dimensions don't match, just use the original image
                    masked_image = image_array
            else:
                # Crop the image and mask to the bounding box
                image_array = np.array(image)
                
                # Handle different image formats
                if len(image_array.shape) == 3:  # RGB image
                    cropped_image = image_array[y_min:y_max, x_min:x_max, :]
                    cropped_mask = mask[y_min:y_max, x_min:x_max]
                    # Expand mask to match image channels
                    mask_3d = np.expand_dims(cropped_mask, axis=2).repeat(3, axis=2)
                    masked_image = cropped_image * mask_3d
                else:  # Grayscale image
                    cropped_image = image_array[y_min:y_max, x_min:x_max]
                    cropped_mask = mask[y_min:y_max, x_min:x_max]
                    masked_image = cropped_image * cropped_mask
                
                # Check if mask actually removed content - if it's all zeros, use the cropped image
                if np.sum(masked_image) < 10:  # Arbitrary small threshold
                    masked_image = cropped_image
        else:
            # Empty mask, use original image
            masked_image = np.array(image)
        
        # Convert back to PIL Image
        masked_image = Image.fromarray(masked_image.astype(np.uint8))
        
        # Apply CLIP's preprocessing
        input_tensor = self.preprocess(masked_image).unsqueeze(0).to(self.device)
        
        # Extract features using CLIP
        with torch.no_grad():
            features = self.model.encode_image(input_tensor)
            
        # Return as numpy array
        return features.cpu().numpy()[0]
    
    def fit(self, images, masks, labels):
        """
        Extract features from images and train logistic regression classifier
        
        Args:
            images: List of images or a single image
            masks: List where each element contains multiple masks for the corresponding image,
                  or a flat list of masks if only one image is provided
            labels: List where each element contains multiple binary labels corresponding to 
                  the masks for each image, or a flat list of labels if only one image is provided
            
        Returns:
            bool: Success or failure
        """
        # Process to collect features
        features = []
        target_labels = []
        
        # Convert single image to list for consistent processing
        if not isinstance(images, list):
            images = [images]
            
        # Handle case of a single image with many masks (flat structure)
        if len(images) == 1 and len(masks) > 1 and not isinstance(masks[0], list):
            # If we have a single image but a flat list of masks
            # We'll restructure to our expected nested format
            masks = [masks]
            labels = [labels]
            
        # Check data structure
        if len(images) != len(masks):
            print(f"Error: Number of images ({len(images)}) does not match number of mask lists ({len(masks)})")
            print("Note: If providing a single image with multiple masks, ensure masks is a list of masks and labels is a matching list of labels")
            return False
            
        if len(images) != len(labels):
            print(f"Error: Number of images ({len(images)}) does not match number of label lists ({len(labels)})")
            return False
        
        # Process each image with its corresponding masks and labels
        for i in range(len(images)):
            image = images[i]
            image_masks = masks[i]
            image_labels = labels[i]
            
            # Ensure masks is a list of masks
            if not isinstance(image_masks, list):
                image_masks = [image_masks]
                
            # Ensure labels is a list of labels
            if not isinstance(image_labels, list):
                image_labels = [image_labels]
                
            # Check if masks and labels have the same length
            if len(image_masks) != len(image_labels):
                print(f"Warning: Number of masks ({len(image_masks)}) does not match number of labels ({len(image_labels)}) for image {i}")
                # Use the shorter length
                min_len = min(len(image_masks), len(image_labels))
                image_masks = image_masks[:min_len]
                image_labels = image_labels[:min_len]
                
            # Process each mask and label for this image
            for j, (mask, label) in enumerate(zip(image_masks, image_labels)):
                try:
                    feature = self.extract_features(image, mask)
                    features.append(feature)
                    target_labels.append(label)
                except Exception as e:
                    print(f"Error processing mask {j} for image {i}: {e}")
        
        # Check if we extracted any features
        if not features:
            print("No valid features extracted")
            return False
            
        # Convert features and labels to numpy arrays
        features = np.array(features)
        target_labels = np.array(target_labels)
        
        # Normalize features with StandardScaler
        self.scaler.fit(features)
        normalized_features = self.scaler.transform(features)
        
        # Train logistic regression model
        self.classifier.fit(normalized_features, target_labels)
        self.is_trained = True
        
        print(f"Trained classifier with {len(features)} examples")
        return True
    
    def predict(self, image, candidate_masks, return_probabilities=True, return_all=False):
        """
        Predict class labels for candidate masks
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            return_probabilities: Whether to return probability scores
            return_all: If True, return all masks with their predictions and probabilities
                       If False (default), only return masks predicted as positive
            
        Returns:
            If return_all=False:
                filtered_masks: List of masks classified as positive
                probabilities: (Optional) Probability scores for positive class
            If return_all=True:
                all_masks: List of all valid masks
                predictions: List of binary predictions (0 or 1)
                probabilities: (Optional) List of probability scores for positive class
        """
        if not self.is_trained:
            print("Model not trained. Call fit() first.")
            if return_all:
                if return_probabilities:
                    return [], [], []
                else:
                    return [], []
            else:
                if return_probabilities:
                    return [], []
                else:
                    return []
            
        if not candidate_masks or len(candidate_masks) == 0:
            if return_all:
                if return_probabilities:
                    return [], [], []
                else:
                    return [], []
            else:
                if return_probabilities:
                    return [], []
                else:
                    return []
        
        # Extract features for all candidate masks
        candidate_features = []
        valid_masks = []
        
        for i, mask in enumerate(candidate_masks):
            try:
                feature = self.extract_features(image, mask)
                candidate_features.append(feature)
                valid_masks.append(mask)
            except Exception as e:
                print(f"Error extracting features for mask {i}: {e}")
        
        if not candidate_features:
            if return_all:
                if return_probabilities:
                    return [], [], []
                else:
                    return [], []
            else:
                if return_probabilities:
                    return [], []
                else:
                    return []
        
        # Normalize features
        normalized_features = self.scaler.transform(candidate_features)
        
        # Get predictions
        predictions = self.classifier.predict(normalized_features)
        
        # Get probabilities if requested
        if return_probabilities:
            probabilities = self.classifier.predict_proba(normalized_features)
            # Get probability for positive class (index 1)
            if probabilities.shape[1] > 1:
                positive_probs = probabilities[:, 1]  # Binary classification
            else:
                positive_probs = probabilities[:, 0]  # Single class
            
            # Convert to Python floats for JSON serialization
            positive_probs = [float(p) for p in positive_probs]
        
        if return_all:
            # Return all masks with their predictions
            pred_list = [int(p) for p in predictions]  # Convert to Python ints
            if return_probabilities:
                return valid_masks, pred_list, positive_probs
            else:
                return valid_masks, pred_list
        else:
            # Filter positive predictions
            filtered_masks = []
            filtered_probs = []
            
            for i, pred in enumerate(predictions):
                if pred == 1:  # Positive class
                    filtered_masks.append(valid_masks[i])
                    if return_probabilities:
                        filtered_probs.append(positive_probs[i])
            
            # Sort by probability (highest first)
            if filtered_masks and return_probabilities:
                sorted_indices = np.argsort(filtered_probs)[::-1]
                filtered_masks = [filtered_masks[i] for i in sorted_indices]
                filtered_probs = [filtered_probs[i] for i in sorted_indices]
            
            if return_probabilities:
                return filtered_masks, filtered_probs
            else:
                return filtered_masks
        

import clip
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class CLIPGradientBoostClassifier:
    def __init__(self, clip_model="ViT-B/32", device=None, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Initialize a CLIP-based classifier that uses Gradient Boosting on CLIP embeddings
        
        Args:
            clip_model: CLIP model variant to use ("ViT-B/32", "ViT-B/16", etc.)
            device: Device to run inference on (will use CUDA if available by default)
            n_estimators: Number of boosting stages to use
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of the individual regression estimators
        """
        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load the CLIP model
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # We don't need to train the CLIP model
        self.model.eval()
        
        # Define a basic transform to use when we're extracting with masks
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Initialize Gradient Boosting classifier
        self.classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,  # Use 80% of samples for building trees
            random_state=42
        )
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Flag to track if the model has been trained
        self.is_trained = False
    
    def extract_features(self, image, mask):
        """
        Extract CLIP features from a masked region of an image
        
        Args:
            image: PIL Image or numpy array
            mask: Binary mask indicating the region of interest (numpy array)
            
        Returns:
            features: Normalized feature vector from CLIP
        """
        # Convert image to PIL if it's numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Ensure mask is numpy
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Find bounding box of the mask
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
            
            # Check if the bounding box is valid
            if x_max <= x_min or y_max <= y_min:
                # Invalid bounding box, use the entire image
                image_array = np.array(image)
                # Apply mask to the image if dimensions allow
                if len(image_array.shape) == 3 and image_array.shape[:2] == mask.shape:
                    mask_3d = np.expand_dims(mask, axis=2).repeat(3, axis=2)
                    masked_image = image_array * mask_3d
                else:
                    # If dimensions don't match, just use the original image
                    masked_image = image_array
            else:
                # Crop the image and mask to the bounding box
                image_array = np.array(image)
                
                # Handle different image formats
                if len(image_array.shape) == 3:  # RGB image
                    cropped_image = image_array[y_min:y_max, x_min:x_max, :]
                    cropped_mask = mask[y_min:y_max, x_min:x_max]
                    # Expand mask to match image channels
                    mask_3d = np.expand_dims(cropped_mask, axis=2).repeat(3, axis=2)
                    masked_image = cropped_image * mask_3d
                else:  # Grayscale image
                    cropped_image = image_array[y_min:y_max, x_min:x_max]
                    cropped_mask = mask[y_min:y_max, x_min:x_max]
                    masked_image = cropped_image * cropped_mask
                
                # Check if mask actually removed content - if it's all zeros, use the cropped image
                if np.sum(masked_image) < 10:  # Arbitrary small threshold
                    masked_image = cropped_image
        else:
            # Empty mask, use original image
            masked_image = np.array(image)
        
        # Convert back to PIL Image
        masked_image = Image.fromarray(masked_image.astype(np.uint8))
        
        # Apply CLIP's preprocessing
        input_tensor = self.preprocess(masked_image).unsqueeze(0).to(self.device)
        
        # Extract features using CLIP
        with torch.no_grad():
            features = self.model.encode_image(input_tensor)
            
        # Return as numpy array
        return features.cpu().numpy()[0]
    
    def fit(self, images, masks, labels):
        """
        Extract features from images and train Gradient Boosting classifier
        
        Args:
            images: List of images or a single image
            masks: List where each element contains multiple masks for the corresponding image,
                  or a flat list of masks if only one image is provided
            labels: List where each element contains multiple binary labels corresponding to 
                  the masks for each image, or a flat list of labels if only one image is provided
            
        Returns:
            bool: Success or failure
        """
        # Process to collect features
        features = []
        target_labels = []
        
        # Convert single image to list for consistent processing
        if not isinstance(images, list):
            images = [images]
            
        # Handle case of a single image with many masks (flat structure)
        if len(images) == 1 and len(masks) > 1 and not isinstance(masks[0], list):
            # If we have a single image but a flat list of masks
            # We'll restructure to our expected nested format
            masks = [masks]
            labels = [labels]
            
        # Check data structure
        if len(images) != len(masks):
            print(f"Error: Number of images ({len(images)}) does not match number of mask lists ({len(masks)})")
            print("Note: If providing a single image with multiple masks, ensure masks is a list of masks and labels is a matching list of labels")
            return False
            
        if len(images) != len(labels):
            print(f"Error: Number of images ({len(images)}) does not match number of label lists ({len(labels)})")
            return False
        
        # Process each image with its corresponding masks and labels
        for i in range(len(images)):
            image = images[i]
            image_masks = masks[i]
            image_labels = labels[i]
            
            # Ensure masks is a list of masks
            if not isinstance(image_masks, list):
                image_masks = [image_masks]
                
            # Ensure labels is a list of labels
            if not isinstance(image_labels, list):
                image_labels = [image_labels]
                
            # Check if masks and labels have the same length
            if len(image_masks) != len(image_labels):
                print(f"Warning: Number of masks ({len(image_masks)}) does not match number of labels ({len(image_labels)}) for image {i}")
                # Use the shorter length
                min_len = min(len(image_masks), len(image_labels))
                image_masks = image_masks[:min_len]
                image_labels = image_labels[:min_len]
                
            # Process each mask and label for this image
            for j, (mask, label) in enumerate(zip(image_masks, image_labels)):
                try:
                    feature = self.extract_features(image, mask)
                    features.append(feature)
                    target_labels.append(label)
                except Exception as e:
                    print(f"Error processing mask {j} for image {i}: {e}")
        
        # Check if we extracted any features
        if not features:
            print("No valid features extracted")
            return False
            
        # Convert features and labels to numpy arrays
        features = np.array(features)
        target_labels = np.array(target_labels)
        
        # Normalize features with StandardScaler
        self.scaler.fit(features)
        normalized_features = self.scaler.transform(features)
        
        # Train Gradient Boosting model
        self.classifier.fit(normalized_features, target_labels)
        self.is_trained = True
        
        # Calculate feature importance
        feature_importances = self.classifier.feature_importances_
        top_indices = np.argsort(feature_importances)[-10:]  # Top 10 features
        print("Top 10 important feature indices:", top_indices)
        print("Their importance values:", feature_importances[top_indices])
        
        print(f"Trained classifier with {len(features)} examples")
        return True
    
    def predict(self, image, candidate_masks, return_probabilities=True, return_all=False):
        """
        Predict class labels for candidate masks
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            return_probabilities: Whether to return probability scores
            return_all: If True, return all masks with their predictions and probabilities
                       If False (default), only return masks predicted as positive
            
        Returns:
            If return_all=False:
                filtered_masks: List of masks classified as positive
                probabilities: (Optional) Probability scores for positive class
            If return_all=True:
                all_masks: List of all valid masks
                predictions: List of binary predictions (0 or 1)
                probabilities: (Optional) List of probability scores for positive class
        """
        if not self.is_trained:
            print("Model not trained. Call fit() first.")
            if return_all:
                if return_probabilities:
                    return [], [], []
                else:
                    return [], []
            else:
                if return_probabilities:
                    return [], []
                else:
                    return []
            
        if not candidate_masks or len(candidate_masks) == 0:
            if return_all:
                if return_probabilities:
                    return [], [], []
                else:
                    return [], []
            else:
                if return_probabilities:
                    return [], []
                else:
                    return []
        
        # Extract features for all candidate masks
        candidate_features = []
        valid_masks = []
        
        for i, mask in enumerate(candidate_masks):
            try:
                feature = self.extract_features(image, mask)
                candidate_features.append(feature)
                valid_masks.append(mask)
            except Exception as e:
                print(f"Error extracting features for mask {i}: {e}")
        
        if not candidate_features:
            if return_all:
                if return_probabilities:
                    return [], [], []
                else:
                    return [], []
            else:
                if return_probabilities:
                    return [], []
                else:
                    return []
        
        # Normalize features
        normalized_features = self.scaler.transform(candidate_features)
        
        # Get predictions
        predictions = self.classifier.predict(normalized_features)
        
        # Get probabilities if requested
        if return_probabilities:
            probabilities = self.classifier.predict_proba(normalized_features)
            # Get probability for positive class (index 1)
            if probabilities.shape[1] > 1:
                positive_probs = probabilities[:, 1]  # Binary classification
            else:
                positive_probs = probabilities[:, 0]  # Single class
            
            # Convert to Python floats for JSON serialization
            positive_probs = [float(p) for p in positive_probs]
        
        if return_all:
            # Return all masks with their predictions
            pred_list = [int(p) for p in predictions]  # Convert to Python ints
            if return_probabilities:
                return valid_masks, pred_list, positive_probs
            else:
                return valid_masks, pred_list
        else:
            # Filter positive predictions
            filtered_masks = []
            filtered_probs = []
            
            for i, pred in enumerate(predictions):
                if pred == 1:  # Positive class
                    filtered_masks.append(valid_masks[i])
                    if return_probabilities:
                        filtered_probs.append(positive_probs[i])
            
            # Sort by probability (highest first)
            if filtered_masks and return_probabilities:
                sorted_indices = np.argsort(filtered_probs)[::-1]
                filtered_masks = [filtered_masks[i] for i in sorted_indices]
                filtered_probs = [filtered_probs[i] for i in sorted_indices]
            
            if return_probabilities:
                return filtered_masks, filtered_probs
            else:
                return filtered_masks
    
            
    def feature_importance(self, top_n=20):
        """
        Return the most important features from the Gradient Boosting model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (index, importance) tuples for the top features
        """
        if not self.is_trained:
            print("Model not trained. Call fit() first.")
            return []
            
        # Get feature importances
        importances = self.classifier.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        # Return top N
        top_features = [(int(idx), float(importances[idx])) for idx in indices[:top_n]]
        
        return top_features