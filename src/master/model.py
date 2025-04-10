import clip
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as SklearnRF
    
class Embedding:
    def __init__(self, config=None):
        self.config = config or {}

        self.embeding_type = None
        #Hyper parameters
        self.padding = config.get('padding', 0)

    def get_embedding_type(self):
        return self.embeding_type

    def get_masked_region(self, image, mask):
        """Helper function to extract and process masked region with padding"""
        # Convert types if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        image_array = np.array(image)
        
        # Find bounding box if mask has content
        if mask.sum() > 0:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Add padding
            y_min = max(0, y_min - self.padding)
            y_max = min(mask.shape[0], y_max + self.padding)
            x_min = max(0, x_min - self.padding)
            x_max = min(mask.shape[1], x_max + self.padding)
            
            # Check if bounding box is valid
            if x_max > x_min and y_max > y_min:
                # Crop image and mask
                if len(image_array.shape) == 3:  # RGB
                    cropped_image = image_array[y_min:y_max, x_min:x_max, :]
                    cropped_mask = mask[y_min:y_max, x_min:x_max]
                    mask_3d = np.expand_dims(cropped_mask, axis=2).repeat(3, axis=2)
                    masked_image = cropped_image * mask_3d
                else:  # Grayscale
                    cropped_image = image_array[y_min:y_max, x_min:x_max]
                    cropped_mask = mask[y_min:y_max, x_min:x_max]
                    masked_image = cropped_image * cropped_mask
                    
                # Check if mask actually removed content
                if np.sum(masked_image) < 10:
                    masked_image = cropped_image
                
                return Image.fromarray(masked_image.astype(np.uint8))
    
        # Default fallback: return original image
        return image
    
    def batch_extract_features(self, image, masks):
        """Extract features for multiple masks from the same image"""
        # Convert single mask to list for consistent processing
        if isinstance(masks, np.ndarray) and len(masks.shape) == 2:
            masks = [masks]
        
        # Extract features for each mask
        features = []
        for mask in masks:
            feature = self.extract_features(image, mask)
            features.append(feature)
        
        return features

class CLIPEmbedding(Embedding):
    def __init__(self, config=None, device=None):
        super().__init__(config)
        self.embeding_type = 'CLIP'

        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")

        # Load the CLIP model
        clip_model = config.get('clip_model', 'ViT-B/32')
    
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # We don't need to train the CLIP model
        self.model.eval()


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
        masked_image = self.get_masked_region(image, mask)
        
        # Apply CLIP's preprocessing
        input_tensor = self.preprocess(masked_image).unsqueeze(0).to(self.device)
        
        # Extract features using CLIP
        with torch.no_grad():
            features = self.model.encode_image(input_tensor)
            
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy()[0]
        

class HoGEmbedding(Embedding):
    def __init__(self, config=None):
        super().__init__(config)
        self.embeding_type = 'HOG'

        # HOG parameters
        self.hog_cell_size = self.config.get('hog_cell_size',(8,8))
        self.hog_block_size = self.config.get('hog_block_size',(2,2))

    def extract_features(self, image, mask):
        """Extract HOG features from the masked region"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert mask to numpy if it's not already
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Find bounding box of the mask
        masked_image = self.get_masked_region(image, mask)
        masked_image = np.array(masked_image)
        
        # Convert to grayscale for HOG
        if len(masked_image.shape) == 3:
            gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = masked_image
        
        # Resize to a standard size
        resized_image = cv2.resize(gray_image, (64, 64))
        
        # Extract HOG features
        win_size = (64, 64)
        block_size = (self.hog_block_size[0] * self.hog_cell_size[0],
                      self.hog_block_size[1] * self.hog_cell_size[1])
        block_stride = (self.hog_cell_size[0], self.hog_cell_size[1])
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                               self.hog_cell_size, 9)
        hog_features = hog.compute(resized_image)
        
        return hog_features.flatten()
        

class ResNET18Embedding(Embedding):
    def __init__(self, config=None, device=None):
        super().__init__(config)
        self.embedding_type = 'ResNet18'

        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")

        # Initialize ResNet18 as base model
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.eval()

        self.layers = self.config.get('layers', [2, 4, 6, 8])

        self.hook_handles = []

        # Set up hooks to capture outputs from specified layers
        for layer_idx in self.layers:
            if layer_idx < len(list(self.base_model.children())):
                layer = list(self.base_model.children())[layer_idx]
                # Store the hook handle returned by register_forward_hook
                handle = layer.register_forward_hook(self._get_hook(layer_idx))
                self.hook_handles.append(handle)

        # Transform definition remains the same
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        
        # Find bounding box of the mask
        masked_image = self.get_masked_region(image, mask)
        
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

    def cleanup(self):
        """
        Remove all hooks to prevent memory leaks.
        This method should be called when the embedding is no longer needed.
        """
        # Remove each hook handle
        for handle in self.hook_handles:
            handle.remove()
        
        # Clear the list
        self.hook_handles = []
        
        # Clear any cached feature maps
        if hasattr(self, 'feature_maps'):
            self.feature_maps = {}
        

class Classifier:
    def __init__(self, config=None, embedding=None):
        """
        Base classifier class with PCA support and feature scaling
        
        Args:
            config: Configuration dictionary
            embedding: Instance of an Embedding class
        """
        if embedding is None:
            raise ValueError("Embedding must be provided")
        elif not isinstance(embedding, Embedding):
            raise ValueError("Embedding must be an instance of Embedding")
        
        self.embedding = embedding
        self.config = config or {}

        self.embedding_type = self.embedding.get_embedding_type()
        self.classifier_type = None

        # PCA setup
        self.pca = None
        if config.get('use_PCA', False):
            self.pca_var = config.get('PCA_var', 0.95)
            # We'll initialize the actual PCA object during fit
        
        # StandardScaler setup
        self.use_scaler = config.get('use_scaler', True)
        self.scaler = StandardScaler() if self.use_scaler else None

    def get_types(self):
        return self.embedding_type, self.classifier_type
        
    def extract_features(self, image, mask):
        """Extract features using the provided embedding"""
        return self.embedding.extract_features(image, mask)
        
    def pca_fit(self, features):
        """
        Fit PCA on the features and transform them
        
        Args:
            features: Feature array to fit PCA on
            
        Returns:
            Transformed features
        """
        if self.config.get('use_PCA', False) and len(features) > 1:
            self.pca = PCA(n_components=self.pca_var, svd_solver='full')
            reduced_features = self.pca.fit_transform(features)
            print(f"Reduced feature dimension from {features[0].shape[0]} to {reduced_features.shape[1]} "
                  f"with {self.pca_var*100:.1f}% variance preserved")
            return reduced_features
        else:
            return features

    def pca_transform(self, features):
        """Transform features using pre-fit PCA"""
        if self.pca is not None:
            return self.pca.transform(features)
        else:
            return features
            
    def fit_scaler(self, features):
        """
        Fit the scaler on training features and transform them
        
        Args:
            features: Feature array to fit scaler on and transform
            
        Returns:
            Transformed features
        """
        if not self.use_scaler or self.scaler is None:
            return features
            
        return self.scaler.fit_transform(features)
            
    def transform_with_scaler(self, features):
        """
        Transform features using the fitted scaler
        
        Args:
            features: Feature array to transform
            
        Returns:
            Transformed features
        """
        if not self.use_scaler or self.scaler is None:
            return features
            
        # Check if scaler has been fit
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            print("Warning: StandardScaler has not been fit yet. Returning unscaled features.")
            return features
            
        return self.scaler.transform(features)
    
    def process_data(self, images, masks, labels):
        """
        Process images, masks, and labels to extract features
        
        Args:
            images: List of images or single image
            masks: List of masks or nested list of masks
            labels: List of labels or nested list of labels
            
        Returns:
            features: List of extracted features
            processed_labels: List of processed labels
        """
        features = []
        processed_labels = []
        
        if labels is None:
            raise ValueError("Labels must be provided")
        
        # Ensure everything is a list for consistent processing
        if not isinstance(images, list):
            images = [images]
            masks = [masks]
            labels = [labels]
        
        # Process each image and its masks
        for i, image in enumerate(images):
            # Handle case where we have single image but multiple masks
            if i >= len(masks):
                continue
                
            # Determine if this is a list of masks for a single image
            current_masks = masks[i]
            if not isinstance(current_masks, list):
                current_masks = [current_masks]
                
            # Get corresponding labels
            current_labels = labels[i] if i < len(labels) else None
            if current_labels is None:
                raise ValueError(f"No labels provided for image {i}")
                
            # Convert single label to list if needed
            if not isinstance(current_labels, list):
                current_labels = [current_labels] * len(current_masks)
            
            # Ensure labels match masks
            if len(current_labels) != len(current_masks):
                raise ValueError(f"Number of labels ({len(current_labels)}) doesn't match "
                                 f"number of masks ({len(current_masks)}) for image {i}")
                
            # Process each mask
            for j, mask in enumerate(current_masks):
                try:
                    feature = self.extract_features(image, mask)
                    features.append(feature)
                    processed_labels.append(current_labels[j])
                except Exception as e:
                    print(f"Error processing mask {j} for image {i}: {e}")
                    continue
                    
        return features, processed_labels


class KNNClassifier(Classifier):
    def __init__(self, config=None, embedding=None):
        """
        K-Nearest Neighbors classifier with embedding support
        
        Args:
            config: Configuration dictionary with hyperparameters
            embedding: Instance of an Embedding class
        """
        super().__init__(config, embedding)

        self.classifier_type = 'KNN'
        
        # Initialize KNN classifier with specified parameters
        self.n_neighbors = self.config.get('n_neighbors', 5)
        self.metric = self.config.get('metric', 'cosine')
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, 
            metric=self.metric,
            **self.config.get('knn_params', {})
        )

    def fit(self, images, masks, labels):
        """
        Train the KNN classifier with features extracted from masked regions
        
        Args:
            images: List of images or a single image
            masks: List of masks or a single mask
            labels: Labels corresponding to masks
            
        Returns:
            bool: True if training was successful
        """
        # Process data to extract features
        features, processed_labels = self.process_data(images, masks, labels)
        
        # Store for potential later use
        self.features = features
        self.labels = processed_labels
        
        if len(features) == 0:
            print("Error: No features were extracted. Cannot train KNN model.")
            return False
            
        # Apply PCA if configured
        reduced_features = self.pca_fit(features)
            
        # Train KNN with features
        self.knn.fit(reduced_features, processed_labels)
        return True

    def predict(self, image, candidate_masks, return_probabilities=True):
        """
        Predict whether candidate masks match trained classes
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            return_probabilities: Whether to return class probabilities
            
        Returns:
            filtered_masks: List of masks with their predicted classes
            probabilities: (Optional) Probability scores for each mask
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
        features = self.pca_transform(features)
        
        # Get class predictions and probabilities
        predictions = self.knn.predict(features)
        
        if return_probabilities:
            probabilities = self.knn.predict_proba(features)
            
            # Create a list of results
            results = list(zip(valid_masks, predictions))
            probs = [prob for prob in probabilities]
            
            return results, probs
        else:
            # Return masks with their predictions
            return list(zip(valid_masks, predictions))

class SVMClassifier(Classifier):
    def __init__(self, config=None, embedding=None):
        """
        Support Vector Machine classifier with embedding support
        
        Args:
            config: Configuration dictionary with hyperparameters
            embedding: Instance of an Embedding class
        """
        super().__init__(config, embedding)

        self.classifier_type = 'SVM'
        
        # Initialize SVM classifier with specified parameters
        self.C = self.config.get('C', 1.0)
        self.kernel = self.config.get('kernel', 'rbf')
        self.probability = self.config.get('probability', True)
        
        self.svm = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=self.probability,
            **self.config.get('svm_params', {})
        )

    def fit(self, images, masks, labels):
        """
        Train the SVM classifier with features extracted from masked regions
        
        Args:
            images: List of images or a single image
            masks: List of masks or a single mask
            labels: Labels corresponding to masks
            
        Returns:
            bool: True if training was successful
        """
        # Process data to extract features
        features, processed_labels = self.process_data(images, masks, labels)
        
        # Store for potential later use
        self.features = features
        self.labels = processed_labels
        
        if len(features) == 0:
            print("Error: No features were extracted. Cannot train SVM model.")
            return False
            
        # Apply PCA if configured
        reduced_features = self.pca_fit(features)
        
        # Fit scaler and transform features
        scaled_features = self.fit_scaler(reduced_features)
            
        # Train SVM with scaled features
        self.svm.fit(scaled_features, processed_labels)
        return True

    def predict(self, image, candidate_masks, return_probabilities=True):
        """
        Predict whether candidate masks match trained classes
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            return_probabilities: Whether to return class probabilities
            
        Returns:
            filtered_masks: List of masks with their predicted classes
            probabilities: (Optional) Probability scores for each mask
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
        features = self.pca_transform(features)
        
        # Transform features using fitted scaler
        scaled_features = self.transform_with_scaler(features)
        
        # Get class predictions and probabilities
        predictions = self.svm.predict(scaled_features)
        
        if return_probabilities and self.probability:
            probabilities = self.svm.predict_proba(scaled_features)
            
            # Create a list of results
            results = list(zip(valid_masks, predictions))
            probs = [prob for prob in probabilities]
            
            return results, probs
        else:
            # Return masks with their predictions
            return list(zip(valid_masks, predictions))


class RandomForestClassifier(Classifier):
    def __init__(self, config=None, embedding=None):
        """
        Random Forest classifier with embedding support
        
        Args:
            config: Configuration dictionary with hyperparameters
            embedding: Instance of an Embedding class
        """
        super().__init__(config, embedding)

        self.classifier_type = 'RF'
        
        # Initialize RF classifier with specified parameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', None)
        self.min_samples_split = self.config.get('min_samples_split', 2)
        
        self.rf = SklearnRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            **self.config.get('rf_params', {})
        )

    def fit(self, images, masks, labels):
        """
        Train the Random Forest classifier with features extracted from masked regions
        
        Args:
            images: List of images or a single image
            masks: List of masks or a single mask
            labels: Labels corresponding to masks
            
        Returns:
            bool: True if training was successful
        """
        # Process data to extract features
        features, processed_labels = self.process_data(images, masks, labels)
        
        # Store for potential later use
        self.features = features
        self.labels = processed_labels
        
        if len(features) == 0:
            print("Error: No features were extracted. Cannot train Random Forest model.")
            return False
            
        # Apply PCA if configured
        reduced_features = self.pca_fit(features)
            
        # Train Random Forest with features
        self.rf.fit(reduced_features, processed_labels)
        return True

    def predict(self, image, candidate_masks, return_probabilities=True):
        """
        Predict whether candidate masks match trained classes
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            return_probabilities: Whether to return class probabilities
            
        Returns:
            filtered_masks: List of masks with their predicted classes
            probabilities: (Optional) Probability scores for each mask
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
        features = self.pca_transform(features)
        
        # Get class predictions and probabilities
        predictions = self.rf.predict(features)
        
        if return_probabilities:
            probabilities = self.rf.predict_proba(features)
            
            # Create a list of results
            results = list(zip(valid_masks, predictions))
            probs = [prob for prob in probabilities]
            
            return results, probs
        else:
            # Return masks with their predictions
            return list(zip(valid_masks, predictions))

class LogRegClassifier(Classifier):
    def __init__(self, config=None, embedding=None):
        """
        Logistic Regression classifier with embedding support
        
        Args:
            config: Configuration dictionary with hyperparameters
            embedding: Instance of an Embedding class
        """
        super().__init__(config, embedding)

        self.classifier_type = 'LR'
        
        # Initialize Logistic Regression classifier with specified parameters
        self.C = self.config.get('C', 1.0)
        self.solver = self.config.get('solver', 'liblinear')
        self.max_iter = self.config.get('max_iter', 1000)
        self.penalty = self.config.get('penalty', 'l2')
        self.class_weight = self.config.get('class_weight', None)
        
        self.lr = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            penalty=self.penalty,
            **self.config.get('lr_params', {})
        )

    def fit(self, images, masks, labels):
        """
        Train the Logistic Regression classifier with features extracted from masked regions
        
        Args:
            images: List of images or a single image
            masks: List of masks or a single mask
            labels: Labels corresponding to masks
            
        Returns:
            bool: True if training was successful
        """
        # Process data to extract features
        features, processed_labels = self.process_data(images, masks, labels)
        
        # Store for potential later use
        self.features = features
        self.labels = processed_labels
        
        if len(features) == 0:
            print("Error: No features were extracted. Cannot train Logistic Regression model.")
            return False
            
        # Apply PCA if configured
        reduced_features = self.pca_fit(features)
            
        # Train Logistic Regression with features
        self.lr.fit(reduced_features, processed_labels)
        return True

    def predict(self, image, candidate_masks, return_probabilities=True):
        """
        Predict whether candidate masks match trained classes
        
        Args:
            image: Input image
            candidate_masks: List of candidate segmentation masks
            return_probabilities: Whether to return class probabilities
            
        Returns:
            filtered_masks: List of masks with their predicted classes
            probabilities: (Optional) Probability scores for each mask
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
        features = self.pca_transform(features)
        
        # Get class predictions and probabilities
        predictions = self.lr.predict(features)
        
        if return_probabilities:
            probabilities = self.lr.predict_proba(features)
            
            # Create a list of results
            results = list(zip(valid_masks, predictions))
            probs = [prob for prob in probabilities]
            
            return results, probs
        else:
            # Return masks with their predictions
            return list(zip(valid_masks, predictions))