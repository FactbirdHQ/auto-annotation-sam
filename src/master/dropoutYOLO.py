from ultralytics import YOLO
import torch.nn.functional as F

class MCDropoutTracker:
    def __init__(self, model_path, dropout_rate=0.05):
        self.model_path = model_path
        self.dropout_rate = dropout_rate
        # Initialize model for immediate use if needed
        self.model = self._create_fresh_model()
        
    def _create_fresh_model(self):
        """Create a fresh model instance with dropout applied"""
        # Load a new model instance
        model = YOLO(self.model_path)
        
        # Find detection head
        detect_head = None
        for module in model.model.modules():
            if 'Detect' in str(type(module)):
                detect_head = module
                break
                
        if detect_head is None:
            raise ValueError("Could not find detection head")
            
        # Store original forward method
        original_forward = detect_head.forward
        
        # Define new forward with dropout
        def forward_with_dropout(x, *args, **kwargs):
            # Apply dropout to feature maps
            if isinstance(x, list):
                x = [F.dropout(xi, p=self.dropout_rate, training=True) for xi in x]
            else:
                x = F.dropout(x, p=self.dropout_rate, training=True)
            
            # Call original forward
            return original_forward(x, *args, **kwargs)
            
        # Replace forward method
        detect_head.forward = forward_with_dropout
        
        # Set to train mode to enable dropout
        model.model.train()
        
        return model
    
    def fine_tune(self, data_yaml, epochs=5, batch_size=16, reset_model=False, **kwargs):
        """Fine-tune the model with dropout enabled
        
        Args:
            data_yaml: Path to data YAML file
            epochs: Number of training epochs
            batch_size: Batch size for training
            reset_model: Whether to create a fresh model before fine-tuning
            **kwargs: Additional arguments for YOLO training
        """
        # Create a fresh model if requested or if model doesn't exist
        if reset_model or not hasattr(self, 'model'):
            self.model = self._create_fresh_model()
        
        # Make sure dropout is active during training
        self.model.model.train()
        
        # Start fine-tuning
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            **kwargs
        )
        
        return results
    
    def track(self, source, fresh_model=True, **kwargs):
        """Run tracking with MC dropout enabled
        
        Args:
            source: Source for tracking (video, image, etc.)
            fresh_model: Whether to create a fresh model for this tracking session
            **kwargs: Additional arguments for YOLO tracking
        """
        # Create a fresh model if requested
        if fresh_model:
            tracking_model = self._create_fresh_model()
        else:
            # Use existing model (ensure it's in train mode for dropout)
            if not hasattr(self, 'model'):
                self.model = self._create_fresh_model()
            tracking_model = self.model
            tracking_model.model.train()
        
        # Run tracking
        return tracking_model.track(source, **kwargs)