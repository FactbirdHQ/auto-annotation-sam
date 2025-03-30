from src.master.evaluate import evaluate_binary_masks
from src.master.data import KFoldSegmentationManager

def load_datasets():

    return

def create_merged_dataset():
    return

def evaluate_embedding( features, labels):

    return

def train_model():
    return

def cross_dataset_metrics():
    return

def calculate_metrics( y_true, y_pred):

    return

def hyper_grid_search(model, train_loader, val_loader, grid_config):
    best_config = {}
    embedding, classifier = model.get_types()

    em_grid_config = grid_config.get(embedding, {})
    cl_grid_config = grid_config.get(classifier, {})

    if em_grid_config == {}:
        raise ValueError(f'Failed to get hyperparamter grid for {embedding}')
    
    if cl_grid_config == {}:
        raise ValueError(f'Failed to get hyperparamter grid for {classifier}')

 

    return best_config

def main():
    pass

if __name__ == "__main__":
    main()