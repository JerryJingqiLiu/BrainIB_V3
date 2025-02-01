# Brain Network Analysis with Bootstrap Training

This project is a brain network analysis based on the Bootstrap sampling method. It is an extended version of the original BrainIB project, with the following major improvements:

## Key Features
1. Dataset Division:
   - Initial split of 70% training set / 30% inference set
   - Random bootstrap sampling method applied to the training set

2. Model Training:
   - Train 10 independent models using the Bootstrap method
   - Each model is trained with 70% random samples from the training set
   - The remaining 30% samples are used for validation

3. Model Evaluation:
   - All models are tested on the 30% inference set
   - Provides detailed performance statistics and uncertainty estimation

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python main_bootstrap.py --save_dir /path/to/save/dir --num_bootstrap_models 10
```

3. Main Parameters:
- `--save_dir`: Path to save models and logs
- `--num_bootstrap_models`: Number of Bootstrap models (default: 10)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--seed`: Random seed (default: 0)

## Project Structure
```
bootstrap_version/
├── data/
│   └── create_dataset.py    # Dataset creation and loading
├── src/
│   ├── GNN.py              # Graph Neural Network model
│   ├── sub_graph_generator.py   # Subgraph generator
│   ├── utils.py            # Utility functions
│   ├── mutual_information.py    # Mutual information calculation
│   └── bootstrap_trainer.py     # Bootstrap training implementation
└── main_bootstrap.py       # Main program entry
``` 