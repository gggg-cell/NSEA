#Mitigating Alignment Loss: Node Partitioning and Graph Diffusion for Scalable Entity Alignment

## Abstract

This project provides a unified and scalable experimental framework for large-scale Knowledge Graph Entity Alignment (EA). As the size of Knowledge Graphs (KGs) continues to grow, traditional entity alignment methods face significant computational and memory challenges. To address this issue, this framework integrates and implements a scalable EA workflow based on a graph partitioning strategy (e.g., the Metis algorithm), enabling researchers to conduct experiments on very large-scale knowledge graphs.

Furthermore, this framework brings together a variety of classic entity alignment models and provides a standardized pipeline for data processing, model training, and performance evaluation. Its modular design allows researchers to easily reproduce the results of baseline models and to quickly integrate and validate their own new models.

## Core Features

- **Modular Design**: The framework has a clear structure, divided into modules for data processing, model definition, training, and evaluation, making it easy to extend and maintain.
- **Scalable Partitioning Strategy**: Includes the built-in `Metis` graph partitioning algorithm, which can divide large-scale knowledge graphs into multiple subgraphs, significantly reducing memory and computational overhead during alignment.
- **Rich Model Zoo**: Integrates several mainstream entity alignment models (e.g., GCN-Align, RREA, Dual-A), facilitating fair performance comparisons.
- **Standardized Evaluation**: Provides a unified evaluation script that supports standard metrics such as `Hits@k` (k=1, 5, 10) and `MRR`.
- **Easy to Use**: Offers a simple command-line interface, allowing users to quickly start experiments by specifying the model and dataset.

## Project Structure

```
.
├── main.py             # Main entry point for experiments
├── framework.py        # Core framework for training and evaluation
├── dataset.py          # Dataset loading and preprocessing
├── Partition.py        # Graph partitioning module
├── Metis.py            # Metis algorithm interface
├── evaluation.py       # Evaluation metrics calculation
├── prev_models/        # Stores implementations of classic models
│   ├── gcn_align/
│   ├── rrea/
│   └── duala/
├── common/             # Common utility modules
└── ...
```

## Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [your_repository_url]
    cd [project_directory]
    ```

2.  **Create a Python virtual environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    The project may depend on the following libraries. It is recommended to create a `requirements.txt` file based on your actual `import` statements.
    ```bash
    pip install torch numpy scipy sklearn networkx
    # Other model-specific libraries may also be required
    ```
    You can generate a `requirements.txt` file with the following command:
    ```bash
    pip freeze > requirements.txt
    ```

## Quick Start

1.  **Prepare the dataset**
    Place your datasets (e.g., DBP15K, DWY100K) in a designated `data/` directory (create it if it doesn't exist). A dataset typically includes the following files:
    - `ent_links`: Aligned entity pairs for training/testing.
    - `rel_triples_1`, `rel_triples_2`: Relation triples for the two knowledge graphs.
    - `attr_triples_1`, `attr_triples_2`: Attribute triples for the two knowledge graphs (optional).

2.  **Run an experiment**
    Start an experiment via `main.py`. You can specify parameters such as the model, dataset, and language pair.
    ```bash
    # Example: Train and evaluate the RREA model on the DBP15K fr-en dataset
    python main.py --model RREA --dataset DBP15K --lang fr_en

    # Example: Use the GCN-Align model
    python main.py --model GCN_Align --dataset DWY100K --lang zh_en
    ```

## Model Zoo

This framework currently supports the following models, with all code located in the `prev_models/` directory:

- **GCN-Align**: A classic GCN-based entity alignment model.
- **RREA**: A model that introduces dual attention for relations and entities.
- **Dual-A**: An alignment model based on a dual attention network.
- **MRAEA**: A model that considers multi-view relational attention.
- ... (Please add more based on the contents of the `prev_models` directory)

## Evaluation

The evaluation script (`evaluation.py`) runs automatically after model training is complete. It calculates `Hits@1`, `Hits@5`, `Hits@10`, and `MRR` (Mean Reciprocal Rank) on the test set and outputs the results to the console and log files.

## How to Contribute

We welcome all forms of contributions! If you wish to contribute to this project, please follow these steps:

1.  Fork this repository.
2.  Create your feature branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

## Citation

If you use this framework in your research, please consider citing our work:
