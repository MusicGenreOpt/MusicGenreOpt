# Optimization-Based Ensemble Methods for Music Genre Classification

This repository implements mathematical models and algorithms for the music genre classification problem, as proposed in the paper "Optimization-Based Ensemble Methods for Music Genre Classification".

## Abstract

With the rise of music streaming services and the vast variety of songs available, automated tools for music classification are increasingly important. This project proposes an optimization-based ensemble approach to automatic music genre classification, leveraging heterogeneity across multiple segments of a song. Each song is divided into disjoint segments, and segment-specific genre classification probabilities are computed. The optimization algorithms then determine the optimal set of weights to combine these probabilities into an aggregated class probability for the entire song. Using a publicly available dataset, the proposed algorithms are compared to simple and complex ensemble methods. The results show that our solutions increase classification accuracy by approximately 3-6% compared to simple benchmarks and decrease accuracy by 2.33% compared to complex ensemble benchmarks. Our methods offer a good balance between model accuracy and interpretability and perform well in multi-genre classification tasks.

## Repository Structure

- **MathModel/**: Contains the mathematical modeling files for the optimization approach.
  - **Run/**:
    - `MusicProject.run`: Script to execute the optimization model using solvers like KNITRO, CPLEX, GUROBI, or BARON. 
  - **Model/**:
    - `NLP-Model.mod`: Defines the nonlinear programming model for the ensemble method.
  - **Data/**:
    - Contains datasets used in the optimization models.

- **Algo/**: Comprises the algorithmic implementations for music genre classification.
  - `Music_Genre Classification - Models.ipynb`: Jupyter notebook detailing the development and evaluation of machine learning models for genre classification.
  - **Data/**:
    - `FoldData.txt`: Lists the datasets used, possibly indicating cross-validation folds or data splits. 

## Getting Started

### Prerequisites

- AMPL (A Mathematical Programming Language)
- Python 3.x
- Jupyter Notebook
- Machine learning libraries: scikit-learn, pandas, numpy

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MusicGenreOpt/MusicGenreOpt.git
   cd MusicGenreOpt
   ```

2. **Set up the Python environment**:

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt (please check Music_Genre Classification - Models.ipynb)
   ```

   Ensure that `requirements.txt` includes all necessary libraries.

4. **AMPL Setup**:

   Ensure that AMPL is installed and accessible in your system's PATH. Visit the [AMPL website](https://ampl.com/) for installation instructions.

## Usage

### Running the Optimization Model

1. **Prepare Data**:

   Place your dataset files in the `MathModel/Data/` directory. Ensure the data format aligns with the model's requirements.

2. **Configure the Model**:

   Modify `MusicProject.run` to specify the desired solver and options. For example, to use the KNITRO solver:

   ```ampl
   option solver knitro;
   ```

3. **Execute the Model**:

   Run the AMPL script:

   ```bash
   ampl MathModel/Run/MusicProject.run
   ```

   This will solve the optimization problem and display results such as the objective value and decision variables.

### Running the Algorithmic Models

1. **Prepare Data**:

   Ensure that `FoldData.txt` in the `Algo/Data/` directory correctly references your dataset splits.

2. **Open the Jupyter Notebook**:

   Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Navigate to `Algo/Music_Genre Classification - Models.ipynb` and open it.

3. **Execute the Notebook**:

   Follow the instructions within the notebook to preprocess data, train models, and evaluate performance.

## Results

The optimization-based ensemble methods developed in this project have demonstrated improved classification accuracy compared to simple ensemble methods and competitive performance relative to complex ensembles. These methods offer a balance between accuracy and interpretability, making them valuable for practical applications in music genre classification.

## Contributing

Contributions are welcome! Please fork the repository and use a feature branch. Pull requests should be submitted against the `main` branch.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

We acknowledge the authors of the datasets and tools utilized in this project. Their contributions have been invaluable to this work.
