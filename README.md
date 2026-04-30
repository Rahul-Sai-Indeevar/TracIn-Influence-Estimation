# TracIn: Estimating Training Data Influence by Tracing Gradient Descent

## Project Team

| Name | Roll Number |
| :--- | :--- |
| Bala Siva | 24AI10012 |
| Nikilesh | 24AI10047 |
| Srikar | 24AI10029 |
| Rahul Sai | 24AI10048 |
| Suprith | 24AI10046 |

**Institution:** Indian Institute of Technology (IIT) Kharagpur
**Department:** Artificial Intelligence
**Course:** Deep Learning

---

## Project Objective

This repository implements and reproduces the methodology proposed in the research paper *Estimating Training Data Influence by Tracing Gradient Descent*. The primary objective is to reproduce the main results and evaluation metrics reported in the paper by mathematically estimating the influence of individual training data points on specific test predictions.

---

## Repository Structure

The codebase is partitioned into two distinct execution tracks to accommodate available computational resources (such as Google Colab or Kaggle) while ensuring strict reproducibility. 

### Optimized Notebooks (Full Execution)
These notebooks are engineered for the full reproduction of the paper's quantitative results. They execute over the entire dataset domains and utilize vectorized gradient computations (`torch.func.vmap`) for maximum GPU throughput.

* `cifar10_full_optimized.ipynb`: Full training for 270 epochs with Multi-GPU support. Replicates accuracy recovery and mislabelled identification metrics.
* `dbpedia_optimized.ipynb`: Full training on 560,000 samples utilizing Hugging Face datasets. Implements checkpoint caching for crash-safe resumption.
* `implementation_gpu_mnist_full_training.ipynb`: Full dataset execution to generate precise accuracy recovery and identification curves.
* `california_housing_tracin.ipynb`: Implements TracIn on a 3-layer MLP for regression. Includes persistent disk-based checkpointing to prevent progress loss during long gradient-tracing loops.

### Demo Notebooks (Fast Execution)
These notebooks are constrained to smaller data subsets and fewer epochs. They are provided for pipeline verification and structural understanding without requiring extensive compute time.

* `cifar10_implementation.ipynb`: Evaluates a ResNet-18 model over 15 epochs on a 5,000-sample subset.
* `dbpedia_tracin_implementation.ipynb`: Evaluates a Simple Word-Embedding Model (SWEM) on a 20,000-sample subset.
* `implementation_gpu_mnist.ipynb`: Evaluates a Simple feed-forward Neural Network on a 10,000-sample subset.

---

## Datasets and Preprocessing

We utilized the datasets specified in the paper. To manage computational constraints within Kaggle environments, specific subsets were utilized in the demo notebooks, while optimized notebooks run on the full datasets. Preprocessing steps are implemented as follows:

### 1. CIFAR-10
* **Augmentation:** `RandomHorizontalFlip` and `RandomCrop` (size 32, padding 4) applied strictly to the training data.
* **Normalization:** Standardized using mean (0.4914, 0.4822, 0.4465) and standard deviation (0.2023, 0.1994, 0.2010).
* **Noise Injection:** 10% of the training labels were systematically flipped to incorrect target classes to simulate real-world mislabelling errors.

### 2. MNIST
* **Normalization:** Standardized using mean (0.1307) and standard deviation (0.3081).
* **Noise Injection:** 10% of the training labels were randomly flipped to simulate data corruption.

### 3. DBPedia 14
* **Tokenization:** Text strings were converted to lowercase and split by whitespace.
* **Vocabulary:** The dataset vocabulary was restricted to the top 40,000 most frequent words. `<pad>` and `<unk>` tokens were included for batch processing.

### 3. California Housing (Regression)
* **Scaling:** Features standardized using `StandardScaler` to ensure stable gradient descent in the MLP.
* **Architecture:** 3 hidden layers (approx. 168K parameters) optimized via Adam with Mean Squared Error (MSE) loss.

---

## Mathematical Methodology

To compute the influence scores efficiently, the implementation utilizes the first-order Taylor approximation and the Checkpoint heuristic ($TracInCP$). It approximates the idealized influence by taking the dot product of gradients at saved model checkpoints, weighted by the respective learning rates:

$$TracInCP(z, z') = \sum_{i=1}^k \eta_i \nabla\ell(w_{t_i}, z) \cdot \nabla\ell(w_{t_i}, z')$$

* $z$: Training example
* $z'$: Test example
* $\eta_i$: Learning rate at checkpoint $i$
* $\nabla\ell$: Gradient of the loss with respect to the final fully connected layer parameters.

---

## Setup and Execution Instructions

The repository is designed to be fully runnable on Kaggle with GPU support.

1. **Environment:** Upload the desired `.ipynb` file to your Kaggle environment.
2. **Hardware Acceleration:** Ensure the notebook session is configured to use a GPU (e.g., NVIDIA T4 x2).
3. **Dependencies:** For the NLP notebooks, execute the first cell to install the required libraries (`!pip install datasets tqdm`).
4. **Execution:** Run all cells sequentially. 
5. **Storage:** The optimized notebooks will automatically instantiate a `./checkpoints` directory in the working environment to persistently save model states, learning rates, and gradient scores.

---

## Experimental Results

The results generated by our optimized implementation closely align with the theoretical metrics reported in the original paper.

| Evaluation Metric | Key Findings |
| :--- | :--- |
| **Mislabelled Identification Performance** | Sorting the training data by decreasing self-influence effectively isolates the injected 10% label noise. By checking the highest-scoring fraction of the training data, the algorithm successfully identifies the corrupted examples. |
| **Test Accuracy Recovery** | Fixing the labels of the identified anomalous data points and completely retraining the model yields a distinct, monotonic recovery of test accuracy. |
| **Proponents and Opponents (NLP)** | For text classification on DBPedia, computing the dot product between training and test gradients successfully identifies training examples (proponents) that share clear semantic and structural similarities with the test prediction, and highlights examples (opponents) that contradict it. |
| **Comparables (Housing)** | In the California Housing task, TracIn identifies "comparables"—recently sold houses in similar markets that most influenced the model's price prediction. |

---

## Video Presentation

**Presentation Link:** [Insert Unlisted YouTube Link Here]