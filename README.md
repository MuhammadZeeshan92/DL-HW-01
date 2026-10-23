# CS4152 --- Deep Learning and Neural Networks

## Homework 1: NumPy Neural Network Implementation

This repository contains the written solutions for Questions 1--4 and
the implementation, experiments, and report for Question 5.

## Repository Contents

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `Q1-Q4.pdf`                         Written solutions for Questions
                                      1--4

  `q5_nn_from_scratch_colab.ipynb`    Jupyter/Google Colab notebook for
                                      Question 5

  `q5_nn_from_scratch_colab.py`       Python script version of the
                                      Question 5 implementation

  `q5_report.tex`                     LaTeX source for the Question 5
                                      report

  `q5_optimizers.png`                 Optimizer comparison plot

  `q5_learning_rates.png`             Learning-rate comparison plot

  `q5_report.pdf`                     Compiled report, if included
  -----------------------------------------------------------------------

## Question 5 Overview

Question 5 implements a fully connected neural network from scratch
using **NumPy** on the **Breast Cancer Wisconsin Diagnostic** dataset.

### Model and implementation

-   Architecture: `30 → 16 → 8 → 1`
-   Hidden activation: ReLU
-   Output activation: sigmoid
-   Loss: binary cross-entropy
-   Weight initialization: He initialization
-   Backpropagation: manually implemented and vectorized
-   Regularization experiments: none, L2, and L2 with early stopping
-   Gradient verification: central finite-difference gradient check
-   Dataset split: stratified training, validation, and test sets (70% /
    15% / 15%)
-   Feature scaling: mean and standard deviation calculated from the
    training set only
-   Random seed: `42`

The neural network calculations use NumPy. Scikit-learn is used for
dataset loading and splitting.

## Experiments

1.  **Gradient check** --- compares analytical gradients with
    finite-difference estimates.
2.  **Optimizer comparison** --- SGD, momentum, RMSprop, and Adam.
3.  **Learning-rate comparison** --- SGD learning rates `0.0001`,
    `0.001`, `0.01`, `0.1`, and `1.0`.
4.  **Regularization comparison** --- no regularization, L2
    regularization, and L2 with early stopping.
5.  **Final test evaluation** --- accuracy, precision, recall, F1 score,
    and confusion matrix.

### Latest recorded results

  Metric                                      Result
  ------------------------------- ------------------
  Gradient-check relative error     `2.5747 × 10⁻¹¹`
  Trainable parameters                         `641`
  Test accuracy                             `0.9884`
  Test precision                            `1.0000`
  Test recall                               `0.9688`
  Test F1 score                             `0.9841`

Test confusion matrix, with malignant cases treated as the positive
class:

``` text
[[54, 0],
 [ 1, 31]]
```

There was one false negative in this test run. In a medical screening
context, false negatives matter because a malignant case classified as
benign could delay further assessment.

Results may vary if code, random seed, library versions, or experimental
settings change. Use the notebook or script to reproduce the results.

## How to Run

### Option 1: Google Colab

1.  Open `q5_nn_from_scratch_colab.ipynb` in Google Colab.
2.  Run the notebook cells in order.
3.  Review the printed metrics and generated plots.

### Option 2: Run the Python script locally

Requirements: Python 3, NumPy, Matplotlib, and scikit-learn.

Install dependencies:

``` bash
pip install numpy matplotlib scikit-learn
```

Run:

``` bash
python q5_nn_from_scratch_colab.py
```

## Report

-   `q5_report.tex` is the editable LaTeX source.
-   Compile it using Overleaf or a local LaTeX installation.
-   Keep `q5_optimizers.png` and `q5_learning_rates.png` in the same
    project when compiling.
-   Include the compiled `q5_report.pdf` in the repository if required
    by your instructor.

## Reproducibility Notes

-   Random seed: `42`.
-   The dataset split is stratified.
-   Standardization parameters are fitted on the training split only to
    avoid information leakage.
-   The positive class is malignant.
-   Before submission, check that the notebook, Python script, and
    report show consistent results.

## Submission Checklist

-   [ ] `Q1-Q4.pdf` included
-   [ ] Question 5 notebook included
-   [ ] Question 5 Python script included
-   [ ] LaTeX report source included
-   [ ] Both experiment plots included
-   [ ] Compiled `q5_report.pdf` included, if required
-   [ ] Notebook runs successfully from top to bottom
-   [ ] GitHub repository link added to the submission document, if
    required
-   [ ] Group member names and roll numbers added wherever required by
    the course

------------------------------------------------------------------------

**Course:** CS4152 --- Deep Learning and Neural Networks\
**Assignment:** Homework 1\
**Topic:** Neural network implementation, optimization, and
regularization
