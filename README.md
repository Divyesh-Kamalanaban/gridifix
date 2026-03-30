# GRIDIFIX - Single Bus Fault Detection and Localization Engine
This is my first ever ML project, where I had to learn about CIGRE-MV Networks, Pandapower implementation of it, DLPF Solver, DNN implementation of Non-Linear Newton Raphson equation, 2 Ygdrassil Random Forest Classifiers and a comparator component to compare between DLPF and DNN.
## Basic ML fundamentals that got me going.
### Underfitting vs Overfitting
- Underfitting means the model is too simple for the complex dataset, hence performing poor on both train and test datasets!
- Overfitting means the model is too flexible for the simple dataset, which leads it to learn the pattern of noises instead of actual data and memorising it.
- Underfitting => Model Capacity < Data Complexity
- Overfitting => Model Capacity>> Data Complexity (Noise Included)
### Bias Variance Tradeoff
- Bias is the error that occurs due to simplification of the function
- Variance is the error that stems from the high sensitivity of model to noise and other changes.

### Model Summary: Sequential Neural Network

| Layer Name        | Type                | Output Shape | Parameters |
|------------------|---------------------|--------------|------------|
| dense_hidden_1   | Dense               | (None, 512)  | 33,280     |
| bn_1             | BatchNormalization  | (None, 512)  | 2,048      |
| relu_1           | Activation (ReLU)   | (None, 512)  | 0          |
| dense_hidden_2   | Dense               | (None, 256)  | 131,328    |
| bn_2             | BatchNormalization  | (None, 256)  | 1,024      |
| relu_2           | Activation (ReLU)   | (None, 256)  | 0          |
| dense_hidden_3   | Dense               | (None, 128)  | 32,896     |
| bn_3             | BatchNormalization  | (None, 128)  | 512        |
| relu_3           | Activation (ReLU)   | (None, 128)  | 0          |
| output_layer     | Dense               | (None, 88)   | 11,352     |

---

### Summary

- **Total Parameters:** 212,440 (~829.84 KB)  
- **Trainable Parameters:** 210,648 (~822.84 KB)  
- **Non-trainable Parameters:** 1,792 (~7.00 KB)

---

### Architecture Overview

- 3 fully connected hidden layers (512 → 256 → 128)
- Each hidden layer followed by:
  - Batch Normalization
  - ReLU activation
- Final output layer:
  - 88 neurons (Multi-Class Classification)

---

### Notes

- Batch Normalization improves training stability and convergence.
- ReLU introduces non-linearity.
- Model size is moderate (~212K params).
