"""
Deep Neural Network Application: Cat vs. Non-Cat Classifier
----------------------------------------------------------
Standalone training and evaluation script comparing:
  1. A 2-Layer Neural Network (LINEAR -> RELU -> LINEAR -> SIGMOID)
  2. A Deep L-Layer (4-layer) Neural Network ([LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID)

Dataset: Cat vs. Non-Cat (H5 dataset)
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dnn_app_utils_v3 import (
    load_data,
    initialize_parameters,
    initialize_parameters_deep,
    linear_activation_forward,
    L_model_forward,
    compute_cost,
    linear_activation_backward,
    L_model_backward,
    update_parameters,
    predict,
    print_mislabeled_images,
)


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    """
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        # Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

        # Compute cost
        cost = compute_cost(A2, Y)

        # Backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        _, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i:4d}: {np.squeeze(cost):.6f}")
        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements an L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    """
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i:4d}: {np.squeeze(cost):.6f}")
        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs


def main():
    parser = argparse.ArgumentParser(description="Train Cat vs Non-Cat Deep Neural Network Classifier")
    parser.add_argument("--iterations", type=int, default=2500, help="Number of gradient descent iterations (default: 2500)")
    parser.add_argument("--lr", type=float, default=0.0075, help="Learning rate (default: 0.0075)")
    parser.add_argument("--save_plot", action="store_true", help="Save the cost curve comparison plot to disk")
    args = parser.parse_args()

    print("=================================================================")
    print("      DEEP NEURAL NETWORK APPLICATION - CAT VS NON-CAT CLASSIFIER")
    print("=================================================================\n")

    # 1. Load and prepare dataset
    print("[1/4] Loading and preprocessing dataset...")
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255.0
    test_x = test_x_flatten / 255.0

    print(f"  Training samples:   {train_x.shape[1]}")
    print(f"  Test samples:       {test_x.shape[1]}")
    print(f"  Feature dimension:  {train_x.shape[0]} (64x64x3 RGB flattened)")
    print(f"  Iterations:         {args.iterations}")
    print(f"  Learning Rate:      {args.lr}\n")

    # 2. Train 2-Layer Model
    print("[2/4] Training 2-Layer Neural Network (12288 -> 7 -> 1)...")
    two_layer_dims = (12288, 7, 1)
    t0 = time.time()
    params_2l, costs_2l = two_layer_model(
        train_x, train_y, layers_dims=two_layer_dims,
        num_iterations=args.iterations, learning_rate=args.lr, print_cost=True
    )
    t_2l = time.time() - t0
    print(f"  Training completed in {t_2l:.2f}s")
    print("  Evaluating 2-Layer Model:")
    print("    Train ", end="")
    p_train_2l = predict(train_x, train_y, params_2l)
    print("    Test  ", end="")
    p_test_2l = predict(test_x, test_y, params_2l)
    acc_train_2l = float(np.mean(p_train_2l == train_y)) * 100
    acc_test_2l = float(np.mean(p_test_2l == test_y)) * 100
    print()

    # 3. Train 4-Layer Model
    print("[3/4] Training 4-Layer Neural Network (12288 -> 20 -> 7 -> 5 -> 1)...")
    four_layer_dims = [12288, 20, 7, 5, 1]
    t0 = time.time()
    params_4l, costs_4l = L_layer_model(
        train_x, train_y, layers_dims=four_layer_dims,
        num_iterations=args.iterations, learning_rate=args.lr, print_cost=True
    )
    t_4l = time.time() - t0
    print(f"  Training completed in {t_4l:.2f}s")
    print("  Evaluating 4-Layer Model:")
    print("    Train ", end="")
    p_train_4l = predict(train_x, train_y, params_4l)
    print("    Test  ", end="")
    p_test_4l = predict(test_x, test_y, params_4l)
    acc_train_4l = float(np.mean(p_train_4l == train_y)) * 100
    acc_test_4l = float(np.mean(p_test_4l == test_y)) * 100
    print()

    # 4. Summary & Comparison
    print("[4/4] Final Model Comparison Summary:")
    print("+" + "-"*26 + "+" + "-"*18 + "+" + "-"*18 + "+")
    print(f"| {'Metric':<24} | {'2-Layer Model':<16} | {'4-Layer Model':<16} |")
    print("+" + "-"*26 + "+" + "-"*18 + "+" + "-"*18 + "+")
    print(f"| {'Architecture':<24} | {'[12288, 7, 1]':<16} | {'[12288, 20, 7, 5, 1]':<16} |")
    print(f"| {'Iterations':<24} | {args.iterations:<16} | {args.iterations:<16} |")
    print(f"| {'Final Cost':<24} | {costs_2l[-1]:<16.4f} | {costs_4l[-1]:<16.4f} |")
    print(f"| {'Training Accuracy':<24} | {f'{acc_train_2l:.1f}%':<16} | {f'{acc_train_4l:.1f}%':<16} |")
    print(f"| {'Test Accuracy':<24} | {f'{acc_test_2l:.1f}%':<16} | {f'{acc_test_4l:.1f}%':<16} |")
    print(f"| {'Training Time':<24} | {f'{t_2l:.1f}s':<16} | {f'{t_4l:.1f}s':<16} |")
    print("+" + "-"*26 + "+" + "-"*18 + "+" + "-"*18 + "+")

    if args.save_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(np.squeeze(costs_2l), label="2-Layer Model (7 hidden units)", color="orange")
        plt.plot(np.squeeze(costs_4l), label="4-Layer Deep Model", color="blue")
        plt.ylabel("Cost")
        plt.xlabel("Iterations (hundreds)")
        plt.title(f"Learning Curve Comparison (learning rate = {args.lr})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        out_fig = "images/cost_comparison.png"
        plt.savefig(out_fig, bbox_inches="tight", dpi=150)
        print(f"\nSaved learning curve plot to '{out_fig}'")


if __name__ == "__main__":
    main()
