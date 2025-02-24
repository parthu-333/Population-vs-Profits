# Linear Regression from Scratch - Predicting Profits

This project demonstrates **Linear Regression** using **Gradient Descent** from scratch in python. The dataset contains city population and their respective business profits.

## 🚀 Project Overview
- Load and visualize data
- Implement **Cost Function**
- Compute **Gradient**
- Perform **Gradient Descents** to optimize parameters
- Predict profits for given populations

## 📂 Dataset
The dataset (`ex1data1.txt`) consist of:
- **Population** (in 10,000s)
- **Profits** (in 10,000s dollors)

## 📊 Data Visualization
we first plot the data using `matplotlib`:
```python
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs Population")
plt.ylabel("Profits in 10,000")
plt.xlabel("Population in 10,000")
plt.show()
```
## Cost Function Implementation
The cost function for linear regression:
```python
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        total_cost += (f_wb - y[i])**2
    return total_cost / (2 * m)
```
## Gradient Descent Implementaion
We compute gradients and update weights iteratively:
```python
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw, dj_db = 0, 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    return dj_dw / m, dj_db / m
```
### Gradient Descent Function
```python
def gradient_descent(x, y, w, b, alpha, num_iters):
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {compute_cost(x, y, w, b):.2f}")
    return w, b
```
## 🏆 Results
After running **Gradient Descent** for `1500` iterations:
```
w = 1.18, b = -3.92
```
Predictions for specific populations:
```
profit1 = 3.5 * w + b  # Population: 35,000
profit2 = 7.5 * w + b  # Population: 75,000
```
Output: 
```
For population 35,000, predicted profit is : $2210.46
For population 75,000, predicted profit is : $49610.29
```
## 📌 Final Visualization
We plot the **best-fit line**:
```python
plt.plot(x_train, predicted, c="b")
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs Population per city")
plt.ylabel("Profits in 10,000")
plt.xlabel("Population in 10,000s")
plt.show()
```
## 🛠 Requirements
- Python
- NumPy
- Pandas
- Matplotlib
## How to run
1. Clone the repository:
```
git clone https://github.com/parthu-333/Population-vs-Profits.git
cd Population-vs-Profits
```
2. Install dependencies:
```
pip install numpy pandas matplotlib
```
3. Run the script:
```
python script.py
```
## 🤝 Contributing 
Give a look at my repo! Open an issue for suggestion or improvements.
