import numpy  as np
import matplotlib.pyplot as plt

def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost_sum=0

    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        cost_sum=cost_sum+cost

    total_cost=(1/(2*m))*cost_sum
    return total_cost

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w = 200
b = 100
print(f"当 w={w}, b={b} 时，成本函数值为: {compute_cost(x_train, y_train, w, b)}")

y_pred = x_train*w+b
plt.scatter(x_train, y_train,marker="x",c='r',label="Actual Values")
plt.plot(x_train,y_pred,c='b',label="Predicted Values")
plt.title("House Price Prediction")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()# 增加图例
plt.show()