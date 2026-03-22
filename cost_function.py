import numpy as np
import matplotlib.pyplot as plt
# 获取不同w的cost
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
w_range=np.linspace(0,400,100)
const_b=100

cost_value=[]
for w in w_range:
    cost_value.append(compute_cost(x_train,y_train,w,const_b))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
'''
subplots(1, 2)：创建了一个 1 行 2 列的布局。
fig：代表整张大画布。
ax1 和 ax2：代表左边和右边两个独立的“坐标系”（Axes）。
就像在 C++ 中创建了两个窗口实例，后续所有操作都要指定是在 ax1 还是 ax2 上画。
figsize=(12, 5)：设置整张图的长宽比，保证并排显示时不拥挤。'''
# --- 左图：画出你的回归线 ---
current_w=200
y_pred=x_train*current_w+const_b
ax1.scatter(x_train,y_train,marker='x',c="r",label="training data")
ax1.plot(x_train,y_pred,c='b',label="prediction ")
ax1.set_title(f"Model Fit (w={current_w})")
ax1.set_xlabel("house size")
ax1.set_ylabel("price")
ax1.legend()

# --- 右图：画出误差的“碗” ---
ax2.plot(w_range,cost_value,c='g')
current_cost=compute_cost(x_train,y_train,current_w,const_b)
ax2.scatter(current_w,current_cost,c='orange',s=100,label="now cost")
ax2.set_title("Cost Function J(w)")
ax2.set_xlabel("w")
ax2.set_ylabel("cost")
ax2.legend()

plt.tight_layout() # 让两个图排版更整齐
plt.show()