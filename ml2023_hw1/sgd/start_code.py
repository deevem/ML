import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def feature_normalization(train, test):
    """将训练集中的所有特征值映射至[0,1]，对验证集上的每个特征也需要使用相同的仿射变换

    Args：
        train - 训练集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        test - 测试集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
    Return：
        train_normalized - 归一化后的训练集
        test_normalized - 标准化后的测试集

    """
    # Hint: 计算训练集的最大值和最小值，然后使用仿射变换将其映射至[0,1]
    
    mins = np.min(train, axis=0)
    maxs = np.max(train, axis=0)
    train_normalized = (train - mins) / (maxs - mins)
    test_normalized = (test - mins) / (maxs - mins)
    
    return train_normalized, test_normalized



def compute_regularized_square_loss(X, y, theta, lambda_reg):
    """
    给定一组 X, y, theta，计算用 X*theta 预测 y 的岭回归损失函数

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小 (num_features)
        lambda_reg - 正则化系数

    Return：
        loss - 损失函数，标量
    """
    # TODO 2.2.2
    
    return (X @ theta - y).T @ (X @ theta - y) / X.shape[0] + lambda_reg * (theta.T @ theta)


def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    计算岭回归损失函数的梯度

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数

    返回：
        grad - 梯度向量，数组大小（num_features）
    """
    # TODO 2.2.4
    
    return 2 * X.T @ (X @ theta - y) / X.shape[0] + 2 * lambda_reg * theta


def grad_checker(X, y, theta, lambda_reg, epsilon=0.01, tolerance=1e-4):
    """梯度检查
    如果实际梯度和近似梯度的欧几里得距离超过容差，则梯度计算不正确。

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数
        epsilon - 步长
        tolerance - 容差

    Return：
        梯度是否正确

    """
    true_gradient = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)  # the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate

    # TODO 2.2.5 (optional)
    for i in range(num_features):
        theta_large = theta.copy()
        theta_large[i] += epsilon
        theta_small = theta.copy()
        theta_small[i] -= epsilon
        approx_grad[i] = (compute_regularized_square_loss(X, y, theta_large, lambda_reg) - compute_regularized_square_loss(X, y, theta_small, lambda_reg)) / (2 * epsilon)
    
    return np.linalg.norm(approx_grad - true_gradient) < tolerance
    
    
    


def batch_grad_descent(X, y, lambda_reg, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    全批量梯度下降算法

    Args：
        X - 特征向量， 数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        alpha - 梯度下降的步长，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        check_gradient - 更新时是否检查梯度

    Return：
        theta_hist - 存储迭代中参数向量的历史，大小为 (num_iter+1, num_features) 的二维 numpy 数组
        loss_hist - 全批量损失函数的历史，大小为 (num_iter) 的一维 numpy 数组
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 2.3.3
    
    for i in range(num_iter):
        if check_gradient:
            if not grad_checker(X, y, theta, lambda_reg):
                print("Gradient is wrong!")
                break
        theta_hist[i + 1] = theta = theta - alpha * compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        loss_hist[i] = compute_regularized_square_loss(X, y, theta, lambda_reg)

    return theta_hist, loss_hist
    
    


def compute_current_alpha(alpha, iter):
    """
    梯度下降步长策略，可自行扩展支持更多策略

    参数：
        alpha - 字符串或浮点数。梯度下降步长
                注意：在 SGD 中，使用固定步长并不总是一个好主意。通常设置为 1/sqrt(t) 或 1/t
                如果 alpha 是一个浮点数，那么每次迭代的步长都是 alpha。
                如果 alpha == "0.05/sqrt(t)", alpha = 0.05/sqrt(t)
                如果 alpha == "0.05/t", alpha = 0.05/t
        iter - 当前迭代次数（初始为1）

    返回：
        current_alpha - 当前采取的步长
    """
    assert isinstance(alpha, float) or (isinstance(alpha, str) and (alpha == '0.05/sqrt(t)' or alpha == '0.05/t'))
    if isinstance(alpha, float):
        current_alpha = alpha
    elif alpha == '0.05/sqrt(t)':
        current_alpha = 0.05 / np.sqrt(iter)
    elif alpha == '0.05/t':
        current_alpha = 0.05 / iter
    return current_alpha


def stochastic_grad_descent(X_train, y_train, X_test, y_test, lambda_reg, alpha=0.1, num_iter=1000, batch_size=1):
    """
    随机梯度下降，并随着训练过程在验证集上验证

    参数：
        X_train - 训练集特征向量，数组大小 (num_instances, num_features)
        y_train - 训练集标签向量，数组大小 (num_instances)
        X_test - 验证集特征向量，数组大小 (num_instances, num_features)
        y_test - 验证集标签向量，数组大小 (num_instances)
                 注意：在 SGD 中，小批量的训练损失函数噪声较大，难以清晰反应模型收敛情况，可以通过验证集上的全批量损失来判断
        alpha - 字符串或浮点数。梯度下降步长，可自行调整为默认值以外的值
                注意：在 SGD 中，使用固定步长并不总是一个好主意。通常设置为 alpha_0/sqrt(t) 或 alpha_0/t
                如果 alpha 是一个浮点数，那么每次迭代的步长都是 alpha。
                如果 alpha == "0.05/sqrt(t)", alpha = 0.05/sqrt(t)
                如果 alpha == "0.05/t", alpha = 0.05/t
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量正则化损失函数的历史，数组大小(num_iter)
        validation hist - 验证集上全批量均方误差（不带正则化项）的历史，数组大小(num_iter)
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    validation_hist = np.zeros(num_iter)  # Initialize validation_hist

    # TODO 2.4.3
    
    for i in range(num_iter):
        idx = np.random.randint(0, num_instances, batch_size)
        # print(idx)
        X_batch = X_train[idx]
        y_batch = y_train[idx]
        
        theta -= compute_current_alpha("0.05/sqrt(t)", i + 1) * compute_regularized_square_loss_gradient(X_batch, y_batch, theta, lambda_reg)
        theta_hist[i + 1] = theta
        loss_hist[i] = compute_regularized_square_loss(X_batch, y_batch, theta, lambda_reg)
        validation_hist[i] = compute_regularized_square_loss(X_test, y_test, theta, 0)
        
    return theta_hist, loss_hist, validation_hist


def newton_method(X_train, y_train, X_test, y_test, lambda_reg, alpha=0.1, num_iter=1000, batch_size=1):
    """
    使用牛顿法求解岭回归问题，并随着训练过程在验证集上验证

    参数：
        X_train - 训练集特征向量，数组大小 (num_instances, num_features)
        y_train - 训练集标签向量，数组大小 (num_instances)
        X_test - 验证集特征向量，数组大小 (num_instances, num_features)
        y_test - 验证集标签向量，数组大小 (num_instances)
        
        alpha - 梯度下降步长，可自行调整为默认值以外的值。你也可以选择除固定步长以外的策略。
        lambda_reg - 正则化系数，可自行调整为默认值以外的值。
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 训练集上全批量损失函数的历史，数组大小(num_iter)
        validation hist - 验证集上全批量均方误差（不带正则化项）的历史，数组大小(num_iter)
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    validation_hist = np.zeros(num_iter)  # Initialize validation_hist

    # TODO 2.6.2 (optional)
    
    

def main():
    # 加载数据集
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # 增加偏置项
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # 增加偏置项

    # TODO
    
    _, loss_hist_0, validation_hist_0 = stochastic_grad_descent(X_train, y_train, X_test, y_test, 0, 0.05, 1000, 1)
    _, loss_hist_1, validation_hist_1 = stochastic_grad_descent(X_train, y_train, X_test, y_test, 0, 0.05, 1000, 5)
    _, loss_hist_2, validation_hist_2 = stochastic_grad_descent(X_train, y_train, X_test, y_test, 0, 0.05, 1000, 10)
    _, loss_hist_3, validation_hist_3 = stochastic_grad_descent(X_train, y_train, X_test, y_test, 0, 0.05, 1000, 20)
    
    # print(validation_lost_0)
    # _, loss_hist_1, validation_lost_1 = stochastic_grad_descent(X_train, y_train, X_test, y_test, 0, 0.05, 1000, 2)
    
    # _, loss_hist_0 = batch_grad_descent(X_train, y_train, 0, 0.01, 1000, True)
    # _, loss_hist_1 = batch_grad_descent(X_train, y_train, 0, 0.05, 1000, True)
    
    fig = plt.figure()
    
    x = np.arange(0, 1000, 1)
    plt.plot(x, validation_hist_0, label='batch size = 1')
    plt.plot(x, validation_hist_1, label='batch size = 5')
    plt.plot(x, validation_hist_2, label='batch size = 10')
    plt.plot(x, validation_hist_3, label='batch size = 20')
    # plt.plot(x, loss_hist_1, label='batch size = 2')
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    main()
