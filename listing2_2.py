# Regularization
import matplotlib.pylab as pylab
import numpy

x = numpy.linspace(-1, 1, 100) # -1 - 1之间等间距生成100个数
signal = 2 * x * x + x + 2 # 信号
noise = numpy.random.normal(0, 0.1, 100) # 高斯分布，sigma=0.1
y = signal + noise # 输出
x_train = x[0:80] # 训练集 - 输入
y_train = y[0:80] # 训练集 - 输出

train_rmse = []
test_rmse = []
degree = 80
lambda_reg_values = numpy.linspace(0.01, 0.99, 100)

for lambda_reg in lambda_reg_values:
    X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, degree)]) # 将x_train ** degree之后，作为列填入X_train中，最后得出X_train为80*80
    model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(), X_train) + lambda_reg * numpy.identity(degree)), X_train.transpose()), y_train) # beta = (X^TX - lambdaI)^(-1)X^Ty / numpy.dot 点乘 /numpy.linalg.inv() 逆矩阵
    predicted = numpy.dot(model, [numpy.power(x, i) for i in range(0, degree)]) # [numpy.power(x, i) for i in range(0, degree)] 创建了80*100的矩阵
    train_rmse.append(numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80]))))
    test_rmse.append(numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicted[80:], y[80:] - predicted[80:]))))

pylab.plot(lambda_reg_values, train_rmse)
pylab.plot(lambda_reg_values, test_rmse)
pylab.xlabel('$\lambda')
pylab.ylabel('RMSE')
pylab.legend(['Train', 'Test'], loc=2)
pylab.show()