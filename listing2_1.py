# Generate Toy Dataset
import matplotlib.pylab as pl
import numpy

x = numpy.linspace(-1, 1, 100)
signal = 2 + x + 2 * x * x
noise = numpy.random.normal(0, 0.1, 100)
y = signal + noise

pl.plot(signal, 'b')
pl.plot(y, 'g')
pl.plot(noise, 'r')
pl.xlabel('x')
pl.ylabel('y')
pl.legend(['Without Noise', 'With Noise', 'Noise'], loc=2)
x_train = x[0:80]
y_train = y[0:80]

# Model with degree1
pl.figure()
degree = 2
X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, degree)])
model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
pl.plot(x, y, 'g')
pl.xlabel('x')
pl.ylabel('y')
predicated = numpy.dot(model, [numpy.power(x, i) for i in range(0, degree)])
pl.plot(x, predicated, 'r')
pl.legend(['Actual', 'Predicated'], loc=2)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicated[0:80], y_train - predicated[0:80])))
test_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicated[80:], y[80:] - predicated[80:])))
print('Train RMSE (Degree = 1)', train_rmse1)
print('Test RMSE (Degree = 1)', test_rmse1)

# Model with degree 2
pl.figure()
degree = 3
X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, degree)])
model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
pl.plot(x, y, 'g')
pl.xlabel('x')
pl.ylabel('y')
predicated = numpy.dot(model, [numpy.power(x, i) for i in range(0, degree)])
pl.plot(x, predicated, 'r')
pl.legend(['Actual', 'Predicated'], loc=2)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicated[0:80], y_train - predicated[0:80])))
test_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicated[80:], y[80:] - predicated[80:])))
print('Train RMSE (Degree = 2)', train_rmse1)
print('Test RMSE (Degree = 2)', test_rmse1)

# Model with degree 8
pl.figure()
degree = 9
X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, degree)])
model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
pl.plot(x, y, 'g')
pl.xlabel('x')
pl.ylabel('y')
predicated = numpy.dot(model, [numpy.power(x, i) for i in range(0, degree)])
pl.plot(x, predicated, 'r')
pl.legend(['Actual', 'Predicated'], loc=3)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicated[0:80], y_train - predicated[0:80])))
test_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicated[80:], y[80:] - predicated[80:])))
print('Train RMSE (Degree = 8)', train_rmse1)
print('Test RMSE (Degree = 8)', test_rmse1)