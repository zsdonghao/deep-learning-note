## Exercise 1: CIFAR10 Classification with TensorLayer


- Load CIFAR10 with TensorLayer API:

```python
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
```
- Data augmentation with TensorFlow Dataset API, see [here](https://tensorflow.org/guide/datasets)

- Solution is [here](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_cifar10_cnn_static.py)

## Exercise 2:
