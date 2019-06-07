## Exercise 1: Implement the error back-propagation with the provided Numpy template.

- [column format template](https://github.com/zsdonghao/deep-learning-note/blob/master/Ch5_Neural_Network_Foundation/exercise1_column_format.py)
- [row format template](https://github.com/zsdonghao/deep-learning-note/blob/master/Ch5_Neural_Network_Foundation/exercise1_row_format.py)

## Exercise 2: Use TensorLayer to classify MNIST handwritten digit dataset including training, validating, and testing.

- Study the basic of TensorLayer in [here](https://tensorlayer.readthedocs.io/en/latest/user/get_start_model.html)
- Load MNIST dataset in the vector format:
```python
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
```
- You can use this referenced model.
```python
def get_model(inputs_shape):
    ni = Input(inputs_shape)
    nn = Dropout(keep=0.8)(ni)
    nn = Dense(n_units=800, act=tf.nn.relu)(nn)
    nn = Dropout(keep=0.8)(nn)
    nn = Dense(n_units=800, act=tf.nn.relu)(nn) 
    nn = Dropout(keep=0.8)(nn)
    nn = Dense(n_units=10, act=tf.nn.relu)(nn) 
    M = Model(inputs=ni, outputs=nn, name="mlp")
    return M
```
- Iterating the dataset, see [here](https://tensorlayer.readthedocs.io/en/latest/modules/iterate.html)

```python
for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
    ...
```

- Solution is [here](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_mlp_static.py)

## Exercise 3: Classify the MNIST dataset by modifying the code from exercise 1.

Yes, you are going to implement the error back-propagation by yourself!

## Exercise 4: Following exercise 2, implement a dataflow for data augmentation. (Optional)

- If you want to improve the accuracy, try data augmentation.
- A good dataflow can speed up the training.
