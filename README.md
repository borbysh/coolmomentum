# A Simulated Annealing Optimizer for Deep Neural Networks 

Stochastic Optimization by Langevin Dynamics with Simulated Annealing



### Requirements

• Python (3 or higher)

• Tensorflow
 2.x 

### How to use

```python
opt=Coolmomentum(learning_rate=0.01, beta_1=0.99, beta_2=0.99997)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```


• learning_rate is a squared timestep. Default learning_rate=0.01

• beta_1 is an initial value of the momentum coefficient. Default beta_1=0.99 

• beta_2 is a cooling rate, being a Simulated Annealing parameter. Calculated as beta_2=(1-beta_1)^(1/S), where S is a total number of iterations. If beta_2=1 the momentum coefficient is constant and Simulated Annealing is not applied. 


### Comparison with Adam and Momentum optimizers 

The comparison was done on CIFAR10 with ResNet, using 
[this](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) keras code


![Training results](https://github.com/borbysh/coolmomentum/blob/master/Train_loss.png)

![Training results](https://github.com/borbysh/coolmomentum/blob/master/Train_loss_mean.png)

![Training results](https://github.com/borbysh/coolmomentum/blob/master/Test_loss.png)

![Training results](https://github.com/borbysh/coolmomentum/blob/master/Test_loss_mean.png)
![Training results](https://github.com/borbysh/coolmomentum/blob/master/Temperature.png)



Kirkpatrick, Scott, C. Daniel Gelatt, and Mario P. Vecchi. "Optimization by simulated annealing." Science 220.4598 (1983): 671-680.


Vanden-Eijnden, Eric, and Giovanni Ciccotti. "Second-order integrators for Langevin equations with holonomic constraints." Chemical physics letters 429.1-3 (2006): 310-316.
