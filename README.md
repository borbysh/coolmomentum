# A Simulated Annealing Optimizer for Deep Neural Networks 



Stochastic Optimization by Langevin Dynamics with Simulated Annealing





### Requirements


• Python (3 or higher)


• Pytorch or Tensorflow 2.x 



### Usage



In TensorFlow:

```python
opt=Coolmomentum(learning_rate=0.01, beta_1=0.99, beta_2=0.99997)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```



• learning_rate is a squared timestep. Default learning_rate=0.01


• beta_1 is an initial value of the momentum coefficient. Default beta_1=0.99 


• beta_2 is a cooling rate, being a Simulated Annealing parameter. Calculated as beta_2=(1-beta_1)^(1/S), where S is a total number of iterations. 
If beta_2=1 the momentum coefficient is constant and Simulated Annealing is not applied. 





### Comparison with Adam and Momentum optimizers on CIFAR-10 with ResNet-20 


The comparison was done using 
[this](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) keras code



![Training results](https://github.com/borbysh/coolmomentum/blob/master/Train_loss.png)


![Training results](https://github.com/borbysh/coolmomentum/blob/master/Temperature.png)



### Comparison with SGD optimizer on the Penn Treebank dataset with LSTM 


In PyTorch:



The comparison was done using 
[this](https://github.com/salesforce/awd-lstm-lm) code, by running



python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 



For the honest comparison of SGD and CoolMomemtum the ASGD optimizer was not used.


SGD was replaced by CoolMomentum with commands

```python
from coolmom_pytorch import SGD
optimizer = SGD(params, lr=0.1, momentum=0.99,  weight_decay=args.wdecay, beta=0.9999998018)
```



![Training results](https://github.com/borbysh/coolmomentum/blob/master/Figure_LSTM.png)





Kirkpatrick, Scott, C. Daniel Gelatt, and Mario P. Vecchi. "Optimization by simulated annealing." Science 220.4598 (1983): 671-680.




Ma, Y. A., Chen, Y., Jin, C., Flammarion, N., & Jordan, M. I. "Sampling can be faster than optimization". Proceedings of the National Academy of Sciences, 116 (2019) 20881-20885.
