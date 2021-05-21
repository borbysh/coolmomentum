# CoolMomentum Optimizer for Deep Neural Networks 


## Stochastic Optimization by Langevin Dynamics with Simulated Annealing


This repository contains implementations for [CoolMomentum: A Method for Stochastic Optimization by Langevin Dynamics with Simulated Annealing](https://www.nature.com/articles/s41598-021-90144-3) (published in Scientific Reports) in TensorFlow and PyTorch.

### Usage

In TensorFlow:

```python
from coolmomentum_tf import Coolmomentum                           
opt=Coolmomentum(learning_rate=0.01, rho_0=0.99, alpha=0.99997)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```



• learning_rate is a squared timestep "dt^2". Default learning_rate=0.01.                   
• rho_0 is an initial value of the momentum coefficient. Default rho_0=0.99.                   
• alpha is a cooling rate, being a Simulated Annealing parameter. Calculated as alpha=(1-rho_0)^(1/S), 
  where S is a total number of iterations. If alpha=1 the momentum coefficient is constant 
  and Simulated Annealing is not applied. Then the optimizer behaves like simple Momentum.   

### Benchmarking Coolmomentum on CIFAR-10 with ResNet-34 
Requirements: Python 3.6+, Pytorch 1.3+, tqdm

The benchmarking was done by modification of 
[this](https://github.com/ifeherva/optimizer-benchmark/blob/master/train_cifar10.py) code by Istvan Fehervari and running:

• python train_cifar10.py       

The results obtained are compared against those calculated by Istvan Fehervari for [other popular optimizers](https://app.wandb.ai/ifeherva/optimizer-evaluation):

![Train Loss](https://github.com/borbysh/coolmomentum/blob/master/Figure_1_a.png)
![Train accuracy](https://github.com/borbysh/coolmomentum/blob/master/Figure_1_b.png)
![Test accuracy](https://github.com/borbysh/coolmomentum/blob/master/Figure_1_c.png)

### Benchmarking Coolmomentum on Imagenet with Efficientnet-B0 

The benchmarking was done by modification of 
[this](https://cloud.google.com/tpu/docs/tutorials/efficientnet) code by Tensorflow Team and running:

python3 main_cool.py \
  --tpu=${TPU_NAME} \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --model_name='efficientnet-b0' \
  --skip_host_call=true \
  --train_batch_size=2048 \
  --train_steps=218948

The result obtained is compared to that of the original [code](https://cloud.google.com/tpu/docs/tutorials/efficientnet) by Tensorflow Team:

![Top accuracies](https://github.com/borbysh/coolmomentum/blob/master/Figure_2.png)






### Comparison with Adam and SGD optimizers on CIFAR-10 with ResNet-20 
Requirements: Python 3.6+, Tensorflow 2.x

The comparison was done by modification of 
[this](https://keras.io/zh/examples/cifar10_resnet/) Keras example and running

• python resnet_adam200.py       
• python resnet_sgd200.py      
• python resnet_cool200.py       

![Training results](https://github.com/borbysh/coolmomentum/blob/master/Fig_1_a.png)
![Training results](https://github.com/borbysh/coolmomentum/blob/master/Fig_1_b.png)

 Rescaled temperature:
 
![Training results](https://github.com/borbysh/coolmomentum/blob/master/Fig_1_c.png)


### Comparison with SGD optimizer on the Penn Treebank dataset with LSTM 
Requirements: Python 3.6+, Pytorch 1.3+

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



### References: 

Kirkpatrick, Scott, C. Daniel Gelatt, and Mario P. Vecchi. "Optimization by simulated annealing." Science 220.4598 (1983): 671-680.




Ma, Y. A., Chen, Y., Jin, C., Flammarion, N., & Jordan, M. I. "Sampling can be faster than optimization". Proceedings of the National Academy of Sciences, 116 (2019) 20881-20885.
