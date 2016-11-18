# 모두를 위한 딥러닝 실습
* 강의 영상: https://www.youtube.com/playlist?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
* 강의 웹사이트: http://hunkim.github.io/ml/
* 코드: https://github.com/FuZer/Study_TensorFlow

## Lec 00 - Machine/Deep learning 수업의 개요와 일정

Acknowledgement

* Andrew Ng
  * https://class.coursera.org/ml-003/lecture
  * http://www.holehouse.org/mlclass - note

* Convolutional Neural Networks for VIsual Recognition
  * http://cs231n.github.io

* Tensorflow
  * https://www.tensorflow.org
  * https://github.com/aymericdamien/TensorFlow-Examples

## Lec 01 - 기본적인 Machine Learning의 용어와 개념

* ML
* Supervised, Unsupervised Learning

## Lab 01 - TensorFlow Basics

### Data Flow Graph

* node: mathematical oprations
* edges: multidimensional data array(tensors)

### Hello World - Everythings is operation 
* 상수나 연산(숫자 더하기 등)이나 텐서플로우 안에서는 모든 게 operation.
* operation은 session에서 실행될 때 값을 가지며 session 밖에서는 값을 갖지 않는 operator로만 정의된다.

### Placeholder

모델의 파라메터를 담는 공간. 함수의 파라메터처럼 모델을 실행시킬때 값을 넘겨줄 수 있음.

## Lec 02 - Linear Regression

일반적인 모델링 순서는

1. 모델 formulize - hypothesis를 표현
2. cost function (loss function) 을 정의
3. training data의 error를 최소화하는 parameter(W, b)를 찾음.

선형모델을 mean squared error를 loss function으로 학습하는 방법

## Lab 02 - Linear Regression

* reduce_mean
* train.GradientDescentOptimizer
  * minimize: 이 operation이 training (eg `train = optimizer.minimize(cost)` )
* initialize_all_variables: 이 함수도 세션 내에서 실행해야 함.
* placeholder를 이용하면 모델을 재사용할 수 있다.

## Lec 03 - How to minimize cost

* linear regression에서 cost(W) 함수는 W에 대한 2차식.
* gradient descent를 사용했음.

update rule: `W = W - alpha * d/dW (cost(W))  
W = W - alpha  * 1 / m sigma ((Wxi - yi) * xi)

linear regression의 cost function은 w에 대한 이차식. convex function으로 진동하지 않는다면 항상 답을 찾을 수 있다.

## Lab 03 - Minimizing Cost

W에 따른 cost function의 변화  
gradient descent 구현

## Lec 04 - Multi-variable(multi-feature) linear regression

matrix로 multi-variable과 bias를 표현  
W 대신 W transpose를 사용.

## Lab 04 - Multi-variable linear regression

placeholder, weight를 matrix 로 표현하고 matrix multiplication으로 구현.

## Lec 05-1 - Logistic (regression) classification

logistic hypothesis = sigmoid(WX) = 1 / ( 1 + exp(-WX))

## Lec 05-2 - Logistic (regression) classification. cost function & gradient descent

바뀐 hypothesis를 기존 mean square error cost function에 그대로 대입하면, cost function이 convex가 아님(local optimum이 있음). 따라서 다른 종류의 cost function이 필요함

cost function = mean (-log (H(x)) : y = 1 or -log(1-H(x)) : y = 0)
= mean(- y log(H) - (1-y) log(1-H)))

minimize cost: gradient descent

## Lab 05 - logistic regression classifier

## Lec 06-1 - Softmax Classification - multinomial classification

여러개의 클래스로 분류

클래스별로 binary classification. 클래스 갯수 만큼의 classification이 필요함  
클래스 별 weight vector를 weight matrix로 만들 수 있음. 각 클래스 별 score 혹은 확률 vector가 matrix multiplication으로 만들어 짐

## Lec 06-2 - Softmax Classification

softmax: output을 확률로 만든다. score의 sum이 1이고, 각각이 0과 1 사이에 있도록.  
그리고 one-hot encoding 해서 하나만 1, 나머지는 0으로.

cost function: cross entropy를 사용한다. `- sum( y log(y_))`

cross entropy는 logistic regression의 cost function을 multinomial로 확장한 일반적인 모양

## Lab 06

**why transpose?** - sample을 row vector로 표현하면 두가지 장점이 있다. 이 경우 W by X 대신 X by W 로 순서가 바뀐다.

1. tf.softmax는 default로 row 방향으로 동작함.
2. python code에서 sample 추가될 때 마다 row 를 추가해주면 된다.

## Lec 07-1 - Learning rate, data preprocessing, overfitting

### Learning rate

* large learning rate: overshooting - 진동, 혹은 발산할 수 있음
* small learning rate: 시간이 너무 오래 걸리거나 local minimum

### Data (X) preprocessing

normalize - feature의 scale 차이가 큰 데이터를 learning하면 scale의 폭이 좁은 feature에 대해서는 성능이 심하게 움직일 수 있어서 안정적이지 않음. 이때 normalize가 필요함.

standardization: x' = (xj - u) / sigma j

### overfitting

트레이닝 데이터를 늘이거나, 피쳐를 줄이거나, regularization을 한다.

regularization?

decision boundary를 데이터에 맞게 복잡한 모양으로 만들면 overfitting의 가능성이 높다고 볼수 있다. learning weight가 크다면 decision boundary가 구부러져 있을 수 있으므로 weight의 크기를 제한하여 decision boundary를 펴는 것이 regularization이다.

cost에 weight의 크기를 반영하여 learning하면 되는데, 예를 들면 weight 제곱 등에 regularization strength를 곱한 값을 cost계산 시 더해주면 된다.

`l2reg = 0.001 * tf.reduce_sum(tf.square(W))`

## Lec 07-2 - Learning and test data sets

learning한 모델이 얼마나 잘 동작하나 - 모델 성능 평가

트레이닝데이터로 평가한다면? 외우면 되므로 다 맞출 수 있다. training과 test set을 나눠서 training set은 모델 학습하고 test set은 점수 평가에만 사용해야 한다.

test set은 단 한번만 사용해야 한다.

validation set은? learning rate alpha, regularization strength lambda의 튜닝은 validation set으로 결정(모의시험 처럼). 모델은 training set으로 build.

online learning - training set을 잘게 잘라서 한 덩어리씩 점진적으로 학습.

평가 지표

* accuracy - 예측 성공 갯수 / 전체 데이터 갯수

## Lec 08-1 - Deep Neural Nets for Everyone

activate(sum of (wx) + bias)

xor problem

MLP - multilayer perceptron - xor을 풀 수 있음. 하지만 training할 수 없어 - minsky

하지만 backpropagation 으로 weight, bias를 update할 수 있음. - paul werbos, geoffrey hinton

CNN

layer가 많으면 backpropagation이 잘 동작하지 않았음. SVM이나 RandomForest등 다른 알고리즘들을 더 많이 사용하게 됨.

## Lec 08-2

2006, 2007의 논문 내용은 초기값을 잘 주면 deep network도 학습할 수 있다. 어려운 문제를 푸는데 오히려 효율적이다.

ImageNet에서 CNN으로 높은 성능을 내서 유명해짐.

...

역사. 족보








