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

Keyword: ML, Supervised, Unsupervised Learning

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

Keyword: 학습 순서 (hypothesis, cost function, optimize), linear regression의 hypothesis, cost function(mean square error), optimize 방법(gradient descent)

## Lab 02 - Linear Regression

* reduce_mean
* train.GradientDescentOptimizer
  * minimize: 이 operation이 training (eg `train = optimizer.minimize(cost)` )
* initialize_all_variables: 이 함수도 세션 내에서 실행해야 함.
* placeholder를 이용하면 모델을 재사용할 수 있다.

## Lec 03 - How to minimize cost

* linear regression의 cost function - 2차식. convex function
* gradient descent

update rule: `W = W - alpha * d/dW (cost(W))  
W = W - alpha  * 1 / m sigma ((Wxi - yi) * xi)

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

logistic regression의 cost function = mean (-log (H(x)) : y = 1 or -log(1-H(x)) : y = 0) = mean(- y log(H) - (1-y) log(1-H)))

## Lab 05 - logistic regression classifier

## Lec 06-1 - Softmax Classification - multinomial classification

여러개의 클래스로 분류할 때 클래스 갯수만큼 classification weight vector 필요. weight matrix 하나로 표현.

## Lec 06-2 - Softmax Classification

* softmax - output score를 확률로 표현(score 합이 1이고 각 score는 0과 1 사이의 값을 갖도록). classification은 one-hot encoding.
* cost function: cross entropy. `- sum( y log(y_))`  
cross entropy는 logistic regression의 cost function을 multinomial로 확장한 일반적인 모양

## Lab 06

**why transpose?** - sample을 row vector로 표현하면 두가지 장점이 있다. 이 경우 W by X 대신 X by W 로 순서가 바뀐다.

1. tf.softmax는 default로 row 방향으로 동작함.
2. python code에서 sample 추가될 때 마다 row 를 추가해주면 된다.

## Lec 07-1 - Learning rate, data preprocessing, overfitting

Keyword: Learning rate, preprocessing(normalize), overfitting

### Learning Rate

* large learning rate: overshooting - 진동, 혹은 발산할 수 있음
* small learning rate: 시간이 너무 오래 걸리거나 local minimum에 빠질 수 있음.

### Data (X) preprocessing

normalize - feature의 scale 차이가 큰 데이터를 learning하면 scale의 폭이 좁은 feature에 대해서는 성능이 심하게 움직일 수 있어서 안정적이지 않음. 이때 normalize가 필요함.

standardization: x' = (xj - u) / sigma j

### overfitting

트레이닝 데이터를 늘이거나, 피쳐를 줄이거나, regularization을 한다.

**regularization**

decision boundary를 데이터에 맞춰서 복잡한 모양으로 만들면 overfitting의 가능성이 높아진다.
decision boundary를 펴 주기 위해서 weight의 크기에 penalty를 주는 것이 regularization.

`l2reg = 0.001 * tf.reduce_sum(tf.square(W))`

## Lec 07-2 - Learning and test data sets

Keyword: test data, validation data, online learning, accuracy

learning한 모델의 성능 평가

* 트레이닝데이터로 평가한다면? 외우면 되므로 다 맞출 수 있다. training과 test set을 나눠서 training set은 모델 학습하고 test set은 점수 평가에만 사용해야 한다.
* test set은 단 한번만 사용해야 한다.
* validation set은? learning rate alpha, regularization strength lambda의 튜닝은 validation set으로 결정(모의시험 처럼). 모델은 training set으로 build.

online learning - training set을 잘게 잘라서 한 덩어리씩 점진적으로 학습.

평가 지표

* accuracy - 예측 성공 갯수 / 전체 데이터 갯수

## Lec 08-1,2 - Deep Neural Nets for Everyone

neural network의 역사(족보)

* 뇌의 neuron - weight sum후에 activate하는 구조
* xor problem
* MLP(multi-layer perceptron)
* backpropagation(paul werbos, geoffrey hinton)
* CNN(convolutional neural netowkr)
* deep network 학습의 어려움과 SVM/Random Forest등
* deep network 학습 문제 해결 방법

## Lec 09-1 - Neural Nets for XOR

