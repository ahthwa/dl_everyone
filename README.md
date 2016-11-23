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

## Lec 09-x - backpropagation 미분

미분의 chain rule: f(g(x))의 미분 - df/dx = df/dg * dg/dx

backpropagation의 미분에서 chain rule을 응용함.

## Lec 09-2

f = wx + b 에서 df/dw, df/dx, df/db 는 chain rule로 거꾸로 계산할 수 있다.

sigmoid 함수도 graph로 표현한 다음 chain rule을 이용하여 미분을 구할 수 있다.

* g(z) = 1 / (1 + e^ -z)
* z -> z * -1 (=v1) -> exp(v1) (=v2) -> v2 + 1 (=v3) -> 1/(v3) -> g
* dg/dz = dg/dv3 * dv3/dv2 * dv2/dv1 * dv1/dz = g * (1-g)

tensorflow에서는 식을 tensor와 operator의 graph로 표현하는데, chain rule을 이용한 미분이 쉬워진다. 이 graph는 tensor board를 통해 확인할 수 있다.

## Lab09-2 Tensor Board

5 steps

* TF graph 중에서 출력할 node를 결정 (`tf.histogram_summary`, `tf.scalar_summary`)
  * scalar\_summary는 event 페이지에 표시된다. cost, accuracy 등 평가 지표를 보여주기 적합하다.
  * histogram\_summary는 distributions, histograms 페이지에서 볼 수 있다.
* summary를 merge (`tf.merge_all_summaries()`)
* summary writer 생성 (`writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph)`)
* summary 실행하고 add\_summary를 적절한 step마다 호출 (`summary = sess.run(merged, ...)`, `writer.add_summary(summary, step)`)
* tensorboard 실행 (`tensorboard --logdir=/tmp/mnist_logs`)

## Lec 10-1, 2

layer를 여럿 쌓아서 deep network 만드는 법을 배웠다.

xor문제를 9 layer deep network으로 풀어보면 cost가 줄어들지 않는다. 학습이 되지 않음.

backpropagation을 해 보면 sigmoid의 output이 0~1 사이이기도 하고 매우 작은 값이 나올 수도 있음. 그래서 gradient가 layer를 넘어가면서 점점 작아진다. 그리고 chain rule에 의해 계속 곱해지면서 gradient가 거의 0에 가까워진다. - vanishing gradient 86년 backpropagation 발견 후 20년간의 2차 겨울을 맞게 된다.

문제가 뭐였고 어떻게 풀었을까?

* sigmoid가 적절하지 않다.
  * 대신 ReLU(Rectified Linear Unit)를 사용한다. 마지막 layer는 확률 form으로 만들기 위해 sigmoid를 사용.
  * sigmoid, ReLU 이외의 activation function은 tanh, maxout, ELU, Leaky ReLU 등이 있다.
* weight 초기화를 잘못했다.
  * ex) 모든 weight를 0으로 초기화하면 gradient가 0이 되는 문제가 있음.
  * A fast learning algorithm for deep belief nets 논문에서 Restricted Boatman Machine(RBM)을 이용한 초기화를 제안함.  
  RBM은 뭐냐? forward(encoder)시 input x와 backward(decoder)시 recreate된 x가 차이가 최소가 되도록 weight를 조정하는 것.
  * how to RBM
    * 인접한 두 레이어 사이에서 encode/decode를 반복하여 입력이 유지되도록 weight를 잡는다 - pre-training을 통해 초기화.
    * pre-training을 해 놓은 상태에서의 learning은 fine tuning이라고 부를 정도로 빠르게 처리됨.
  * RBM을 안써도 되는 다른 방법들이 발견됨.
    * Xavier initialization (2010): 노드의 입력, 출력 갯수에 맞춰서 세팅 하는 방법 - `W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)`
    * He initialize (2015) - `W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in / 2)`
* labeling된 데이터가 너무 적다.
* 계산 속도가 너무 느리다.

## Lec 10-3 - Drop out과 ensemble

오버피팅은 알겠는데 오버피팅 되어있는지 어떻게 알 수 있나 - layer, feature등 모델 복잡도가 높아졌는데 testing accuracy가 점점 낮아진다면 오버피팅.

* more training data
* reduce feature - deep learning에서는 적합하지 않은 방법.
* regularization
* neural net에서는 drop out이라는 방법도 있다.

dropout

학습할 때 랜덤하게 일부 노드를 쓰지않는 방법.
주의할 점은 학습하는 동안만 drop한다는 것. evaluate, prediction에서는 dropout을 하지 않고 전체 노드를 사용한다.

ensemble

data set을 n개로 쪼개고 network도 n개를 만들어서 각각 따로따로 학습시키고, 마지막에 n개의 network을 모두 합치는 방법.

## Lec 10-4

자유롭게 쌓고 점프하고 split했다가 merge도 하고 할 수 있다.  
신경 세포의 연결에 제약이 없는 것 처럼.

연결을 옆으로 한 것 - RNN

## Lab 10 - NN, ReLU, Xavier, Dropout and Adam

gradient descent 대신 adam optimizer를 사용하고 있다. 현존하는 가장 빠른 optimizer라고.

[Alec Radford's animations for optimization algorithms](www.denizyuret.com/2015/03/alec-radfords-animations-for.html)

