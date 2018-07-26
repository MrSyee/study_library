# On-policy Prediction with Approximation
---------------------------------------

-	지금까지의 강화학습은 state-action의 value function을 table로 나타내서 문제를 해결하는 **Tabular method** 였다.
-	하지만 state의 크기가 커지면 table로 표현하는 데는 한계가 있고 학습시간도 오래 걸리게 되어 사실상 문제를 풀 수 없다.
-	결국 state space를 **generalization** 해주어야 한다.
-	generalization의 한 방법인 **fucntion approximation** 으로 해결해보자.

### Value-function approximation

-	value function을 **w** 라는 변수로 함수화 해보자. $$ v_\pi(s) \approx \hat v(s, w) $$
-	이제 value function을 update하는 문제에서 w를 update 하는 문제로 바뀐다.
-	function apporximation 하는 방법은 여러가지가 있다.
	-	Linear combinations of features
	-	Artificial Neural Network
	-	decision trees
	-	etc
-	강화학습 문제의 특수성 때문에 모든 방법이 다 맞지는 않는다.
	-	target functions are non-stationary

### The prediction Objective($\bar{VE}$)

-	w를 update 해서 value function을 좋게 하려면 어떻게 해야할까?
-	objective function(or Loss function)을 만들어 maximize(or minimize)한다.
-	target value($v_\pi(s)$)와 approximate value($\hat v(s,w)$) 간의 loss를 구하고 loss가 minimize하는 방향으로 w를 update 하자.
-	Mean Squared Value Error $$ \bar{VE} \doteq \Sigma \mu(s)[v_\pi(s) - \hat v(s,w)]^2 $$
-	$\mu(s)$ 는 state distribution으로 이 state가 value error에 얼마나 영향을 주는가를 나타낸다.
	-	on-policy episodic tasks에서 한 episode에서 이 state에서 얼마나 많은 시간을 보냈는지에 대한 지표를 nomalize 한 값이 된다.
	-	시간을 많이 보낸 state 일수록 value error에 영향을 많이 준다.
-	target value는 RL 알고리즘에 따라 다른 값으로 대체할 수 있다.
	-	For MC, $G_t$
	-	For TD(0), $R_{t+1}+\gamma v(S_{t+1})$
	-	For TD($\lambda$), $G_t^\lambda$

### Stochastic-gradient and Semi-gradient Methods

- w를 update 하기 위해 **Gradient Descent** 를 사용한다.
$$ w_{t+1} = w_t + \Delta w $$
$$ \Delta w = \alpha[v_\pi (S_t) - \hat v(S_t, w_t)]\Delta \hat v(S_t, w_t) $$
- step-by-step update를 위해 **Stochastic Gradient Descent** 방식으로 한다.
