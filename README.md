
## Introduction

The most considerable issue to me in scam detection is class imbalance. BCE is the negative log-likelihood of a Bernoulli distribution. A logistic regression model with weights $\mathbf{w}$ and intercept $b$ assumes

$$
Y_i \mid \mathbf{x}_i \sim \mathrm{Bernoulli}(p_i)
$$

where

$$
p_i = \sigma(\mathbf{w}^{\top}\mathbf{x}_i + b)
$$

The maximum-likelihood estimates are

$$(\hat{\mathbf{w}}, \hat{b}) =
\arg\max_{\mathbf{w},b}
\sum_{i=1}^{N}
\left[
y_i\log p_i + (1-y_i)\log(1-p_i)
\right]$$

At the optimum, the gradient with respect to the intercept is zero

$$ \frac{\partial \mathcal{L}}{\partial b} =
\sum_{i=1}^{N}(p_i-y_i)=0
$$

Therefore

$$ \sum_{i=1}^{N}\hat{p}_i =
\sum_{i=1}^{N}y_i
$$

and

$$ \boxed{ \frac{1}{N}\sum_{i=1}^{N}\hat{p}_i =
\frac{N_{\text{scams}}}{N_{\text{total}}}
}
$$

Thus, logistic regression learns the empirical scam base rate i.e. the mean predicted scam probability equals the observed scam prevalence. Assuming features weakly separate classes, using conventional BCE prediction thresholds of $\gtrless 0.5$ will mean on average, scam recall is poor, since $N_{\text{total}} \gg N_{\text{scams}}$.

This produces a tradeoff. We want high recall to detect scams, and sufficient precision to not block legitimate transfers. Weighting the objective function directly produces a parameter that would require model re-training over a grid search to optimise, as well as possible increase in gradient variance. IMO it is preferable to choose a decision threshold as a separate optimisation problem, and then use the optimised threshold(s) as heuristic at test time. This framework may be provided already, in which case the below can be discarded.

You could choose a threshold from a grid search, and select the threshold with the best F1 score. I noticed in Table 2 of https://arxiv.org/pdf/2405.11230 , that this does not produce the lowest total cost. It is not clear how total cost is calculated, but I considered it could be something like this. Let

- $A_i$: Transaction amount.
- $y_i \in \{0,1\}$: True scam label.
- $\hat y_i(t) = \mathbb{1}[p_i \geq t]$: Predicted scam label.
- $c_{\mathrm{FN}}(A_i)$: False negative transaction amount $A_i$ (allowed scam transfer).
- $c_{\mathrm{FP}}(A_i)$: False positive transaction amount $A_i$ (blocked legitimate transfer).

$$
\boxed{
\begin{aligned}
C(t)=&
\underbrace{
\sum_{i=1}^{N}y_i(1-\hat y_i(t))c_{\mathrm{FN}}(A_i)
}_{\text{Cost of missed scams}}
&+
\underbrace{
\sum_{i=1}^{N}(1-y_i)\hat y_i(t)c_{\mathrm{FP}}(A_i)
}_{\text{Cost of blocking legitimate transfers}}
\end{aligned}
}
$$

This is an oversimplification of what business costs may be. Repeatedly blocking small legitimate transfers may cause customers to become frustrated and leave the platform for example, and does not account for regulatory restrictions. I wanted to try and tie the optimisation to something outside of label frequency. Ideally you could compare $\arg\max_t F_1(t)$ and $\arg\min_t C(t)$. In practice, this can be realised as generating a confusion matrix for each threshold in a grid search to find the F1 score, and then additionally use a transaction amount field to find $C(t)$.

The rest of the ML pipeline should follow standard ML task practices. Dataset used was https://huggingface.co/datasets/EpiphanyTech/SynSEPA. Please note this whole project was done in one and a half days and as such is not particularly tidy.

## Other Considerations

### Splitting

An issue I found in testing was leakage when doing a stratified split for a dataset of transactions (https://huggingface.co/datasets/EpiphanyTech/SynSEPA). With a random stratified split, transactions are assigned to training and test sets regardless of when they occurred. As a result, the model may be trained on later transactions and evaluated on earlier ones, potentially giving an overly optimistic estimate of future performance.

Generating any historical features after splitting is problematic too. For example, if you generate a feature derived from point-in-time values across the whole dataframe, such as a cumulative sum for sender and beneficiary previous transfers, you will leak information across a subsequent stratified splits. If you generate the feature per split, then an account with a total value of 4 prior transactions in the train set will have this value reset to at least zero in the test set, which is incorrect.

It seems a similar issue to these was encountered in https://wise.jobs/when-money-has-no-borders-your-data-cant-either.

### SMOTE

Synthetic Minority Oversampling Technique (SMOTE) may alleviate the need to perform an optimisation over boundaries, but I'd need more time to consider if this introduces any bias. To me in theory SMOTE just moves the decision boundary back towards $\gtrless 0.5$.

## Results

The cost-optimal threshold reduces cost on the validation set by £4,094.55 (approximately 0.09%) under current settings with logistic regression.

- F1 optimisation favours a lower threshold, catching more scams but incorrectly flagging more legitimate transactions.
- Cost optimisation favours a higher threshold, reducing false positives but missing more scams.
- The test F1 scores are close in either case.
