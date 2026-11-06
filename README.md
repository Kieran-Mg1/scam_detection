
## Introduction

The most considerable issue to me in scam detection is class imbalance. BCE is the negative log-likelihood of a Bernoulli distribution. A logistic regression model with weights $\mathbf{w}$ and intercept $b$ assumes$$
Y_i \mid \mathbf{x}_i \sim \operatorname{Bernoulli}(p_i)
$$where $$
p_i = \sigma(\mathbf{w}^{\top}\mathbf{x}_i + b)
$$The maximum-likelihood estimates are
$$
(\hat{\mathbf{w}}, \hat{b})
=
\arg\max_{\mathbf{w},b}
\sum_{i=1}^{N}
\left[
y_i\log p_i + (1-y_i)\log(1-p_i)
\right]
$$
At the optimum, the gradient with respect to the intercept is zero
$$
\frac{\partial \mathcal{L}}{\partial b}
=
\sum_{i=1}^{N}(p_i-y_i)=0
$$Therefore

$$
\sum_{i=1}^{N}\hat{p}_i
=
\sum_{i=1}^{N}y_i
$$
and $$
\boxed{
\frac{1}{N}\sum_{i=1}^{N}\hat{p}_i
=
\frac{N_{\text{scams}}}{N_{\text{total}}}
}
$$Thus, logistic regression learns the empirical scam base rate i.e. the mean predicted scam probability equals the observed scam prevalence. Assuming features weakly separate classes, using conventional BCE prediction thresholds of $\gtrless 0.5$  will mean on average, scam recall is poor, since $N_{\text{total}} \gg N_{\text{scams}}$.

This produces a tradeoff. We want high recall to detect scams, and sufficient precision to not block legitimate transfers. Weighting the objective function directly produces a parameter that would require model re-training over a grid search to optimise, as well as possible increase in gradient variance. IMO it is preferable to choose a decision threshold as a separate optimisation problem, and then use the optimised threshold(s) as heuristic at test time. This framework may be provided already, in which case the below can be discarded. 

You could choose a threshold from a grid search, and select the threshold with the best F1 score. I noticed in Table 2 of https://arxiv.org/pdf/2405.11230 , that this does not produce the lowest total cost. It is not clear how total cost is calculated, but I considered it could be something like this. Let
- $A_i$: Transaction amount.
- $y_i \in \{0,1\}$: True scam label.
- $\hat y_i(t) = \mathbb{1}[p_i \geq t]$: Predicted scam label.
- $c_{\mathrm{FN}}(A_i)$: False negative transaction amount $A_i$ (allowed scam transfer).
- $c_{\mathrm{FP}}(A_i)$: False positive transaction amount $A_i$ (blocked legitimate transfer).
$$
\boxed{
C(t)=
\underbrace{
\sum_{i=1}^{N}y_i(1-\hat y_i(t))c_{\mathrm{FN}}(A_i)
}_{\text{Cost of missed scams}}
+
\underbrace{
\sum_{i=1}^{N}(1-y_i)\hat y_i(t)c_{\mathrm{FP}}(A_i)
}_{\text{Cost of blocking legitimate transfers}}
}
$$
This is an oversimplification of what business costs may be. Repeatedly blocking small legitimate transfers may cause customers to become frustrated and leave the platform for example, and does not account for regulatory restrictions. Alternatively, there may be specifc constraints such as a minimum recall and maximum false positive rate. I went for this constraint in the end over the cost function.

The rest of the ML pipeline should follow standard ML task practices. Dataset used was https://huggingface.co/datasets/EpiphanyTech/SynSEPA. Please note this whole project was done in one and a half days and as such is not particularly tidy.

## Results

Utilised a Random Forest classifier using a chronological 60/20/20 train, validation and test split, with time-series cross-validation on the training set. Rather than optimising F1 or transaction-weighted cost, I selected a decision threshold on the validation set subject to two business constraints:
- **Minimum recall:** 85% of scams must be detected.
- **Maximum false positive rate:** No more than 1% of legitimate transactions may be incorrectly flagged.

The logit threshold which optimsied this was then applied to the out-of-time test set without further optimisation. The model returned good train and validation performance, but generalised poorly on the test set. Observing the Gini impurity of the Random Forest splits showed the model relied heavily on beneficiary country. After some investigating, I found this was due to distribution shift over time between the train and test sets. Six beneficiary countries (IT, BE, DE, ES, FR and NL) had no scams in the training or validation sets, but collectively contained 1,516 scams in the test set. This results in the model using a feature for predictions on the train set that doesn't generalise to the test set. Removing these features collapsed model performance, so all that was left was to try and regularise the model. Since this is my own made up example, I'm not sure how achievable the business constraints are, so have not pursued a full parameter sweep to this end. Overall this demonstrates the limitations of such models. The dataset itself mentioned about using transformers, so perhaps they overcome this issue.

## Other considerations

### Splitting
An issue I considered was leakage when doing a stratified split for a dataset of transactions (https://huggingface.co/datasets/EpiphanyTech/SynSEPA). With a random stratified split, transactions are assigned to training and test sets regardless of when they occurred. As a result, the model may be trained on later transactions and evaluated on earlier ones, potentially giving an overly optimistic estimate of future performance. This is mentioned in https://wise.jobs/when-money-has-no-borders-your-data-cant-either. 

### SMOTE
Synthetic Minority Oversampling Technique (SMOTE) may alleviate the need to perform an optimisation over boundaries, but I'd need more time to consider if this introduces any bias. To me in theory SMOTE just moves the decision boundary back towards $\gtrless 0.5$. I also think using the distributional change information that occured in the test set to generate synthetic examples for training is cheating, so did not perform this.  


