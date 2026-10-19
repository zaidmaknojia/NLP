# Module 1: From text to a trained sentiment classifier

An interactive study guide for AI 388. Based on all eight Module 1 lecture transcripts, all three course exercise screenshots, the Fall 2026 topic sheet, relevant past midterms, and the local Assignment 1 implementation.

## How we'll study together

Work in this order: **features → scores → perceptron → losses → logistic regression → optimization → mixed practice**. Expect about 90–120 minutes across several sessions; take longer wherever you need it.

For each checkpoint, answer before opening its answer. In our chat, send your reasoning and confidence: **sure / partly sure / guessing**. I'll check the reasoning, give a hint when useful, and ask a follow-up before we move on. You can say “hint,” “show a worked example,” or “quiz me on section 4.” An incorrect answer tells us what to practice next.

Questions marked **Course exercise** reproduce or closely adapt the supplied exercises. **Past exam** questions name the source. Other practice questions are original. The answers are reference solutions; looking at them is optional during our guided review.

### What you should be able to do

- Build count and indicator vectors in a fixed vocabulary and explain what they discard.
- Compute a linear score, make a prediction, and handle a stated tie rule.
- Trace perceptron updates and connect them to a loss and subgradient.
- Compute logistic probabilities, negative log likelihood, gradients, and updates.
- Explain separability, useful sentiment features, learning-rate behavior, and generalization.
- Connect the equations to Assignment 1 without mixing label conventions.

## 1. The pipeline and our notation

**Raw text → tokens → features → score → prediction.** During training, a gold label also enters a loss; the loss determines how weights change.

For “the movie was good,” tokenization identifies units, feature extraction turns them into numbers, and the classifier weights those numbers. Training learns the weights from labeled examples. Prediction uses the resulting weights without updating them.

| Symbol | Meaning |
|---|---|
| x | An input sentence or document |
| f(x) ∈ Rᵈ | Its feature vector; d is the number of features |
| w ∈ Rᵈ | Learned weights, one per feature |
| z = wᵀf(x) | Real-valued score, also called a logit in logistic regression |
| y ∈ {−1,+1} | Gold label in the lecture/perceptron convention |
| t = (y+1)/2 ∈ {0,1} | Same gold label in the binary cross-entropy convention |
| ŷ | Predicted label |
| p | P(y=+1 given x), the model's positive-class probability |
| α > 0 | Step size / learning rate |
| N | Number of training examples |

**Default for this guide:** positive iff z > 0; a zero score is negative. Exceptions explicitly state a different rule. No bias feature is included in numerical exercises unless stated.

An epoch is one pass through the training examples. An update is one change to the weights; an epoch may contain many updates.

### Checkpoint 1 — Separate the objects

For “good movie,” identify which of these is learned during training: the input text, its fixed bag-of-words counts, the weight on “good,” or its gold label.

<details><summary>Reveal answer</summary>

The **weight on “good”** is learned. With a fixed feature-extraction procedure, the feature values come from the text. The gold label is supplied by the dataset. In this module, we design the feature representation and learn the classifier weights.

</details>

## 2. Feature extraction: what the model gets to see

### Count versus indicator features

Fix the vocabulary order **[good, bad, not, movie]**. For “good good movie”:

| Representation | Vector | Meaning of the first coordinate |
|---|---|---|
| Count | [2,0,0,1] | “good” occurs twice |
| Indicator / presence | [1,0,0,1] | “good” occurs at least once |

The dimension is vocabulary size, not sentence length. A 10,000-word vocabulary produces 10,000-dimensional vectors even for a four-token sentence. Most coordinates are zero, so sparse storage is useful.

Unigram bag-of-words discards word order. “dog bites man” and “man bites dog” have identical unigram vectors. Any classifier receiving only that vector must treat the two inputs identically.

### Bigrams and other features

An n-gram is a sequence of n consecutive tokens. “the movie was good” has bigrams **the movie**, **movie was**, **was good**, assuming no boundary tokens. A bag of bigrams preserves these local pairs, while discarding the global order of the pairs. Unigram and bigram features can be concatenated into one larger vector.

“not good” can then receive its own weight instead of having only the independent contributions of “not” and “good.” This helps with local negation; it does not fully solve “not really very good,” sarcasm, or discourse such as “bad acting, but I loved it.”

### Preprocessing decisions

| Choice | Potential benefit | What to watch |
|---|---|---|
| Split punctuation | Share “great” across “great!” and “great.” | Punctuation can itself carry sentiment |
| Lowercase | Share “Good” and “good” | Loses case information, including emphasis |
| Remove stopwords | Reduce common, less informative features | Removing “not” or “never” can destroy sentiment |
| Stem words | Share related forms such as singular/plural | May merge distinctions useful for a task |
| Drop unknown tokens | Keep a fixed vocabulary | Unknown words contribute nothing |
| Map to UNK | Represent unknown-token occurrence | Requires an explicitly defined UNK feature |

Build the vocabulary using training data, then keep the mapping fixed for development/test prediction. Unknown tokens do not automatically become similar known words: **goodx is not good** under exact matching.

TF-IDF is supporting lecture content: a common definition is **TF-IDF(word, document) = count(word, document) × log(N / df(word))**, where df counts documents containing the word. It downweights words found in many documents. Smoothing, normalization, and log bases vary; use the problem's definition. Weighting existing features does not by itself change their number.

### Checkpoint 2 — Build both vectors

Vocabulary **[good, bad, not, movie]**; text **“not good good movie”**. Give its count vector and indicator vector. Would “good movie not good” differ?

<details><summary>Reveal answer</summary>

Count: **[2,0,1,1]**. Indicator: **[1,0,1,1]**. The reordered text has the same vector under either unigram representation. The bigram representations can differ.

</details>

### Checkpoint 3 — Useful features (Course exercise 1.0)

For “the movie was good” (+) and “the movie was bad” (−), which two word indicators are most helpful? Why are “movie” and “was” unhelpful for distinguishing these examples?

<details><summary>Reveal answer</summary>

**good** and **bad** distinguish the labels. “movie” and “was” occur in both examples, so they supply no distinction here. With a suitable threshold/bias, either distinguishing word alone can suffice; this observation is about these examples, not every sentiment dataset.

</details>

### Checkpoint 4 — Negation parity (Course exercise 1.0)

The labels are “good” (+), “not good” (−), “not not good” (+). Choose the best extra feature: contains “not”; count of “not”; or indicator that the count of “not” is odd. Explain.

<details><summary>Reveal answer</summary>

**Odd-count indicator**, with values 0,1,0. Presence merges the last two examples. The raw counts 0,1,2 preserve more information, but a linear threshold on that single varying coordinate cannot put both endpoints in the positive class and the middle in the negative class. The parity feature expresses the needed interaction. This toy pattern is not a universal rule for English negation.

</details>

## 3. Linear scores, decisions, and bias

The score is **z = Σⱼ wⱼ fⱼ(x)**. Each active feature contributes its value times its learned weight. Positive weights favor the positive class; negative weights favor the negative class. An inactive feature contributes zero.

Example: vocabulary [good,bad,not,movie], w=[2,−3,−2,0.5]. For “not good movie,” f=[1,0,1,1], so z=2−2+0.5=**0.5**, hence positive under our default rule. This is an arithmetic result; it also shows that plausible individual weights can still give a linguistically poor prediction.

The decision boundary **wᵀf(x)=0** is perpendicular to w for nonzero w. Without an intercept it passes through the origin. An intercept b gives z=wᵀf(x)+b. The lecture folds it into the vectors by appending a constant **1** to f(x) and b to w. A bias is a trainable parameter; the appended 1 is constant.

### Checkpoint 5 — Score and tie

With the weights above, score “good bad movie.” Then classify a score of exactly zero under (a) z>0 and (b) z≥0.

<details><summary>Reveal answer</summary>

z=2−3+0.5=**−0.5**, hence negative. A zero score is (a) negative, (b) positive. Always write down the rule before tracing an algorithm.

</details>

### Checkpoint 6 — Bias

If f=[2,0] and w=[1,−1], how do you represent an intercept b=−3 using one enlarged dot product? What is the score?

<details><summary>Reveal answer</summary>

Use f′=[2,0,1] and w′=[1,−1,−3]. Their dot product is **−1**. The extra coordinate allows a boundary that need not pass through the original feature-space origin.

</details>

## 4. Perceptron: learn from prediction mistakes

For each example, predict with the **current** weights. If correct, do nothing. If wrong:

**w ← w + α y f(x)**, using y∈{−1,+1}.

For a missed positive, add features; for a missed negative, subtract features. With nonzero f, the change to the score of that same example is **Δz=α y ‖f(x)‖²**, so the update moves its score in the desired direction. One update need not classify it correctly yet.

### Worked trace (Course exercise 1.1, first question)

Vocabulary [good,bad,not]; examples in order: **good (+), not good (−), bad (−)**. Start w=[0,0,0], α=1; positive iff z>0.

| Epoch | Example | Score before update | Prediction | Action | Weights afterward |
|---|---|---:|---|---|---|
| 1 | good (+) | 0 | − | Add [1,0,0] | [1,0,0] |
| 1 | not good (−) | 1 | + | Subtract [1,0,1] | [0,0,−1] |
| 1 | bad (−) | 0 | − | None | [0,0,−1] |
| 2 | good (+) | 0 | − | Add [1,0,0] | [1,0,−1] |
| 2 | not good (−) | 0 | − | None | [1,0,−1] |
| 2 | bad (−) | 0 | − | None | [1,0,−1] |

A further pass has no mistakes. **Final weights: [1,0,−1].** “bad” has zero weight here because the negative tie rule already classifies it correctly. A zero weight does not prove that a word is linguistically neutral.

### Checkpoint 7 — Trace it yourself (Course exercise 1.1)

Same vocabulary, initialization, step size, and tie rule. Trace one epoch of **good (+), bad (−), not good (−), not bad (+)**. Show each score and final weights.

<details><summary>Reveal answer</summary>

1. good: score 0 → wrong negative → w=[1,0,0].
2. bad: score 0 → correct negative → w=[1,0,0].
3. not good: score 1 → wrong positive → w=[0,0,−1].
4. not bad: score −1 → wrong negative → w=[0,1,0].

Final **[0,1,0]**. A final vector can misclassify examples visited earlier; an epoch does not guarantee a solution.

</details>

### Why the negation example cannot be separated

For good (+), bad (−), not good (−), not bad (+), the strict-positive rule requires:

- w_good > 0, w_bad ≤ 0;
- w_good + w_not ≤ 0, so w_not ≤ −w_good < 0;
- w_bad + w_not > 0, so w_not > −w_bad ≥ 0.

The last two requirements contradict each other. Adding a bias does not fix this example either: the feature vectors for good + not bad sum to those for bad + not good, so two strictly positive scores cannot have the same sum as two nonpositive scores.

**The issue is the linear classifier's capacity with these features.** Logistic regression with these same features and a 0.5 threshold also has a linear decision boundary. More epochs cannot produce a perfect linear separator where none exists.

Two different remedies: add interaction features, such as bigrams, or use a sufficiently expressive nonlinear classifier on the existing features. For [good,bad,not,not_good,not_bad], weights **[1,−1,0,−2,2]** classify all four correctly. Entire-sentence indicators can memorize these examples, but offer weak sharing with new sentences.

### Checkpoint 8 — Convergence versus generalization

Is perceptron guaranteed to converge on every dataset? If training becomes perfect, does that guarantee good test accuracy?

<details><summary>Reveal answer</summary>

No. The standard convergence theorem assumes linear separability with a positive margin and standard update conditions. Nonseparable data need not converge. Perfect training accuracy also does not guarantee performance on unseen examples; the model can overfit or face a different data distribution. A single epoch with mistakes proves neither nonseparability nor eventual failure.

</details>

## 5. Perceptron as loss minimization

Define signed margin **m=y z=y wᵀf(x)**. Positive margins mean correct classification away from ties; negative margins mean incorrect classification. The perceptron loss is:

**ℓ_perc = max(0,−m).**

For m<0, its gradient with respect to w is **−y f(x)**. For m>0, it is zero. Subtracting α times that gradient gives the mistake update **w←w+α y f(x)**.

At m=0 the loss is nondifferentiable: choose a valid subgradient consistent with the algorithm's tie policy. With our strict-positive rule, a positive example at zero needs an update; a negative example at zero does not. Both choices are valid at the kink.

**Subtlety:** w=0 makes this loss zero on every example, even if the chosen tie rule misclassifies some. Zero perceptron loss alone does not guarantee zero classification errors at ties. Also, perceptron loss differs from the SVM hinge loss **max(0,1−m)**, which demands a positive margin of at least 1 before becoming zero.

### Checkpoint 9 — Loss and subgradient

Let y=−1, f=[1,2], w=[1,0], α=0.1. Compute z, m, perceptron loss, gradient, and updated weights.

<details><summary>Reveal answer</summary>

z=1; m=−1; loss=1; gradient=−y f=[1,2]. Updated w=[1,0]−0.1[1,2]=**[0.9,−0.2]**. Its new score is 0.5: improved for a negative example, but still misclassified.

</details>

## 6. Logistic regression: probabilities and training

### Model and prediction

**p=P(y=+1|x)=σ(z)=1/(1+exp(−z)).** The negative-class probability is **1−p=σ(−z)**. This is a discriminative model: it models P(y|x), rather than the joint P(x,y).

| z | p, approximately |
|---:|---:|
| −2 | 0.1192 |
| −1 | 0.2689 |
| 0 | 0.5000 |
| 1 | 0.7311 |
| 2 | 0.8808 |

Since sigmoid is increasing, **p>0.5 iff z>0**. The probability is nonlinear in z, but the classification boundary at threshold 0.5 is still linear in the features. Model probabilities are estimates, not guarantees of correctness or calibration.

### Objective: likelihood → log likelihood → loss

Assuming the usual independent labeled-example model, maximize **∏ᵢ P(yᵢ|xᵢ)**. Taking the logarithm preserves the maximizing weights and turns the product into **Σᵢ log P(yᵢ|xᵢ)**. Negating gives the objective to minimize: **L(w)=Σᵢ ℓᵢ(w)**, with:

**ℓ=−log P(gold label|x)=log(1+exp(−y z)).**

Equivalently, with t∈{0,1}, binary cross-entropy is:

**ℓ=−t log p−(1−t) log(1−p).**

Use natural logs unless told otherwise. A confidently wrong example has a large loss. A correctly classified example still has positive loss at finite z. Do not use the probability of the predicted class when the loss requires the probability of the gold class.

### Gradient and update

Using t∈{0,1}:

**∇wℓ=(p−t)f(x)** and **w←w+α(t−p)f(x)**.

Using y∈{−1,+1}, the equivalent update is:

**w←w+α y σ(−y z)f(x)**.

Thus a positive example adds **α(1−p)f**; a negative example subtracts **α p f**. In the negative case this is also **α(1−P(y=−1|x))f** subtracted. Do not put y=−1 into the formula (t−p).

### Where the gradient comes from

σ′(z)=p(1−p), and differentiating cross-entropy gives **dℓ/dz=p−t**. Since **∂z/∂wⱼ=fⱼ**, the chain rule gives **∂ℓ/∂wⱼ=(p−t)fⱼ**. Every coordinate shares the same prediction error but is scaled by its own feature value.

### Worked update

Take f=[2,1], w=[0,0], t=1, α=0.2. Then z=0, p=0.5, loss=−ln(0.5)=**0.6931**, gradient=(0.5−1)[2,1]=[−1,−0.5]. New w=**[0.2,0.1]**. The new score for this example is 0.5, giving p≈**0.6225**. The positive example's probability increased.

### Checkpoint 10 — A negative example

Let f=[1,2], w=[0,0], t=0, α=0.1. Compute p, loss, gradient, and new weights. What happens to the score?

<details><summary>Reveal answer</summary>

p=0.5; loss=0.6931; gradient=[0.5,1]; new w=**[−0.05,−0.1]**. New z=−0.25, so p≈0.4378. The negative-class probability increases to about 0.5622.

</details>

### Checkpoint 11 — Correct but still learning

A positive example has p=0.8. Does perceptron update? Does logistic regression update? With f=[1,1] and α=0.5, what is the logistic weight change?

<details><summary>Reveal answer</summary>

Perceptron makes no update because the prediction is correct. Logistic regression adds **0.5(1−0.8)[1,1]=[0.1,0.1]**. Correct classification does not make cross-entropy zero.

</details>

### Checkpoint 12 — Compare updates (Course exercise 1.2)

When are logistic updates numerically closest to perceptron updates: near p=0.5 or near p=0 / p=1? Explain for a positive gold label, then a negative gold label.

<details><summary>Reveal answer</summary>

Near the extremes. For positive gold, p≈0 gives nearly +αf, matching a perceptron mistake update; p≈1 gives nearly zero, matching no update. For negative gold, p≈1 gives nearly −αf; p≈0 gives nearly zero. Near 0.5 the logistic update is about a half-sized feature update. At finite scores sigmoid is strictly between 0 and 1, so these are limiting comparisons.

</details>

### Perceptron versus logistic regression

| Property | Perceptron | Logistic regression |
|---|---|---|
| Basic score | wᵀf(x) | wᵀf(x) |
| Boundary | Linear | Linear at the usual threshold |
| Probability model | No inherent probability | Sigmoid gives class probabilities |
| Loss | max(0,−yz) | log(1+exp(−yz)) |
| Update behavior | Only prediction mistakes | Generally every example with nonzero features at finite scores |
| Confidently correct | No update | Small update |
| Confidently wrong | Full mistake update | Approaches full mistake update |

Both losses are convex in w for fixed features; logistic loss is smooth. Convexity does not guarantee that arbitrary learning rates converge. The methods do not necessarily learn identical weights or decision boundaries, even on separable data. For separable data, unregularized logistic loss can approach its infimum while weight magnitudes grow without a finite minimizer.

## 7. Optimization and generalization

**Gradient descent:** w←w−α∇L(w). A gradient points toward locally increasing loss, so subtract it to minimize. For log-likelihood maximization, gradient ascent uses a plus sign instead.

| Method | Gradient computed from |
|---|---|
| Full-batch gradient descent | Entire training set |
| Stochastic gradient descent (SGD) | One sampled example per update |
| Minibatch gradient descent | A subset of examples per update |

Full-batch gradients can be expensive. Single-example updates are cheaper but noisy: one update can increase the loss on other examples or even the total training loss. Shuffling between epochs helps avoid a fixed order effect. Using a sum versus an average of losses rescales the gradient, so keep that choice consistent with the step size.

### A learning-rate example you should be able to calculate

For **L(w)=w²**, gradient=2w, so **w_next=(1−2α)w**. Start w=−1.

| α | First few values after initialization | Behavior |
|---:|---|---|
| 0.1 | −1 → −0.8 → −0.64 | Gradual convergence |
| 0.5 | −1 → 0 → 0 | Reaches minimum in one update |
| 0.75 | −1 → 0.5 → −0.25 | Oscillates with shrinking magnitude |
| 1 | −1 → 1 → −1 | Oscillates without convergence |
| 1.1 | −1 → 1.2 → −1.44 | Diverges |

For this particular quadratic and nonzero start, convergence requires **|1−2α|<1**, hence **0<α<1**. This range is not a universal learning-rate rule. A smaller rate is not always better: too small can make progress impractically slow.

### Development data and regularization

Learn parameters on training data. Use development/validation data to choose features, rates, epochs, and stopping points. Reserve test data for final evaluation. Falling training loss with worsening development performance suggests overfitting.

Early stopping can limit overfitting. Explicit L2 regularization, as an optional extension, minimizes **L(w)+(λ/2)‖w‖²**, adding λw to the gradient. The lecture emphasizes step size and mentions early stopping; it does not require detailed adaptive-optimizer derivations here.

The lecture also introduces Newton's method and adaptive methods. Newton uses curvature through a Hessian; a dense Hessian needs quadratic storage in parameter count and solving the Newton step adds computational cost. AdaGrad and Adam adapt coordinatewise updates using gradient statistics. They do not generally compute an exact inverse Hessian. Adam details return later with neural networks.

### Checkpoint 13 — Learning rate

For L(w)=w², w=−2, α=0.25, compute the next two weights. Does a negative gradient mean you should decrease w?

<details><summary>Reveal answer</summary>

First gradient=−4: w_next=−2−0.25(−4)=**−1**. Next gradient=−2: new w=**−0.5**. A negative gradient means subtracting it increases w, moving toward the minimum here.

</details>

### Checkpoint 14 — Diagnose training

Training accuracy rises while development accuracy falls. Would simply training longer be a good response? Name two things to investigate.

<details><summary>Reveal answer</summary>

This suggests overfitting; longer training may worsen generalization. Consider early stopping, simpler/less sparse features, appropriate regularization, more representative labeled data, and validation-based hyperparameter selection. Also inspect whether the development distribution differs from training. Do not tune against test labels.

</details>

## 8. Connect the guide to Assignment 1

The local `a1-distrib/models.py` provides concrete versions of these ideas:

- **UnigramFeatureExtractor:** sets each observed token's feature value to 1, so it uses presence, not counts.
- **BigramFeatureExtractor:** uses adjacent token pairs and presence values. This extractor alone does not concatenate unigrams.
- **BetterFeatureExtractor:** lowercases and removes stopwords while preserving no, nor, not, and never. Its actual benefit requires development evaluation; the name is not evidence of an improvement.
- **Sparse features:** counters map feature indices to nonzero values. The indexer connects words/pairs to weight coordinates.
- **Prediction:** freezes the vocabulary; unknown features are ignored. No constant bias feature is explicitly added by these extractors.
- **Perceptron:** labels and predictions are 0/1; the code updates by α(label−prediction)f on mistakes. That is equivalent to adding/subtracting αf.
- **Logistic regression:** uses α(label−probability)f, matching t−p. Prediction uses p>0.5, so ties are negative.
- **Training loop:** builds the vocabulary, initializes zero weights, and shuffles training examples each epoch. Saved settings are implementation choices, not values to memorize.

These are observations from reading the implementation, not results of a new training run.

### Checkpoint 15 — Read the representation correctly

Why would “good good good” have the same score as “good” in the local unigram model, assuming no other tokens or features? Would the same be true for count features?

<details><summary>Reveal answer</summary>

The local extractor sets the good feature to 1 in both cases. With counts, its values would be 3 versus 1, so the scores would be 3w_good versus w_good, absent a bias or other features.

</details>

## 9. Mixed practice: try without notes

For calculations, show **features → score → prediction/probability → loss or update**. Always state vocabulary order, gold-label encoding, tie rule, learning rate, and whether bias is present.

### M1. Unknown words and a different tie rule (2023 Q2a–b)

Vocabulary [good,great,not]. Inputs in order: **goodx (+), great (+), not greatq (−), not good (−)**. Drop unknown words. Start w=0, α=1, and for this question only predict positive when **z≥0**. Give all four vectors and the final weights after one epoch.

<details><summary>Hint</summary>

An unknown word has no coordinate. A zero vector can still produce a prediction, and this problem classifies zero as positive.

</details>

<details><summary>Reveal solution</summary>

Vectors: **[0,0,0], [0,1,0], [0,0,1], [1,0,1]**. First two scores are zero and correctly positive. Third score is zero, incorrectly positive, so subtract [0,0,1], giving **[0,0,−1]**. The fourth score is −1, correctly negative. Final **[0,0,−1]**.

Under the guide's usual strict-positive rule, the trace would differ. A nonzero-feature bias could also change the behavior, which is why this problem excludes it.

</details>

### M2. Perceptron in two dimensions (2022 Q3a)

Examples (f,y): ([1,0],+), ([1,1],+), ([0,1],−). Start w=0, α=1; positive iff z>0. Execute one epoch.

<details><summary>Reveal solution</summary>

The scores are **0,1,0**. Only the first example triggers an update. Final w=**[1,0]**.

</details>

### M3. True or false, with a reason (adapted from 2023 Q1(1))

1. Perceptron always converges.
2. Fixed-feature binary logistic regression has a convex negative log-likelihood objective.
3. SGD can train logistic regression.
4. Perceptron can be interpreted using a loss and SGD/subgradients.
5. On separable data, the two methods must find the same decision boundary.

<details><summary>Reveal solution</summary>

**False, true, true, true, false.** Nonseparable data defeats the universal convergence claim. Logistic loss is convex for fixed features. Its gradient supports SGD. Perceptron corresponds to its piecewise-linear loss with tie-aware subgradient selection. Agreement on training labels does not require the same separating boundary.

</details>

### M4. A complete logistic calculation

Vocabulary [good,bad,not]; count features. Text “good good not,” gold negative. w=[0.5,−1,−0.5], α=0.2. Compute f,z,p,loss,gradient, and new w.

<details><summary>Hint</summary>

The gold label is t=0, so loss=−ln(1−p) and gradient=p f.

</details>

<details><summary>Reveal solution</summary>

f=**[2,0,1]**; z=2(0.5)−0.5=**0.5**; p≈**0.622459**; loss≈**0.974077**. Gradient≈**[1.244919,0,0.622459]**. New w≈**[0.251016,−1,−0.624492]**. The inactive bad feature stays unchanged because no explicit regularization is used. New score on the same input is approximately −0.122459: it is now classified negative.

</details>

### M5. Distinguish two limitations

Case A: two texts with opposite labels have the same feature vector. Case B: they have different vectors, but the entire dataset is not linearly separable. Can switching to a nonlinear model help in each case?

<details><summary>Reveal solution</summary>

**A:** No deterministic classifier receiving only the identical vector can assign the two required different labels. Change the representation or provide more information. **B:** A sufficiently expressive nonlinear classifier may help because the information can be present but not separable by one linear boundary. These are different failure modes.

</details>

### M6. Propose an improvement (inspired by 2022 Q2c)

A new negative slang word is absent from a model's vocabulary. Propose two concrete ways to improve predictions on texts containing it. Explain what must change beyond collecting examples.

<details><summary>Reveal solution</summary>

One option: add the word to the training vocabulary, extend/rebuild its weight representation consistently, and train on representative labeled examples containing it. Another: add a feature for the word and assign an appropriate negative weight using domain knowledge, then validate it. Retraining with the token still discarded cannot teach a dedicated word weight. Character/subword features can also help share information, but belong to a later module and require changing the representation.

</details>

### M7. TF-IDF and feature-space size

In 100 documents, “the” occurs in all 100, while “excellent” occurs in 5. In one document both occur twice. Using count × ln(N/df), calculate their values. Does this weighting shrink the defined vocabulary?

<details><summary>Reveal solution</summary>

the: **2 ln(100/100)=0**. excellent: **2 ln(100/5)=2 ln20≈5.9915**. It reweights the defined coordinates; it does not by itself prune vocabulary entries. A separate feature-pruning step could do that.

</details>

### M8. Derive the negative-label update

For y=−1, write ℓ as a function of z, differentiate with respect to w, and state the SGD update.

<details><summary>Reveal solution</summary>

ℓ=−ln(1−σ(z))=**ln(1+exp(z))**. Its derivative with respect to z is σ(z)=p. Therefore ∇wℓ=**p f(x)** and **w←w−α p f(x)**. This lowers the score of a negative example with nonzero features.

</details>

## 10. Quick reference and common traps

| Task | Formula or rule |
|---|---|
| Linear score | z=wᵀf(x) |
| Default prediction | + if z>0, otherwise − |
| Perceptron mistake update | w←w+α y f, with y∈{−1,+1} |
| Perceptron loss | max(0,−yz), tie-aware subgradient at zero |
| Positive probability | p=1/(1+exp(−z)) |
| Negative probability | 1−p |
| Logistic loss | log(1+exp(−yz)) |
| Cross-entropy | −t ln p−(1−t)ln(1−p), t∈{0,1} |
| Logistic gradient | (p−t)f |
| Logistic update | w←w+α(t−p)f |
| General minimization | w←w−α∇L |

Avoid these mistakes:

1. Reordering vector coordinates mid-calculation.
2. Treating count and presence features as interchangeable.
3. Giving an unknown token the feature of a similarly spelled known word.
4. Forgetting the zero-score rule or adding an unstated bias.
5. Computing every example's score from the initial weights instead of the current ones.
6. Updating perceptron on correct predictions, or skipping logistic updates merely because predictions are correct.
7. Plugging ±1 labels into a formula requiring 0/1.
8. Using a plus gradient for loss minimization.
9. Claiming sigmoid gives a nonlinear classification boundary in the fixed feature space.
10. Treating low training loss as proof of good test performance.

### Mastery check

Mark each item **can explain / need a hint / need practice**. Revisit any weak item with a fresh example.

- [ ] I can build count, indicator, and bigram features in a stated vocabulary.
- [ ] I can explain tokenization, indexing, unknown words, and useful sentiment features.
- [ ] I can calculate a score and distinguish a bias weight from its constant feature.
- [ ] I can trace two perceptron epochs with either tie rule.
- [ ] I can explain why the four negation examples are not linearly separable.
- [ ] I can distinguish missing information from insufficient classifier capacity.
- [ ] I can connect signed margin, perceptron loss, subgradient, and update.
- [ ] I can calculate a logistic probability, gold-label loss, and gradient update.
- [ ] I can derive the negative-label logistic update.
- [ ] I can compare perceptron and logistic updates for correct and incorrect predictions.
- [ ] I can calculate the effect of a step size and explain SGD versus full-batch updates.
- [ ] I can distinguish training, development, and test roles.

## Sources and suggested follow-up

**Current scope:** [Fall 2026 midterm topics](../MIDTERM_TOPICS.pdf), pages 1–2. Module 1 covers the listed binary classification, bag-of-words, sentiment-feature, and optimization foundations. Other modules remain separate.

**Lectures:** 1.1.0 linear binary classification; 1.1.1 sentiment/basic features; 1.1.2 gradient descent; 1.2.0 perceptron; 1.2.1 perceptron loss; 1.3.0 logistic regression; 1.4.0 sentiment analysis; 1.4.1 optimization basics. All are the local caption transcripts beside this guide. Caption slips are resolved using equations, exercise feedback, and implementation context. Historical sentiment leaderboard claims are not treated as current rankings.

**Course exercise screenshots:** [features](%5BExercise%5D%201.0%20Sentiment%20Analysis%20and%20Features.png), [perceptron](%5BExercise%5D%201.1%20Perceptron%20Execution.png), [logistic update](%5BExercise%5D%201.2%20Logistic%20Regression%20Update.png). The guide includes the feature-selection/parity questions, both perceptron traces, the nonlinear-model/feature remedies, and the logistic comparison.

**Past exams:** [2023 solutions](../../midterms/fa23-cs388-midterm-solns.pdf), Q1(1), Q2(a–b); [2022 solutions](../../midterms/fa22-cs388-midterm-solns.pdf), Q1(3–4), Q2(c), Q3(a). Additional practice: explain when lowercasing, stemming, stopword removal, and punctuation tokenization can reduce vocabulary; TF-IDF alone changes values rather than vocabulary. Vocabulary reduction depends on the actual corpus, not a universal guarantee. Old exams illustrate question styles and do not establish current exam administration rules.

**Assignment connection:** [local models.py](../../a1-distrib/models.py). Its label encoding, feature choices, and tie rule are documented above.

**Our starting point:** Checkpoint 2. Build the two vectors and explain your reasoning. We'll move to scores only after the representation is clear.
