# Module 7 study guide: POS tagging, HMMs, and constituency parsing

Prepared for AI 388's Fall 2026 midterm from all 10 Module 7 lecture caption transcripts, all six exercise screenshots, the current exam topic sheet, and relevant 2022/2023 midterm solutions. Created October 17, 2026.

**Use this two ways:** study the explanations and worked examples in order; during the exam, use the navigation below or the separate [exam quick reference](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/EXAM_QUICK_REFERENCE.md>). Answers are visible so this also works as a reference. Practice questions are collected at the end for retrieval practice.

## Navigation

- [Exam scope and priorities](#exam-scope-and-priorities)
- [1. POS tagging and ambiguity](#1-pos-tagging-and-ambiguity)
- [2. Tagging with classifiers and features](#2-tagging-with-classifiers-and-features)
- [3. HMM definition and probability calculations](#3-hmm-definition-and-probability-calculations)
- [4. HMM estimation and smoothing](#4-hmm-estimation-and-smoothing)
- [5. Viterbi decoding](#5-viterbi-decoding)
- [6. Constituency structure and ambiguity](#6-constituency-structure-and-ambiguity)
- [7. CFGs and PCFGs](#7-cfgs-and-pcfgs)
- [8. Binarization and Markovization](#8-binarization-and-markovization)
- [9. CKY parsing](#9-cky-parsing)
- [10. Past-exam problems and solutions](#10-past-exam-problems-and-solutions)
- [11. Practice and answer key](#11-practice-and-answer-key)
- [12. Common traps and mastery checklist](#12-common-traps-and-mastery-checklist)
- [Sources and coverage audit](#sources-and-coverage-audit)

## Exam scope and priorities

The current topic sheet explicitly includes POS ambiguity; HMM definition, count-and-normalize estimation, and Viterbi; constituency trees and ambiguity; PCFG definition and estimation; binarization, Markovization, and CKY. It says tagging/parsing, **especially Viterbi**, are less emphasized than in older exams. This changes how much study time to allocate, not whether these topics are in scope.

Dependency parsing is explicitly excluded. CRFs, forward-backward, and NER/BIO evaluation occur in older materials but are not named in the current topic list or developed in this module's supplied lectures. They are not core targets for this guide. Detailed POS definitions and encyclopedic grammar knowledge are not required; necessary information will be supplied. Focus on reasoning with a given tag set or grammar.

A useful study order is **ambiguity → features → HMM factors/counts → Viterbi → trees/PCFGs → grammar refinement → CKY**. First master the probability directions, normalization denominators, STOP factors, span indices, and what each dynamic-programming cell means. Those explain most calculations.

For partial credit, show the factors, candidates, or rules you used. The topic sheet asks for concise explanations rather than essays.

## 1. POS tagging and ambiguity

**Part-of-speech tagging** maps a word sequence x = (x₁,…,xₙ) to an equally long tag sequence y = (y₁,…,yₙ). Each tag identifies a word's grammatical role in that occurrence. A word type can have several possible tags; context selects the intended one.

| Example | Possible distinction |
|---|---|
| record | A noun in “a record”; a verb in “record the meeting.” |
| gains | A noun in “small gains”; a verb in “she gains confidence.” |
| make | A verb in “I make dinner”; a noun in “what make is the car?” |
| up | A preposition in “hike up a mountain”; a particle in “made up the story.” |
| sold | Past tense in “I sold shares”; participle in “recently sold shares.” |

Open classes include nouns, verbs, adjectives, and adverbs; new vocabulary readily joins them. Closed classes include determiners, pronouns, prepositions, conjunctions, auxiliaries, and modals. These categories help interpret examples; memorizing a full tag inventory is not the goal.

### The lecture's “Fed raises interest rates” ambiguity

For the intended financial reading:

| Word | Intended role | Other lexical possibilities |
|---|---|---|
| Fed | Proper noun (the Federal Reserve) | Past-tense verb or participle: “fed the dog,” “had fed.” |
| raises | Verb | Plural noun: “pay raises.” |
| interest | Noun | Verb: “these topics interest me.” |
| rates | Plural noun | Verb: “she rates movies.” |

“Interest rates” is a noun–noun compound. A noun modifying another noun does not automatically become an adjective.

The lecture gives 3 × 2 × 3 × 2 = **36 candidate tag paths**, using its more detailed tag inventory and ignoring fixed tags for the number/percent. In general, K choices at every position give Kⁿ paths; varying choices give ∏ᵢKᵢ.

An alternative structural interpretation makes **Fed raises** a noun phrase and **interest** its verb, as if raises interested animate “rates.” It is semantically strange but illustrates that local syntax alone does not capture world knowledge. Lexical possibility, grammatical compatibility, and plausible meaning are different tests.

### Course exercise 7.0: “I make small gains on my investments”

- Possible tags for **gains**, without context: noun and verb. In this sentence: noun.
- Possible tags for **make**, without context: noun and verb. In this sentence: verb.
- A more linguistically appropriate sequence is **Pronoun Verb Adjective Noun Preposition Determiner Noun**, treating “my” as a possessive determiner (the exact possessive category depends on the tag set).

**Screenshot caveat:** the exercise marks “N V ADJ N Prep N N” correct, although its own available categories include pronoun. That sequence captures the intended content-word choices but labels “I” and “my” too coarsely. Do not learn that “I” is ordinarily a noun or that “my” is ordinarily a common noun. Follow the category definitions supplied on the exam.

## 2. Tagging with classifiers and features

An independent multiclass classifier can predict at each position:

**P(yᵢ=t | x,i) = exp(wₜᵀf(x,i)) / Σᵤ exp(wᵤᵀf(x,i)).**

Training treats each labeled token position as an example. Independence here refers to output decisions given the input; the classifier may still inspect surrounding input words.

### Why sentence bag-of-words is inadequate

Using the same sentence-level bag-of-words vector at every position, with no information identifying the target word, gives the same score distribution at every position. A bigger classifier cannot recover which word you meant to tag from identical inputs.

Use features anchored to the target position:

| Feature template | At “interest” in “Fed raises interest rates” |
|---|---|
| CurrentWord | interest |
| PreviousWord | raises |
| NextWord | rates |
| Character/suffix/prefix | e.g., suffix=est |
| Shape/case/digits/length | lowercase, no digit, length 8 |
| SeenInTraining | whether this word appeared in training |

In joint-feature notation, a coordinate can be **1[xᵢ = interest AND candidate tag = NN]**. Candidate-tag conditioning is normal: it lets each tag receive a different score. It does not reveal the gold tag.

**Exercise 7.1:** CurrentWord=interest is useful. Contains(interest) does not identify the tagging position. ThirdWord=interest hardcodes an absolute location rather than a reusable relation to the target. “CurrentWord=Noun” is not an ordinary observed word-identity feature; if it means the unknown gold POS, it leaks the answer. Prefixes/suffixes, neighboring words, seen/unseen status, length, and digits are all reasonable observable features.

### What independent predictions miss

Each individual label can look plausible while the sequence is poorly coordinated. HMM transitions explicitly model adjacent tag compatibility. A local classifier with good context features can still work well; structured output modeling is a separate design choice.

A neural window classifier can concatenate embeddings:

**h = activation(W[e(xᵢ₋₁); e(xᵢ); e(xᵢ₊₁)] + b)**, followed by a tag classifier.

Three d-dimensional embeddings give a **3d-dimensional** input. Concatenation preserves the relative slots; averaging them loses that distinction. Character n-grams help generalize to unfamiliar words.

Connection to earlier modules: A1's feature engineering becomes token-relative; A2's averaging limitation matters when word order distinguishes grammatical roles; A3's contextual representations offer richer input information, while joint decoding remains its own choice.

The lecture's most-frequent-tag baseline performs well because many frequent words are unambiguous. Still, 90% token accuracy can mean roughly two mistakes in a 20-word sentence. If errors were independent with identical accuracy 0.9, whole-sentence accuracy would be 0.9²⁰ ≈ 0.122; independence is an illustrative assumption, not a factual claim about errors. Historical accuracy numbers in captions are context, not current rankings or memorization targets.

## 3. HMM definition and probability calculations

An **HMM is a generative sequence model of the joint distribution P(x,y)**. The tags are hidden at test time; words are observed. “Hidden” does not mean tags must be unobserved during training: this module estimates parameters from labeled sequences.

### Notation and parameter shapes

Let K be the number of ordinary tags and M the vocabulary size.

| Parameter | Definition | Shape / normalization |
|---|---|---|
| s(t) | P(y₁=t) | K entries, summing to 1 |
| A(u,t) | P(next tag=t | current tag=u) | K rows; K ordinary destinations plus STOP here |
| B(t,w) | P(word=w | tag=t) | K × M; each tag row sums to 1 |

**Convention:** transition rows are source tags, columns are destination tags. STOP has no word emission. A model with STOP normalizes each transition row across ordinary tags **and STOP**. A fixed-length HMM without STOP instead normalizes across ordinary tags only. Read the supplied table and use its convention consistently.

### Generative story and assumptions

Sample y₁, emit x₁ from it, transition to y₂, emit x₂, continue, then stop.

- First-order tag assumption: P(yᵢ | y₁,…,yᵢ₋₁) = P(yᵢ | yᵢ₋₁).
- Emission assumption: given the relevant tags, each word depends only on its own tag in this basic model.
- Transition/emission parameters are shared across positions.

These are **conditional** independence assumptions. Earlier tags can influence later tags through intermediate states. Words are not unconditionally independent. Also, the generative transition table does not condition on words, even though inference about tags uses the observed sentence.

The model cannot directly encode that “San” strongly predicts “Francisco”: it sees a sequence of tags rather than a lexical dependency between these two words. “San York” can receive probability if its individual factors permit it.

### Joint probability, including STOP

**P(x,y) = s(y₁) × B(y₁,x₁) × ∏ᵢ₌₂ⁿ [A(yᵢ₋₁,yᵢ) B(yᵢ,xᵢ)] × A(yₙ,STOP).**

For n words, include one initial factor, n emissions, n−1 ordinary transitions, and one STOP transition: **2n+1 factors**. Omit STOP only when the model omits it.

In log space, multiply probabilities by **adding their logs**. log(0)=−∞. Use the supplied log base consistently. Bigger log scores are better: −4 beats −7.

### Course exercise 7.2: exact calculation

Initial: s(N)=1, s(V)=0.

| From / To | N | V | STOP |
|---|---:|---:|---:|
| N | 1/5 | 3/5 | 1/5 |
| V | 1/5 | 1/5 | 3/5 |

| Tag / Word | they | can | fish |
|---|---:|---:|---:|
| N | 1 | 0 | 0 |
| V | 0 | 1/2 | 1/2 |

For **they/N can/V fish/V**:

**P(x,y) = 1 × 1 × (3/5) × (1/2) × (1/5) × (1/2) × (3/5) = 9/500 = 0.018.**

No other tag path has positive probability: “they” must be N; “can” and “fish” must be V. Smoothing transitions has not smoothed the emission zeros.

This is a joint probability, not generally P(y|x). Here only one tag sequence survives, so P(x)=9/500 and its conditional probability P(y|x)=1. The distinction matters even though decoding selects the same path.

## 4. HMM estimation and smoothing

For fully labeled training data, maximize **Σsentences log P(x,y)**. Because each multinomial distribution separates, estimation is **count and normalize**, without needing gradient descent.

| Estimate | Numerator | Denominator |
|---|---|---|
| s(t) | Sentences starting with t | Number of sentences |
| A(u,t) | Observed u→t events | All outgoing events from u, including STOP |
| B(t,w) | Occurrences of w labeled t | All tokens labeled t |

If each occurrence of an ordinary tag has one successor (ordinary tag or STOP), its outgoing-event count equals its token count. Never count transitions across a sentence boundary. STOP does not emit a word.

### Lecture example: two training sentences

1. they/N can/V STOP
2. they/N fish/V STOP

Counts: starts N=2, V=0; N→V=2; V→STOP=2; emissions N: they=2; emissions V: can=1, fish=1.

Unsmoothed estimates:

- s=(1,0).
- A(N,·)=(0,1,0), A(V,·)=(0,0,1), with columns N,V,STOP.
- B(N,·)=(1,0,0), B(V,·)=(0,1/2,1/2), with columns they,can,fish.

Unsmoothed **they/N can/V fish/V** has probability zero because V→V was never observed.

### Add-α smoothing

For a conditional distribution with L legal outcomes:

**Pα(outcome | context) = [count(context,outcome)+α] / [count(context)+αL].**

Adding 1 to every legal transition count gives N row (1,3,1)/5 and V row (1,1,3)/5, exactly the transition table in Exercise 7.2. Here L=3 because STOP is an outcome. Do not add a STOP emission column.

Smoothing prevents unseen-but-allowed events from getting zero probability and reduces extreme estimates. It does not make impossible structural events legal. Smooth only the distributions the question specifies. A context with zero observations has undefined unsmoothed MLE proportions; smoothing or an explicit fallback is needed.

**Direction trap:** B(V,can)=1/2 is P(can|V), not P(V|can). Normalization is over words for a fixed tag. Bayes' rule would also require tag priors to reverse the conditioning.

Unknown words need an explicit strategy, such as mapping to UNK and assigning it emission mass. Context cannot rescue a sentence if every candidate path still has a zero emission at that word.

## 5. Viterbi decoding

### What we optimize

**ŷ = argmaxᵧ P(y|x) = argmaxᵧ P(x,y) = argmaxᵧ log P(x,y).**

The first equality follows because P(y|x)=P(x,y)/P(x), and P(x) is constant for the fixed observed sentence. This preserves the maximizing sequence, not the numerical probability value. Assume the sentence has positive model probability.

### Cell meaning, recurrence, and backtracking

Let **vᵢ(t)** be the best log score for a prefix ending at position i with tag t, having emitted x₁,…,xᵢ. It has not yet included STOP.

```text
Initialize:
  v₁(t) = log s(t) + log B(t,x₁)

For i = 2,...,n:
  vᵢ(t) = log B(t,xᵢ) + max_u [vᵢ₋₁(u) + log A(u,t)]
  bpᵢ(t) = an argmax_u of the bracketed expression

Finish:
  best_score = max_t [vₙ(t) + log A(t,STOP)]
  yₙ = an argmax_t of that expression

Recover:
  yᵢ₋₁ = bpᵢ(yᵢ), working backward to i=2
```

For a probability-space calculation, replace additions by multiplications, keep max, and use 0 for impossible states. Do not multiply numbers that are already logs.

**Why it is exact:** once the position and final tag of a prefix are fixed, future factors cannot distinguish how that prefix got there. Keeping its best path is sufficient. Keep one best prefix **per ending tag**, not one overall prefix. A globally weaker prefix can lead to a better completion via a different transition.

### Course exercise 7.3: full chart

These supplied integer log scores only approximately normalize; treat them as the given additive scores. They are a separate model from Exercise 7.2.

```text
Initial:       N=-1, V=-1
Transitions:   N→N=-2, N→V=-1, N→STOP=-1
               V→N=-1, V→V=-1, V→STOP=-2
Emissions:     N: they=-1, can=-3, fish=-1
               V: they=-3, can=-1, fish=-1
```

For “they can fish,” initialization gives N=−2 and V=−4.

| Cell | Candidate via previous N | Candidate via previous V | Best; predecessor |
|---|---:|---:|---|
| can/N | −2−2−3 = −7 | −4−1−3 = −8 | −7; N |
| can/V | −2−1−1 = −4 | −4−1−1 = −6 | −4; N |
| fish/N | −7−2−1 = −10 | −4−1−1 = −6 | −6; V |
| fish/V | −7−1−1 = −9 | −4−1−1 = −6 | −6; V |

Final chart:

| Ending tag | they | can | fish | After STOP |
|---|---:|---:|---:|---:|
| N | −2 | −7 | −6 | **−7** |
| V | −4 | −4 | −6 | −8 |

Best final tag N. Backtrack fish/N ← can/V ← they/N. Answer: **N V N, score −7**. Direct factor check: −1−1−1−1−1−1−1 = −7.

The two final token cells tie, but STOP breaks the tie. Choosing each column's largest entry independently would not be a valid substitute for backtracking.

### Complexity and variants

With n words and K tags, there are nK cells and K predecessor candidates per cell: **O(nK²) time**. The full score/backpointer chart uses **O(nK) space**; scores alone can use O(K) rolling space if you do not need the path. Viterbi implicitly searches Kⁿ possible tag paths without enumerating them.

A **trigram HMM** conditions on two preceding tags. Represent its state as a pair (previous,current), and allow only overlapping transitions (a,b)→(b,c). An optimized recurrence costs O(nK³), not the O(nK⁴) cost of densely connecting all K² pair states. Emissions may depend only on the current tag in a common formulation; the supplied lecture also discusses conditioning emissions on the pair state. Use the definition in the question.

Viterbi is a max algorithm for the best sequence. Summing over all sequences instead computes the sentence probability; independently maximizing tag marginals targets a different objective. You do not need the full forward-backward derivation for this guide's core scope.

## 6. Constituency structure and ambiguity

A constituency parse groups words into nested phrases. **Leaves** are words; **preterminals** are POS tags directly above words; higher **nonterminals** include NP (noun phrase), VP (verb phrase), PP (prepositional phrase), and S (sentence). A node's **yield** is its left-to-right word sequence. In the ordinary trees used here, each constituent covers a contiguous span.

Example:

```text
(S
  (NP (PRP I))
  (VP (V like)
      (NP (J green) (N salsa))))
```

Every internal node specifies a production: S→NP VP, NP→PRP, VP→V NP, NP→J N. Preterminal nodes specify lexical productions such as V→like. Repeated occurrences count repeatedly during estimation.

Trees can have unary, binary, or higher-arity branches. Constituency is not inherently binary. A treebank is a collection of sentences annotated with trees.

### Attachment changes meaning

For “eat spaghetti with chopsticks,” simplified phrase structures are:

```text
Instrument reading:
(VP (VP eat spaghetti) (PP with chopsticks))

Noun-modifier reading:
(VP eat (NP spaghetti (PP with chopsticks)))
```

The PP modifies the eating event in the first and the spaghetti in the second. These are schematic phrase brackets, not complete POS trees. The second is strange in ordinary circumstances, but syntax can permit unlikely meanings.

For **“Ban on nude dancing on governor's desk”**, the final PP can attach to the ban (the ban/document is on the desk) or to the dancing (the dancing takes place there). POS tagging alone cannot distinguish these attachments: the same tags can support different trees.

Other lecture ambiguities:

- **plastic cup holder:** [[plastic cup] holder] versus [plastic [cup holder]]; a holder for plastic cups versus a plastic holder for cups.
- **picked up his hammer and saw:** coordination of two objects (“hammer” and noun “saw”) versus coordination involving the verb “saw.”
- **complained to the professor that they didn't understand:** the final clause can express the complaint or help describe the professor, depending on the analysis.

Substitution by a pronoun/pro-form, cleft constructions, and question-answer fragments are useful constituency diagnostics. For instance, replacing “the children” with “they” supports treating it as a unit. These are diagnostic tools rather than universal proofs. Co-indexation and traces in complex linguistic analyses are lecture context, not algorithms to master here.

## 7. CFGs and PCFGs

A **CFG** consists of nonterminals N, terminals Σ, a start/root symbol S, and production rules R. Nonterminals rewrite into ordered sequences of symbols. A valid parse must use the supplied productions and yield exactly the input words in order.

A **PCFG** adds probabilities to rules:

**Σβ P(A→β | A) = 1 for each left-hand side A.**

Normalize all expansions of the same parent together, including lexical/unary/binary rules when they share that parent. Do not normalize over all rules in the grammar.

For a tree τ:

**P(τ,x) = ∏nodes P(rule at node | its left-hand side).**

When τ includes its terminal yield, this is often written simply P(τ). Each occurrence of a rule contributes its probability. The probability of the sentence is **P(x)=Στ:yield(τ)=x P(τ,x)**, a sum over parses, whereas the best parse uses a max. For fixed x, maximizing the joint tree score also maximizes P(τ|x).

### Estimation from trees

**P(A→β | A) = count(A→β) / Σγ count(A→γ).**

If NP→DT NN occurs three times and NP→PRP four times, their estimates are 3/7 and 4/7. Count nodes, not sentences: a tree can contain several NP expansions. A rule never observed gets zero MLE probability unless smoothing or another modeling choice is specified.

### Course exercise 7.4: count parses and score them

Root symbol NP. Grammar:

```text
NP  → NP CC NP   [0.3]
NP  → NP PP      [0.3]
NP  → NNS        [0.4]
PP  → P NP       [1.0]
NNS → cats       [1.0]
CC  → and        [1.0]
P   → in         [1.0]
```

“cats and cats” has **one parse**, coordinating two NPs. Its probability is 0.3 × 0.4² = **0.048**.

“cats and cats in cats” has **two parses**:

```text
[NP [NP cats and cats] [PP in cats]]
[NP cats and [NP cats [PP in cats]]]
```

These show attachment scope; each “cats” expands through NP→NNS→cats and each “and” through CC→and. Both trees use NP→NP CC NP once, NP→NP PP once, and NP→NNS three times. Each has probability **0.3² × 0.4³ = 0.00576**. Thus P(sentence)=**0.01152** and each parse has conditional probability 1/2.

This reveals a basic PCFG limitation: if two trees use the same multiset of rules, they have the same probability. An unlexicalized grammar cannot prefer one simply because its meaning seems better.

## 8. Binarization and Markovization

### Binarization: change branching structure

CKY's binary recurrence combines two children at a time. Replace productions with more than two children by binary chains using artificial symbols.

A lossless, rule-specific transformation of a rule with probability p is:

```text
Original: VP → V NP PP   [p]

VP           → V @VP_NP_PP   [p]
@VP_NP_PP    → NP PP         [1]
```

The artificial symbol must encode enough context to force the intended continuation. The transformed fragment has probability p×1=p. Remove artificial nodes to recover the original structure. If symbols are safely shared through a full-history factorization, the chain rule supplies equivalent conditional probabilities; not every intermediate rule must then have probability 1.

**Lossy binarization** merges intermediate contexts, allowing additional combinations or changing probabilities. Replacing every artificial symbol with an ordinary VP can make unrelated expansions compete for probability mass and create unintended recursion. Binarization is lossless only when the transformation actually preserves the original derivations and probabilities.

Binarized trees may retain nonterminal unary rules such as NP→PRP. Consequently, this is not necessarily strict Chomsky normal form, which ordinarily permits binary nonterminal rules and lexical rules, with limited start-symbol exceptions.

### Vertical Markovization: remember ancestors

Annotate a category with its parent, for example **NP^S**, **NP^VP**, **NP^PP**. This lets subject-position NPs, verb-object NPs, and prepositional-object NPs have separate expansion distributions.

The model still uses CFG rules, but a label carries extra context. Annotate the training trees **before extracting and counting rules**. Annotating only a test tree does not train a refined grammar.

**Benefit:** distinguishes contexts wrongly pooled by the original PCFG. **Cost:** more categories/rules and fewer observations per refined distribution. More ancestor history can improve specificity while increasing sparsity.

Some notations count vertical order including the current node; others count remembered ancestors. “One parent annotation” is unambiguous; use the example and definition given in the question rather than assuming a universal numeric convention.

### Horizontal Markovization: remember siblings

During incremental binarization, keep only a limited history of previously generated siblings. For a schematic right-branching expansion A→B C D E:

```text
Full sibling history:
A       → B @A_B
@A_B    → C @A_BC
@A_BC   → D E

History limited to one previous sibling:
A       → B @A_B
@A_B    → C @A_C
@A_C    → D E
```

The h=1 labels forget B after C has been generated, enabling sharing with other expansions that have the same parent and latest sibling C. With h=0, even that sibling identity is forgotten. With h=2, the most recent two siblings are retained. These are illustrative labels; particular binarization directions/head conventions may differ.

Reducing remembered history is an approximation, and probabilities must be estimated for the resulting transformed grammar. Do not transfer old probabilities blindly after merging contexts. The lecture notes practical schemes may additionally preserve a head tag.

| Operation | Context/structure affected | Main purpose |
|---|---|---|
| Binarization | Number of children per production | Enable two-child dynamic programming |
| Vertical Markovization | Ancestors | Separate expansions by parent/history |
| Horizontal Markovization | Sibling history in artificial nodes | Control size, sharing, and sparsity |
| Lexicalization | Head word of each constituent | Model lexical preferences in attachment |

Lexicalization helps with preferences such as coordinating “dogs and cats” rather than “houses and cats.” It also greatly expands the model and requires additional estimation/search machinery. Know the motivation; detailed lexicalized-parser construction is not a core listed topic.

## 9. CKY parsing

### State and indexing

Use **fencepost spans [i,j)** with 0 ≤ i < j ≤ n. The words in the span are xᵢ,…,xⱼ₋₁; indices here start at zero, unlike the HMM section's one-based token indices.

**C[i,j,A] = best log probability of a subtree rooted at A covering exactly [i,j).**

There are n(n+1)/2 nonempty spans, and a score for each relevant category in each span. Store a best derivation per category, not just a single category for the span.

### Binary recurrence

```text
Initialize every cell to -∞.
Lexical base:
  C[i,i+1,A] = log P(A→xᵢ | A), for licensed lexical rules

For span length = 2,...,n:
  For each start i, let j=i+length:
    For each split k with i<k<j:
      For each binary rule A→B D:
        candidate = log P(A→B D | A) + C[i,k,B] + C[k,j,D]
        Keep the largest candidate for C[i,j,A].
        Record (rule,k) as its backpointer.

Finish at C[0,n,S] (or the supplied root category).
Backtrack to reconstruct the tree.
```

The child spans must be adjacent and cover the parent span exactly. Rule order matters: A→B D needs B on the left and D on the right. Fill short spans first so all children are already computed. If the root score is −∞, there is no parse under the grammar.

**Unary-rule caveat:** the lecture develops the binary recurrence and explicitly sets aside unary handling. If nonterminal unary rules remain, apply unary closure, propagating A→B on the same span until no score improves (or preprocess them correctly). For example, PRP→I alone cannot build NP unless NP→PRP is also applied. A single arbitrary-order pass need not handle a multi-step unary chain. The worked exercise below requires only binary and lexical rules.

### Course exercise 7.5: full chart for “child raises raises it”

```text
DT  → the       [1]
NN  → child     [1]
NNS → raises    [1]
VBZ → raises    [1]
PRP → it        [1]
S   → NP VP     [1]
NP  → DT NN     [1/2]
NP  → NN NNS    [1/2]
VP  → VBZ PRP   [1]
```

Use base-2 logs: log₂1=0, log₂(1/2)=−1.

| Span | Words | All finite scores |
|---|---|---|
| [0,1) | child | NN: 0 |
| [1,2) | raises | NNS: 0; VBZ: 0 |
| [2,3) | raises | NNS: 0; VBZ: 0 |
| [3,4) | it | PRP: 0 |
| [0,2) | child raises | NP: −1, from NN NNS, split 1 |
| [1,3) | raises raises | none |
| [2,4) | raises it | VP: 0, from VBZ PRP, split 3 |
| [0,3) | child raises raises | none |
| [1,4) | raises raises it | none |
| [0,4) | child raises raises it | S: −1, from NP VP, split 2 |

At the root, candidate = 0+(−1)+0=−1. The parse is:

```text
(S
  (NP (NN child) (NNS raises))
  (VP (VBZ raises) (PRP it)))
```

The first “raises” is a plural noun (“child raises,” interpreted as a compound); the second is a verb. Whether that sentence sounds natural does not override the supplied grammar. Best tree probability is **2⁻¹=1/2**. This grammar gives only one full parse here, so its posterior given this sentence is 1.

### Exercise 7.5: repair “the child raises raises”

Originally NP can cover “the child,” but VP→VBZ PRP cannot cover the last two words because “raises” has no PRP lexical production. Either addition below works **individually**:

- **VP→VBZ NNS**: last two words become a VP directly.
- **PRP→raises**: last word can supply PRP to the existing VP rule.

NP→NNS VBZ does not provide the missing VP. S→NP NP also fails because “raises raises” cannot form an NP under the original NP rules. These additions establish parseability; to maintain a PCFG, assign new probabilities and renormalize the affected parent distribution. Adding a second probability-1 rule to the same parent is not a normalized PCFG.

### Runtime and relation to Viterbi

For a rule-loop implementation, CKY costs **O(n³|R₂|)** for binary rules R₂ and **O(n²|N|)** chart space plus backpointers. With a dense inventory of all category triples, runtime is often expressed O(n³|N|³); with a fixed grammar, O(n³). Unary closure can add work.

| Property | Viterbi | CKY |
|---|---|---|
| Output | Best tag sequence | Best constituency tree |
| State | Position, ending tag | Start, end, root category |
| Candidate choice | Previous tag | Split point and production |
| Local term | Transition plus emission | Production probability |
| Recovery | Previous-tag backpointer | Rule/split backpointer |
| Exactness relies on | Markov property | Context-free subproblem structure |

Both keep the best solution **per state** and reuse it. Neither is ordinary beam search. For sentence probability instead of best parse, replace max by summation in probability space (or log-sum-exp in log space); that is a different objective.

## 10. Past-exam problems and solutions

Use these after the course exercises. The old papers demonstrate question styles, not current point weights or exam rules.

### 2022 Q1(2), Q1(8–10): HMM factors and viable paths

Q1(2): a transition is the distribution of the next tag given the preceding tag, without conditioning the transition table on observed words. The correct supplied choice is D.

For Q1(8–10), the tags are adjective J and noun N:

```text
Initial: J=2/3, N=1/3
Transitions:
  J→J=1/2, J→N=1/4, J→STOP=1/4
  N→J=0,   N→N=1/2, N→STOP=1/2
Emissions:
  J: big=1/2, red=1/4, convertible=1/4, car=0
  N: big=0,   red=0,   convertible=1/3, car=2/3
```

- **big/J car/N:** (2/3)(1/2)(1/4)(2/3)(1/2)=**1/36**.
- **big convertible car:** positive paths **J J N** and **J N N**, so **2**.
- **big car convertible:** only **J N N**, so **1**. The alternative J N J is killed by N→J=0 even though “convertible” has a J emission.

Distinguish **Kⁿ nominal assignments** from paths with **nonzero probability**. The question's “possible” refers to viable paths under the supplied model.

### 2023 Q1(19–20): Viterbi complexity

Runtime is **O(nK²)**. The number of full paths it searches implicitly is **Kⁿ** (the supplied solution expresses this as O(Kⁿ)). Dynamic programming reduces computation, not the number of assignments considered by the optimization problem.

### 2022 Q4: a grammar that overgeneralizes word order

Grammar:

```text
S→PRP VP [1]; VP→V NP [1]; NP→J N [1]
PRP→I [1]; V→like [1/2]; V→eat [1/2]
J→green [1/4]; J→red [1/4]; J→verde [1/4]; J→roja [1/4]
N→salsa [1]
```

“I like green salsa” has parse **(S (PRP I) (VP (V like) (NP (J green) (N salsa))))**, probability **(1/2)(1/4)=1/8**.

Adding **NP→N J** admits “I like salsa verde,” but also “I like salsa red,” the undesired English order. To distinguish the intended patterns, introduce separate categories:

```text
NP→NP_EN | NP_ES
NP_EN→J_EN N
NP_ES→N J_ES
J_EN→green | red
J_ES→verde | roja
N→salsa
```

Here “|” lists alternative productions; probabilities are omitted because this is a structural repair. The NP wrapper keeps the categories reachable from VP→V NP. Splitting N into language-specific versions is optional, since both have “salsa.” If probabilities are requested, normalize each parent after choosing or counting its expansions.

**Lesson:** broad categories share behavior that may need to remain distinct; category refinement can constrain combinations.

### 2023 Q4: extract a grammar, then annotate NP parents

The two supplied trees have these bracket structures, preserving their distinct lexical items:

```text
(S (NP (PRP I))
   (VP (VBP teach) (NP (NNS students))
       (PP (IN about) (NP (NNP NLP)))))

(S (NP (PRP he))
   (VP (VBP gives) (NP (NNS lessons))))
```

Extract each **distinct internal rule**, excluding lexical productions and without binarization:

```text
S  → NP VP
NP → PRP
NP → NNS
NP → NNP
VP → VBP NP PP
VP → VBP NP
PP → IN NP
```

For estimation, repeated rules would count repeatedly. The first tree has three NPs (I, students, NLP), and the second has two (he, lessons): **five NP expansions** in total. PRP occurs twice, NNS twice, and NNP once, giving PRP=2/5, NNS=2/5, NNP=1/5. Counting each listed rule once would incorrectly give 1/3 each. S→NP VP has count two, VP alternatives each count one, and PP→IN NP has count one.

Annotate **only the NP categories**, as the exam requests:

```text
S     → NP^S VP
NP^S  → PRP
NP^VP → NNS
NP^PP → NNP
VP    → VBP NP^VP PP
VP    → VBP NP^VP
PP    → IN NP^PP
```

Now pronoun-derived subject NPs cannot automatically serve as verb-object NPs: NP^VP has no PRP expansion.

**The “him teach I” erratum:** “him” was removed from the shown trees/lexicon during editing. With only the displayed lexical rules, it is not parseable even before refinement; the solutions explicitly accept that answer. If one assumes PRP→him, as the intended question did, the unrefined grammar permits a subject “him” and an object “I.” The refined grammar rejects it because object NP^VP cannot expand to PRP. State the lexical assumption rather than silently inventing a rule.

With exact parent annotation and extraction from the same trees, the refined grammar's language, after erasing annotations, is a **subset (possibly equal in general)** of the unrefined grammar's language. Every refined rule maps back to an old rule, but some old combinations lose compatible annotations. In this example the restriction removes combinations, so it is a proper subset. It does not guarantee that every surviving sentence is grammatical.

## 11. Practice and answer key

These are original checks unless explicitly attributed above. Cover the answers while working. A useful response format is **definition/factors → calculation → conclusion**.

### Questions

1. Why can the same word receive different tags? Give two contexts for “gains.”
2. A tagging classifier receives only the sentence's bag-of-words vector at every position. Why will adding another hidden layer not fix the missing target information?
3. In Exercise 7.2, what is P(can|V)? Is it the same quantity as P(V|can)?
4. A transition row has counts N=2, V=5, STOP=3. Give its MLE and add-one estimates.
5. A tag emits “fish” three times and “can” once. With vocabulary {fish,can,they}, give the MLE and add-one emission distributions.
6. For two training sentences a/N b/V and a/N a/N, estimate starts, transitions including STOP, and emissions without smoothing. What is P(a/N b/V)?
7. In Exercise 7.3, recompute the cell fish/N and explain why its predecessor is V.
8. Suppose the final Viterbi cells are N=−4 and V=−5, with STOP scores N→STOP=−3 and V→STOP=−1. Which final tag wins?
9. Is Viterbi's best joint score automatically a normalized conditional probability? Explain.
10. A treebank contains 8 NP nodes: 5 expand to DT NN and 3 to PRP. Give the rule probabilities. Does the number of sentences determine the denominator?
11. Under Exercise 7.4, score “cats in cats.” How many parses does it have?
12. Binarize A→B C D E with probability 0.2 losslessly using unique intermediate symbols. Give all probabilities.
13. How do vertical and horizontal Markovization differ? What tradeoff does remembering more context introduce?
14. For five words, how many nonempty CKY spans exist? How many splits can [0,5) try?
15. Could a finite CKY score for NP over [0,n) prove the sentence parseable if the required root is S?
16. If all trees for a sentence use the same rule counts, can a basic PCFG choose among them by meaning?

### Answers

1. POS describes an occurrence in context. “Small gains” uses a noun; “she gains confidence” uses a verb.
2. Every position still receives identical inputs. The representation must identify the current word/position or otherwise supply position-dependent information.
3. 1/2. No: emissions normalize over words given V; the reversed conditional is a different distribution.
4. MLE **(2/10,5/10,3/10)**; add-one **(3/13,6/13,4/13)**. STOP belongs in both denominators.
5. MLE fish=3/4, can=1/4, they=0; add-one fish=4/7, can=2/7, they=1/7.
6. s(N)=1, s(V)=0. N has three outgoing events: to V, to N, to STOP once each, so its row is (1/3,1/3,1/3). V→STOP=1. N emits a with probability 1; V emits b with probability 1. Joint probability **1×1×(1/3)×1×1=1/3**. The unobserved other emissions/transitions are zero.
7. Via N: −7−2−1=−10. Via V: −4−1−1=−6. Take −6, with predecessor V.
8. N gives −7; V gives −6; **V** wins. Include STOP before selecting the final tag.
9. No. Dividing the joint by P(x) is required to obtain P(y|x); dropping this denominator is valid only for finding the argmax.
10. **5/8 and 3/8**. Count NP expansions/nodes, not sentences.
11. One parse, NP→NP PP with two terminal NPs: **0.3×0.4²=0.048**.
12. **A→B @A_CDE [0.2]; @A_CDE→C @A_DE [1]; @A_DE→D E [1]**. Preserve all other A alternatives. The fragment product is 0.2.
13. Vertical remembers ancestors; horizontal remembers sibling context in artificial nodes. More context gives finer distinctions but a larger, sparser model.
14. **5×6/2=15** spans; split points **1,2,3,4**, so four.
15. No. You need a finite S score over the whole sentence, perhaps through a licensed unary chain. A full-span NP is insufficient by itself.
16. No. Multiplying the same rule probabilities the same number of times produces a tie. Refined or lexical/context-sensitive information is needed to distinguish the analyses.

## 12. Common traps and mastery checklist

| Trap | Correct check |
|---|---|
| Reversing emission direction | P(word|tag), not P(tag|word) |
| Transposing transitions | Read row/column headings; this guide uses source rows |
| Forgetting STOP | Include it in the model's joint score and outgoing normalization |
| Emitting a word from STOP | STOP ends generation; it has no token here |
| Smoothing one table and assuming all zeros disappear | Inspect each start/transition/emission factor |
| Reporting joint as conditional probability | Normalize by P(x) if the numerical posterior is requested |
| Picking the most likely emission at each word | Transitions couple decisions; use the complete model |
| Keeping only one prefix or one category per span | Keep a best result for every state |
| Reading off each column's winner | Recover a coherent sequence using backpointers |
| Treating −10 as better than −6 | Larger log score is better |
| Summing logs where a max is required | Add local factors; max over competing derivations |
| Confusing parse probability with sentence probability | A sentence may have several parses; sum their probabilities |
| Normalizing all grammar rules together | Normalize separately for each left-hand side |
| Counting each distinct rule once | Count every observed expansion occurrence |
| Claiming all CFG trees are binary | Binarization is a preprocessing transformation |
| Applying vertical annotation after estimation | Transform training trees, then extract/count |
| Using a plausible but absent lexical rule | Parseability depends on the supplied grammar |
| Missing unary chains | Apply unary closure or use valid preprocessing |
| Combining nonadjacent CKY spans | Children must partition [i,j) at one split k |
| Accepting any category over the whole input | Require the specified root |
| Adding rules without changing probabilities | Structural repair and PCFG renormalization are separate steps |

Before the exam, check that you can do each item with a fresh example:

- [ ] Explain lexical POS ambiguity versus structural attachment ambiguity.
- [ ] Build target-relative tagger features and identify label leakage.
- [ ] State the HMM conditional independence assumptions precisely.
- [ ] Multiply all initial, transition, emission, and STOP factors.
- [ ] Estimate and smooth the correct conditional distributions.
- [ ] Complete a Viterbi chart, terminate, and backtrack.
- [ ] Explain O(nK²) computation versus Kⁿ possible tag sequences.
- [ ] Read/draw a bracketed tree and extract its rules.
- [ ] Estimate PCFG rule probabilities from repeated node counts.
- [ ] Distinguish one tree's probability from a sentence's total probability.
- [ ] Binarize a production while preserving its probability.
- [ ] Explain vertical versus horizontal Markovization and their sparsity tradeoffs.
- [ ] Apply parent annotation to training trees and explain what combinations disappear.
- [ ] Fill a CKY chart using spans, split points, and root categories.
- [ ] Explain why grammatical possibility and semantic plausibility differ.

## Sources and coverage audit

The source labels below identify course materials; numerical extensions and new practice are guide-authored. Auto-generated caption errors in technical names/tags are interpreted using the surrounding material and exercise tables. No external web sources were needed.

| Source | Coverage in this guide |
|---|---|
| MIDTERM_TOPICS.pdf, pp. 1–2 | Scope, question styles, exclusions, reduced structured-prediction emphasis |
| 7.0.0 Part-of-Speech Tagging | §1: roles, ambiguity, Fed example |
| 7.1.0 Sequence Labeling: Tagging with Classifiers | §2: token-relative features, independent tagging |
| 7.2.0 Hidden Markov Models | §3: joint model, assumptions, parameters |
| 7.2.1 HMMs: Parameter Estimation | §4: supervised counts, STOP, smoothing |
| 7.3.0 HMMs: Viterbi Algorithm | §5: recurrence, exactness, backtracking |
| 7.3.1 HMMs for POS Tagging | §§2,4,5: errors, unknowns, neural windows, trigram states |
| 7.4.0 Constituency Parsing | §6: trees, ambiguity, diagnostics |
| 7.4.1 Probabilistic Context-Free Grammars | §§7–8: rules, estimation, binarization |
| 7.4.2 CKY Algorithm | §9: fenceposts, recurrence, runtime, unary caveat |
| 7.4.3 Refining Grammars | §8: vertical/horizontal history, lexicalization |
| Exercises 7.0–7.5 | Full question coverage in §§1–5,7,9, with worked reasoning |
| 2022 solutions Q1(2,8–10), Q4 | §10: HMM direction/paths, PCFG scoring and refinement |
| 2023 solutions Q1(19–20), Q4 | §10: Viterbi complexity, extraction, annotation, lexicon erratum |

The older discovery map's claim that Module_7 is empty is obsolete: the materials used here are present now. This guide covers Module 7, not the entire midterm.

### Open original materials

- [Current exam topics](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/MIDTERM_TOPICS.pdf>)
- [2022 midterm with solutions](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/midterms/fa22-cs388-midterm-solns.pdf>)
- [2023 midterm with solutions](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/midterms/fa23-cs388-midterm-solns.pdf>)
- [7.0.0 Part-of-Speech Tagging_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.0.0 Part-of-Speech Tagging_Captions_English (United States).txt>)
- [7.1.0 Sequence Labeling_ Tagging with Classifiers_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.1.0 Sequence Labeling_ Tagging with Classifiers_Captions_English (United States).txt>)
- [7.2.0 Hidden Markov Models_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.2.0 Hidden Markov Models_Captions_English (United States).txt>)
- [7.2.1 HMMs_ Parameter Estimation_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.2.1 HMMs_ Parameter Estimation_Captions_English (United States).txt>)
- [7.3.0 HMMs_ Viterbi Algorithm_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.3.0 HMMs_ Viterbi Algorithm_Captions_English (United States).txt>)
- [7.3.1 HMMs for POS Tagging_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.3.1 HMMs for POS Tagging_Captions_English (United States).txt>)
- [7.4.0 Constituency Parsing_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.4.0 Constituency Parsing_Captions_English (United States).txt>)
- [7.4.1 Probabilistic Context-Free Grammars_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.4.1 Probabilistic Context-Free Grammars_Captions_English (United States).txt>)
- [7.4.2 CKY Algorithm_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.4.2 CKY Algorithm_Captions_English (United States).txt>)
- [7.4.3 Refining Grammars_Captions_English (United States)](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/7.4.3 Refining Grammars_Captions_English (United States).txt>)
- [Exercise 7.0 POS Tagging example](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/[Exercise] 7.0 POS Tagging example.png>)
- [Exercise 7.1 Feature extraction for taggers](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/[Exercise] 7.1 Feature extraction for taggers.png>)
- [Exercise 7.2 Computing HMM Probabilities](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/[Exercise] 7.2 Computing HMM Probabilities.png>)
- [Exercise 7.3 Running Viterbi](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/[Exercise] 7.3 Running Viterbi.png>)
- [Exercise 7.4 PCFGs](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/[Exercise] 7.4 PCFGs.png>)
- [Exercise 7.5 CKY algorithm](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/[Exercise] 7.5 CKY algorithm.png>)
