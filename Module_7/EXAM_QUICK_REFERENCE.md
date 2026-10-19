# Module 7: exam quick reference

Use alongside [full study guide](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/STUDY_GUIDE.md>) for explanations and full worked examples. Scope: POS ambiguity, HMMs, constituency trees, PCFGs, binarization/Markovization, CKY. The current topic sheet reduces tagging/parsing emphasis, especially Viterbi; dependency parsing is excluded.

## Find the right method

| Question asks for… | Do this |
|---|---|
| Possible tags versus intended tag | Separate lexical possibilities from contextual use |
| Tagger features | Identify current word, relative neighbors, morphology/shape |
| HMM probability for given words and tags | Multiply initial, emissions, transitions, STOP |
| HMM parameters from labeled data | Count events and normalize each conditional row |
| Best tag sequence | Viterbi; terminate with STOP; backtrack |
| Constituency ambiguity | Draw alternative attachments/groupings and explain meaning |
| Grammar from trees | Read parent→ordered children at each node |
| PCFG probabilities | Count each expansion / all expansions of that parent |
| Probability of one tree | Product of its rule probabilities, including lexical rules |
| Probability of a sentence | Sum probabilities of every parse for that sentence |
| Best parse | CKY max chart; recover rule/split backpointers |
| Reduce branching to two children | Binarize, with context-preserving artificial symbols |
| Distinguish subject/object phrase behavior | Vertical parent annotation |
| Control sibling context in artificial symbols | Horizontal Markovization |

## POS and classifier reminders

“Fed raises interest rates”: intended **proper noun, verb, noun, plural noun**. “Interest rates” is a noun–noun compound. A different tag sequence can be syntactically possible yet semantically implausible.

Same sentence bag-of-words vector at every position → same independent tag distribution. Use **f(x,i)** with CurrentWord, PreviousWord, NextWord, suffix/prefix, capitalization, digits, word length, seen/unseen status. The candidate tag may select a scoring block; the gold tag must not be used as an observed feature.

A three-word neural window concatenating d-dimensional embeddings has dimension **3d**. Averaging loses the slot identities.

**Source caveat:** Exercise 7.0's selected sequence labels “I” and “my” as N. Ordinarily “I” is a pronoun and “my” a possessive determiner; use the exam's supplied category definitions.

## HMM: define symbols before calculating

K tags, M words. **s(t)=P(first tag=t)**; **A(u,t)=P(next=t|current=u)**; **B(t,w)=P(word=w|tag=t)**. Here A's rows are source tags and columns are destination tags.

**P(x,y) = s(y₁) B(y₁,x₁) × ∏ᵢ₌₂ⁿ A(yᵢ₋₁,yᵢ)B(yᵢ,xᵢ) × A(yₙ,STOP).**

For n words: **one start + n emissions + (n−1) ordinary transitions + one STOP = 2n+1 factors**. No STOP emission. Use the question's convention if STOP is absent.

- Start estimate: **#sentences starting t / #sentences**.
- Transition estimate: **#u→t / #outgoing events from u**, including STOP.
- Emission estimate: **#tokens w tagged t / #tokens tagged t**.
- Add-α for L legal outcomes: **(c+α)/(total+αL)**.
- No cross-sentence transitions. Smoothing one distribution leaves zeros elsewhere untouched.
- Markov: next tag depends on preceding tag; each word depends on its own tag in the basic model. These are conditional assumptions.
- Unknown words require emission support (e.g., UNK); transitions alone cannot overcome an all-zero emission.

**Probability direction:** B(V,can) is P(can|V), not P(V|can).

**Joint versus posterior:** P(y|x)=P(x,y)/P(x). They have the same maximizing y for fixed x, but different numerical values.

### Viterbi in log space

```text
v₁(t) = log s(t) + log B(t,x₁)
vᵢ(t) = log B(t,xᵢ) + max_u [vᵢ₋₁(u)+log A(u,t)]
bpᵢ(t) = maximizing u

Finish: max_t [vₙ(t)+log A(t,STOP)]
Backtrack from the maximizing final tag using bp.
```

Cell = best prefix score **ending in the specified tag**, with all words through that position emitted. Keep one best prefix per ending tag. A column's overall winner need not be on the final best path.

**Time O(nK²); full chart/backpointers O(nK); nominal paths Kⁿ.** Pair-state trigram HMM with overlap constraints: optimized O(nK³).

Logs: add factors, max competing paths; **log 0=−∞; −4 beats −7**. Probability space: multiply factors, max paths. Sum/log-sum-exp over paths computes a different quantity.

### Course numerical anchors

| Example | Result |
|---|---|
| Exercise 7.2, they/N can/V fish/V | 1·1·(3/5)·(1/2)·(1/5)·(1/2)·(3/5) = **9/500** |
| Exercise 7.2, other viable tag paths | None; emission zeros force N V V |
| Exercise 7.3 log chart, N row | they −2; can −7; fish −6; after STOP **−7** |
| Exercise 7.3 log chart, V row | they −4; can −4; fish −6; after STOP −8 |
| Exercise 7.3 best sequence | **N V N**, score **−7** |
| 2022 Q1(8), big/J car/N | **1/36**, including STOP |
| 2022 Q1(9), big convertible car | **2** viable paths: J J N, J N N |
| 2022 Q1(10), big car convertible | **1** viable path: J N N |

Exercises 7.2 and 7.3 use **different models**. Do not transfer their zeros or scores between them.

## Constituency and PCFGs

Words = terminals/leaves; POS tags = preterminals; NP, VP, PP, S = phrase nonterminals. Root is specified by the grammar. Trees need not initially be binary.

Attachment examples: “with chopsticks” modifies either **eat** (instrument) or **spaghetti** (noun modifier). “On governor's desk” modifies either the **ban** or the **dancing**. Same words/tags can have different trees.

**Normalize per parent:** Σβ P(A→β|A)=1.

**MLE:** P(A→β|A)=count(A→β)/Σγcount(A→γ).

**Tree probability:** ∏rule occurrences P(rule|parent). Include repeats and lexical rules. Identical rule multisets imply identical PCFG scores. The model cannot break that tie with meaning alone.

**Sentence probability:** sum over its trees. **Best tree:** max over its trees. Divide a tree's joint probability by P(sentence) for its conditional probability.

### Grammar transformations

```text
Lossless binarization:
VP→V NP PP [p]
becomes
VP→V @VP_NP_PP [p]
@VP_NP_PP→NP PP [1]
```

Unique/context-preserving intermediate symbols force the intended continuation. Remove artificial nodes to recover original trees. Collapsing intermediate symbols can change combinations/probabilities.

**Vertical:** annotate ancestors, e.g. NP^S versus NP^VP. Transform training trees **before** extraction. More context → more distinctions, more parameters/sparsity. Exact parent refinement cannot add strings relative to the original extracted grammar after labels are erased; it may remove some.

**Horizontal:** limit remembered sibling history during binarization; h=0 none, h=1 one prior sibling, h=2 two. Follow the question's direction/history convention. Merged contexts require re-estimation.

**Lexicalization:** annotate head words to capture lexical preferences; increases model complexity.

## CKY: spans and recurrence

Use zero-based **[i,j)**, words xᵢ,…,xⱼ₋₁. **C[i,j,A]** = best log score of an A-rooted subtree covering exactly that span. Initialize missing states to −∞.

```text
Lexical:
C[i,i+1,A] = log P(A→xᵢ | A)

Binary:
C[i,j,A] = max over i<k<j and rules A→B D of
  log P(A→B D | A) + C[i,k,B] + C[k,j,D]
```

Fill by increasing span length. Record winning rule and split. Child order matters; child spans must partition the parent. Apply unary closure if nonterminal unary rules remain; the binary recurrence alone cannot apply NP→PRP.

Finish at **C[0,n,required root]**. Finite score → a parse exists. Root score −∞ → no parse. Backtrack for the tree.

**Time O(n³|R₂|); space O(n²|N|)** for binary-rule CKY. Fixed grammar → O(n³). There are n(n+1)/2 nonempty spans; span length ℓ has ℓ−1 candidate split points.

### Exercise 7.5 anchor

Grammar: S→NP VP [1]; NP→DT NN [1/2] or NN NNS [1/2]; VP→VBZ PRP [1]. Lexicon: DT→the, NN→child, NNS→raises, VBZ→raises, PRP→it, each probability 1.

For **child raises raises it**, lexical scores are 0; NP over [0,2) is −1; VP over [2,4) is 0; S over [0,4) is **−1**, using log base 2. Parse:

```text
(S (NP (NN child) (NNS raises))
   (VP (VBZ raises) (PRP it)))
```

Tree probability **1/2**. No other full parse in this grammar.

For **the child raises raises**, adding either **VP→VBZ NNS** or **PRP→raises** makes a parse possible individually. Renormalize affected probabilities if a valid PCFG is required.

## Past-exam caveats and final checks

2023 Q4's displayed lexicon omits **him**. “Not parseable” is accepted as written. With the intended extra PRP→him rule, the unrefined grammar admits “him teach I”; parent annotation blocks it because NP^VP→PRP is absent.

Before submitting a calculation:

1. Check what is requested: joint, posterior, best path/tree, or total sentence probability.
2. Check direction, log base, STOP, vocabulary/tag set, and required root.
3. Show factors or candidate scores and the maximizing backpointer/rule.
4. Verify every word/tag/rule is licensed and every conditional distribution uses the right denominator.
5. Distinguish an ugly but allowed analysis from an analysis the supplied model cannot generate.

Full sources and explanations are in [full study guide](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_7/STUDY_GUIDE.md>), including all ten lecture transcripts, all six course exercises, the Fall 2026 topic sheet, and selected 2022/2023 questions.
