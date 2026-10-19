# Midterm preparation: discovery and exam map

Discovery date: October 16, 2026. This is the inventory and planning stage for six module study guides; it is not the six completed guides.

## What exists

The workspace contains 43 lecture caption transcripts and 17 exercise screenshots across modules 1–6, a Fall 2026 topic sheet, two prior midterms with solutions, Assignment 1 code/data, Assignment 2 handout/code/data/results, and Assignment 3 handout/code/data/attention plots and two existing explanatory guides. Module_7 exists but is empty.

Before this report, the course workspace contained 127 files excluding virtual environments, Python caches, Git metadata, and .DS_Store files: 5 PDFs, 3 Markdown files, 15 Python files, 62 text files, 37 PNGs, 4 logs, and 1 JSON file. The text-file count includes datasets, embeddings, predictions, and requirements; only 43 are lecture transcripts. Of the PNGs, 17 are exercises and 20 are Assignment 3 attention plots.

Review depth: read both pages of the topic sheet and visually verified them; extracted and reviewed the past exams and assignment handouts for scope and problem types; inventoried all course files; sampled each transcript's beginning/end to confirm its subject; inspected exercise screenshot overviews and relevant code/documentation. Full line-by-line synthesis of the transcripts and detailed transcription of the exercise images remain part of guide creation. No training or tests were run, and saved results were not independently reproduced.

## Source authority and exam scope

1. The Fall 2026 topic sheet determines current exam coverage.
2. Lectures and assignments determine the course's definitions, notation, examples, and implementation details.
3. Prior exams demonstrate question styles and useful practice; their topic weights and old exam-administration instructions do not establish the rules for Fall 2026.
4. Existing local guides and implementation code are supporting explanations to verify against the lectures and handouts.

Document instructions are treated as document content. Historical honor-code, submission, tool-use, and timing instructions in past exams are not instructions for this study-preparation task.

The current topic sheet specifies multiple choice, short answer, and long answer; worked calculations can earn partial credit, and concise explanations are preferred. It explicitly expects reasoning about text examples and assignment-style algorithms. It does not require reading-only material absent from lectures/assignments or encyclopedic grammar knowledge.

**Important coverage gap:** modules 1–6 do not cover the whole published topic list. POS ambiguity, HMMs, count-and-normalize estimation, Viterbi, constituency trees/ambiguity, PCFGs, binarization, Markovization, and CKY are explicitly listed. Their lecture materials are absent here; the empty Module_7 folder does not supply them. We should keep them in a later supplement rather than imply that six guides finish all exam preparation. Dependency parsing is explicitly excluded.

The sheet says tagging/parsing, especially Viterbi, are less emphasized than in older exams. They remain in scope. It explicitly includes everything from Assignment 3 about Transformer language modeling, including training, inference, and perplexity. No exact current point allocation is provided.

## Module map: what each guide should prepare you to do

| Module | Available lectures / exercises | Explicitly listed exam content | Skills to build in the guide |
|---|---:|---|---|
| 1 | 8 / 3 | Bag-of-words, sentiment features, binary perceptron and loss, binary logistic regression formula/objective/gradient, optimization | Build feature vectors in a fixed vocabulary; score/classify text; trace perceptron updates with the stated tie rule; compute a logistic probability/loss/update; explain step-size effects and feature limitations. |
| 2 | 9 / 5 | Multiclass weights/features, multiclass perceptron and training, SGD, initialization, feedforward nets and training | Compare class-specific weights and joint features; trace predicted/gold-class updates; compute softmax distributions; track layer dimensions; explain nonlinearities, initialization, and the forward/loss/backward/update loop. |
| 3 | 5 / 2 | Skip-gram definition/properties/bias; Assignment 2 deep averaging networks and limitations | Form word-context pairs; compute dot-product scores and normalized context probabilities; distinguish word/context vectors; compare similarity measures; explain bias; trace embedding averaging through a classifier and explain order loss. |
| 4 | 9 / 3 | N-gram definitions and count estimation; self-attention mathematics; foundations for Transformer LM evaluation | Normalize counts, multiply conditional probabilities, reason about unseen contexts; compute attention weights and weighted sums; track Q/K/V shapes and attention axes; explain positional information; calculate perplexity. |
| 5 | 6 / 1 | Transformer architecture; Assignment 3 LM training/inference/evaluation; decoder inference, beam search, nucleus sampling | Identify attention/FFNN/residual/norm components; distinguish training targets from inputs; apply causal masks; trace inference and log-probability scoring; compare probability outputs with argmax; compute perplexity; execute beam search and top-p truncation/renormalization. |
| 6 | 6 / 3 | Masked language modeling/BERT; subword tokenization; connections to decoder inference | Explain bidirectional encoding and masked prediction versus autoregression; trace BERT classification/fine-tuning; segment with a supplied subword vocabulary; explain vocabulary/rare-word tradeoffs; compare encoder, decoder, and encoder-decoder use. |

These skills are an interpretation of the published scope plus observed exercises/past questions, not a prediction of exact exam questions.

### Supporting content versus explicitly named coverage

- Module 2 fairness in classification is present in lectures but not separately named in the topic sheet. Embedding bias in Module 3 is explicitly named.
- Module 3 contains other embedding methods, useful for contrasting skip-gram with CBOW/GloVe; their detailed algorithms are not separately named in the current list.
- Module 4 includes smoothing, RNN limitations, multi-head attention, and positional encodings. Use these to support the named language-modeling and Transformer topics. The sheet does not separately promise detailed smoothing derivations or RNN training derivations.
- Module 5 Transformer extensions/scaling ideas are available but not separately named. Core architecture and Assignment 3 deserve priority.
- Module 6 includes seq2seq, BART, and T5. Include a concise comparison for context, with most effort on the explicitly named BERT/MLM and tokenization topics. Their presence in lectures means they should not be declared excluded.

## What the available practice actually tests

### Prior exam evidence

**2023:** Part 1 Q1(1) tests perceptron/SGD/convexity distinctions; Q1(3–4) DAN parameters and embedding bias; Q1(7–8) bigram probabilities and changing corpus counts; Q1(9–11) architectural changes and parameter counts; Q1(12) BERT pretraining choices; Q1(13–14) attention/output shapes; Q1(15) generalization; Q1(16–18) beam search and sampling. Long Q2 combines bag-of-words, perceptron, subwords, and typo robustness. Long Q3 computes skip-gram probabilities and reasons about embeddings. Long Q4 extracts and modifies grammars with vertical Markovization.

**2022:** Part 1 Q1(1) forms skip-gram training pairs; Q1(3–6) checks sigmoid, feature preprocessing, bias, and valid neural architectures; Q1(11–13) compares dot product/cosine similarity. Q2(a,c) asks for brief model comparisons and concrete classifier improvements. Q3 traces perceptron, reasons about separability, and contrasts neural capacity. Q5 adapts skip-gram and calculates runtime/parameter counts. Other questions include HMM/PCFG and historical structured-prediction topics.

Use the 2023 paper first for overlap with modules 4–6, and select relevant 2022 questions for foundations and embeddings. Do not copy the old papers' topic proportions into a new mock exam. CRFs, forward-backward, BIO tagging/NER evaluation, and dependency parsing appear in the 2022 paper; they should not automatically become priorities for the current six guides. Dependency parsing is explicitly excluded in the current topic sheet.

The topic sheet calls language modeling/Transformers new material, although the supplied 2023 exam already includes them. Use the actual questions as useful practice and the current sheet as the coverage authority; do not assume the historical note is a literal description of every supplied past paper.

### Recurring task patterns

1. **Execute:** turn text into features, follow an update, normalize counts, compute a distribution, or trace a decoding step.
2. **Explain:** identify why a model fails, what information it discards, or how a change affects its behavior.
3. **Check structure:** determine valid tensor shapes, layer connections, parameter counts, or runtime.
4. **Transfer:** adapt an existing model to a modified example or representation.
5. **Distinguish:** separate learned parameters from activations, raw logits from probabilities/log probabilities, attention from a complete Transformer, and masked prediction from next-token prediction.

### Details that deserve explicit warnings in the guides

- Perceptron tie rules differ across supplied exams: 2022 Q3 uses strictly positive scores; 2023 Q2 uses nonnegative scores. The rule changes the zero-initialized trace.
- The 2023 embedding question Q3(c–d) has an acknowledged problem involving the zero word vector. Present the correction with the practice question rather than treating the intended solution as unqualified truth.
- The 2023 grammar question Q4(b) acknowledges a removed lexical item. Keep the accepted alternatives and assumptions visible.
- Caption transcripts contain transcription errors; equations and technical names should be reconstructed carefully from the course context, exercises, handouts, and code.
- Screenshot correctness indicators and feedback are useful, but selected answers alone should not replace an explanation.
- Architecture questions must state assumptions: single-head versus multi-head, batched versus unbatched, projection sizes, and whether biases count.

## Assignments and existing guides

### Assignment 1: classification foundation

`a1-distrib/` includes model implementations, driver, data reader, utilities, requirements, sentiment train/dev/blind-test data, and blind predictions. `models.py` contains unigram/bigram/better feature extractors and perceptron/logistic-regression training. This supports Module 1, with connections to Module 2. No Assignment 1 PDF handout is present.

### Assignment 2: optimization and deep averaging

`a2/a2-1.pdf` covers quadratic optimization, a DAN using pretrained embeddings, and typo robustness. `a2/a2-distrib/` contains the implementation, tests, sentiment data, typo-development data, and relativized 50d/300d GloVe vectors. `a2/README.md` explains the local implementation and saved results. `a2/results/` contains four logs. This supports Modules 2–3 and generalization connections to Module 6. Saved results are historical records, not a fresh validation in this discovery pass.

### Assignment 3: attention and Transformer language modeling

`a3/a3-2.pdf` covers the letter-counting Transformer and causal character language model. Implementation files include `transformer.py`, `transformer_lm.py`, `letter_counting.py`, and `lm.py`; datasets include letter-counting and text8 splits. There are 20 saved attention plots and an evaluation JSON.

Two existing Markdown guides explain P0/P1 and the transition to P2. Reuse their explanations of positions, tensor shapes, shifted targets, masking, and sequence scoring after checking them against the final study-guide notation. Treat the local implementation's simplified attention design separately from a standard multi-head Transformer. The saved evaluation JSON is not evidence of a run performed today.

## Proposed structure for each of the six guides

1. Exam targets: the concrete tasks to perform, linked to the topic sheet.
2. Concepts and notation: enough intuition to understand the model, with defined symbols.
3. Essential equations: meanings, dimensions, and when each formula applies.
4. Worked examples: numerical or text-based examples, including intermediate steps.
5. Assignment connections: explain the relevant implementation without requiring memorization of code.
6. Common traps and model limitations.
7. Practice: selected course exercises and past-exam references, plus clearly labeled original questions where coverage is thin.
8. Answers and a concise mastery checklist.

Build in module order: 1 → 2 → 3 → 4 → 5 → 6. Share notation across guides. Give Modules 4–5 substantial worked practice because the current scope explicitly singles out Assignment 3. Keep the missing tagging/parsing coverage visible as a separate follow-on supplement.

## Full course-file inventory

Paths below link to local files. Virtual environments, caches, and OS metadata are excluded. This list records files present before this report was created.

### Module_1

- [1.1.0 Linear Binary Classification [REVIEW]_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/1.1.0 Linear Binary Classification [REVIEW]_Captions_English (United States).txt>)
- [1.1.1 Sentiment Analysis and Basic Feature Extraction_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/1.1.1 Sentiment Analysis and Basic Feature Extraction_Captions_English (United States).txt>)
- [1.1.2 Basics of Learning_ Gradient Descent_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/1.1.2 Basics of Learning_ Gradient Descent_Captions_English (United States).txt>)
- [1.2.0 Perceptron [REVIEW]_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/1.2.0 Perceptron [REVIEW]_Captions_English (United States).txt>)
- [1.2.1 Perceptron as minimizing loss_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/1.2.1 Perceptron as minimizing loss_Captions_English (United States).txt>)
- [1.3.0 Logistic Regression [REVIEW]_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/1.3.0 Logistic Regression [REVIEW]_Captions_English (United States).txt>)
- [1.4.0 Sentiment Analysis_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/1.4.0 Sentiment Analysis_Captions_English (United States).txt>)
- [1.4.1 Optimization Basics_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/1.4.1 Optimization Basics_Captions_English (United States).txt>)
- [[Exercise] 1.0 Sentiment Analysis and Features.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/[Exercise] 1.0 Sentiment Analysis and Features.png>)
- [[Exercise] 1.1 Perceptron Execution.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/[Exercise] 1.1 Perceptron Execution.png>)
- [[Exercise] 1.2 Logistic Regression Update.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_1/[Exercise] 1.2 Logistic Regression Update.png>)

### Module_2

- [2.0.0 Multiclass Classification_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.0.0 Multiclass Classification_Captions_English (United States).txt>)
- [2.0.1 Multiclass Perceptron and Logistic Regression_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.0.1 Multiclass Perceptron and Logistic Regression_Captions_English (United States).txt>)
- [2.0.2 Multiclass Classification Examples_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.0.2 Multiclass Classification Examples_Captions_English (United States).txt>)
- [2.0.3 Fairness in Classification_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.0.3 Fairness in Classification_Captions_English (United States).txt>)
- [2.1.0 Neural Networks_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.1.0 Neural Networks_Captions_English (United States).txt>)
- [2.1.1 Neural Network Visualization_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.1.1 Neural Network Visualization_Captions_English (United States).txt>)
- [2.2.0 Feedforward Neural Networks_ Backpropagation [REVIEW]_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.2.0 Feedforward Neural Networks_ Backpropagation [REVIEW]_Captions_English (United States).txt>)
- [2.2.1 Neural Net Implementation_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.2.1 Neural Net Implementation_Captions_English (United States).txt>)
- [2.2.2 Neural Net Training_ Optimization_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/2.2.2 Neural Net Training_ Optimization_Captions_English (United States).txt>)
- [[Exercise] 2.0 Multiclass Perceptron.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/[Exercise] 2.0 Multiclass Perceptron.png>)
- [[Exercise] 2.1 Multiclass Updates.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/[Exercise] 2.1 Multiclass Updates.png>)
- [[Exercise] 2.2 Classification techniques.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/[Exercise] 2.2 Classification techniques.png>)
- [[Exercise] 2.3a Neural Network Optimization and Initialization.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/[Exercise] 2.3a Neural Network Optimization and Initialization.png>)
- [[Exercise] 2.3b FFNN Hands-on.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_2/[Exercise] 2.3b FFNN Hands-on.png>)

### Module_3

- [3.0.0 Word Embeddings_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_3/3.0.0 Word Embeddings_Captions_English (United States).txt>)
- [3.1.0 Skip-gram_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_3/3.1.0 Skip-gram_Captions_English (United States).txt>)
- [3.1.1 Other word embedding methods_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_3/3.1.1 Other word embedding methods_Captions_English (United States).txt>)
- [3.1.2 Bias in Word Embeddings_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_3/3.1.2 Bias in Word Embeddings_Captions_English (United States).txt>)
- [3.2.0 Applying Embeddings_ Deep Averaging Networks_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_3/3.2.0 Applying Embeddings_ Deep Averaging Networks_Captions_English (United States).txt>)
- [[Exercise] 3.0 Word Embedding Semantics.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_3/[Exercise] 3.0 Word Embedding Semantics.png>)
- [[Exercise] 3.1 Calculating Skipgram Embeddings.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_3/[Exercise] 3.1 Calculating Skipgram Embeddings.png>)

### Module_4

- [4.0.0 N-gram LMs_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.0.0 N-gram LMs_Captions_English (United States).txt>)
- [4.0.1 Smoothing in N-gram LMs_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.0.1 Smoothing in N-gram LMs_Captions_English (United States).txt>)
- [4.0.2 LM Evaluation_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.0.2 LM Evaluation_Captions_English (United States).txt>)
- [4.1.0 Neural Language Models_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.1.0 Neural Language Models_Captions_English (United States).txt>)
- [4.1.1 RNNs and their Shortcomings_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.1.1 RNNs and their Shortcomings_Captions_English (United States).txt>)
- [4.2.0 Attention_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.2.0 Attention_Captions_English (United States).txt>)
- [4.2.1 Self-Attention_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.2.1 Self-Attention_Captions_English (United States).txt>)
- [4.2.2 Multi-Head Self Attention_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.2.2 Multi-Head Self Attention_Captions_English (United States).txt>)
- [4.2.3 Position Encodings_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/4.2.3 Position Encodings_Captions_English (United States).txt>)
- [[Exercise] 4.0 Networks for LMs.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/[Exercise] 4.0 Networks for LMs.png>)
- [[Exercise] 4.1 Attention.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/[Exercise] 4.1 Attention.png>)
- [[Exercise] 4.2 Self Attention.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_4/[Exercise] 4.2 Self Attention.png>)

### Module_5

- [5.0.0 Transformer Architecture_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_5/5.0.0 Transformer Architecture_Captions_English (United States).txt>)
- [5.0.1 Using Transformers_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_5/5.0.1 Using Transformers_Captions_English (United States).txt>)
- [5.0.2 Transformer Language Modeling_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_5/5.0.2 Transformer Language Modeling_Captions_English (United States).txt>)
- [5.0.3 Transformer Extensions_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_5/5.0.3 Transformer Extensions_Captions_English (United States).txt>)
- [5.1.0 Beam Search_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_5/5.1.0 Beam Search_Captions_English (United States).txt>)
- [5.1.1 Nucleus Sampling_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_5/5.1.1 Nucleus Sampling_Captions_English (United States).txt>)
- [[Exercise] 5.0 Nucleus Sampling.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_5/[Exercise] 5.0 Nucleus Sampling.png>)

### Module_6

- [6.0.0 BERT_ Masked Language Modeling_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/6.0.0 BERT_ Masked Language Modeling_Captions_English (United States).txt>)
- [6.0.1 BERT_ Model and Applications_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/6.0.1 BERT_ Model and Applications_Captions_English (United States).txt>)
- [6.1.0 Seq2seq Models_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/6.1.0 Seq2seq Models_Captions_English (United States).txt>)
- [6.1.1 BART_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/6.1.1 BART_Captions_English (United States).txt>)
- [6.1.2 T5_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/6.1.2 T5_Captions_English (United States).txt>)
- [6.2.0 Word Piece and Byte Pair Encoding_Captions_English (United States).txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/6.2.0 Word Piece and Byte Pair Encoding_Captions_English (United States).txt>)
- [[Exercise] 6.0 BERT properties.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/[Exercise] 6.0 BERT properties.png>)
- [[Exercise] 6.1 Seq2seq models.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/[Exercise] 6.1 Seq2seq models.png>)
- [[Exercise] 6.2 Subword tokenization.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/Module_6/[Exercise] 6.2 Subword tokenization.png>)

### Module_7

Empty directory.

### Topic sheet and past exams

- [MIDTERM_TOPICS.pdf](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/Midterm_Prep_By_Module/MIDTERM_TOPICS.pdf>)
- [fa22-cs388-midterm-solns.pdf](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/midterms/fa22-cs388-midterm-solns.pdf>)
- [fa23-cs388-midterm-solns.pdf](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/midterms/fa23-cs388-midterm-solns.pdf>)

### a1-distrib

- [data/dev.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/data/dev.txt>)
- [data/test-blind.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/data/test-blind.txt>)
- [data/train.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/data/train.txt>)
- [models.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/models.py>)
- [requirements.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/requirements.txt>)
- [sentiment_classifier.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/sentiment_classifier.py>)
- [sentiment_data.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/sentiment_data.py>)
- [test-blind.output.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/test-blind.output.txt>)
- [utils.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a1-distrib/utils.py>)

### a2

- [README.md](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/README.md>)
- [a2-1.pdf](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-1.pdf>)
- [a2-distrib/data/dev-typo.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/data/dev-typo.txt>)
- [a2-distrib/data/dev.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/data/dev.txt>)
- [a2-distrib/data/glove.6B.300d-relativized.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/data/glove.6B.300d-relativized.txt>)
- [a2-distrib/data/glove.6B.50d-relativized.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/data/glove.6B.50d-relativized.txt>)
- [a2-distrib/data/test-blind.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/data/test-blind.txt>)
- [a2-distrib/data/train.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/data/train.txt>)
- [a2-distrib/models.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/models.py>)
- [a2-distrib/neural_sentiment_classifier.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/neural_sentiment_classifier.py>)
- [a2-distrib/optimization.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/optimization.py>)
- [a2-distrib/requirements.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/requirements.txt>)
- [a2-distrib/sentiment_data.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/sentiment_data.py>)
- [a2-distrib/test-blind.output.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/test-blind.output.txt>)
- [a2-distrib/test_solution.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/test_solution.py>)
- [a2-distrib/utils.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/a2-distrib/utils.py>)
- [results/checks.log](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/results/checks.log>)
- [results/clean-run.log](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/results/clean-run.log>)
- [results/optimization-run.log](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/results/optimization-run.log>)
- [results/typo-run.log](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a2/results/typo-run.log>)

### a3

- [P0-and-P1-practical-guide.md](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/P0-and-P1-practical-guide.md>)
- [P1-to-P2-guide.md](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/P1-to-P2-guide.md>)
- [a3-2.pdf](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-2.pdf>)
- [a3-distrib/data/lettercounting-dev.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/data/lettercounting-dev.txt>)
- [a3-distrib/data/lettercounting-train.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/data/lettercounting-train.txt>)
- [a3-distrib/data/text8-100k.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/data/text8-100k.txt>)
- [a3-distrib/data/text8-dev.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/data/text8-dev.txt>)
- [a3-distrib/data/text8-test.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/data/text8-test.txt>)
- [a3-distrib/letter_counting.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/letter_counting.py>)
- [a3-distrib/lm.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/lm.py>)
- [a3-distrib/output.json](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/output.json>)
- [a3-distrib/plots/0_attns0.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/0_attns0.png>)
- [a3-distrib/plots/0_attns1.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/0_attns1.png>)
- [a3-distrib/plots/0_attns2.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/0_attns2.png>)
- [a3-distrib/plots/0_attns3.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/0_attns3.png>)
- [a3-distrib/plots/1_attns0.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/1_attns0.png>)
- [a3-distrib/plots/1_attns1.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/1_attns1.png>)
- [a3-distrib/plots/1_attns2.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/1_attns2.png>)
- [a3-distrib/plots/1_attns3.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/1_attns3.png>)
- [a3-distrib/plots/2_attns0.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/2_attns0.png>)
- [a3-distrib/plots/2_attns1.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/2_attns1.png>)
- [a3-distrib/plots/2_attns2.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/2_attns2.png>)
- [a3-distrib/plots/2_attns3.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/2_attns3.png>)
- [a3-distrib/plots/3_attns0.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/3_attns0.png>)
- [a3-distrib/plots/3_attns1.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/3_attns1.png>)
- [a3-distrib/plots/3_attns2.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/3_attns2.png>)
- [a3-distrib/plots/3_attns3.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/3_attns3.png>)
- [a3-distrib/plots/4_attns0.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/4_attns0.png>)
- [a3-distrib/plots/4_attns1.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/4_attns1.png>)
- [a3-distrib/plots/4_attns2.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/4_attns2.png>)
- [a3-distrib/plots/4_attns3.png](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/plots/4_attns3.png>)
- [a3-distrib/requirements.txt](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/requirements.txt>)
- [a3-distrib/transformer.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/transformer.py>)
- [a3-distrib/transformer_lm.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/transformer_lm.py>)
- [a3-distrib/utils.py](</Users/zaidmaknojia/Projects/MSAI/NLP (AI 388)/a3/a3-distrib/utils.py>)
