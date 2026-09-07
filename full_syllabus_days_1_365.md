# MD Mutasim Billah — 52-Week Data Science → ML/AI Roadmap
## Full Syllabus, Day 1 to Day 365

Repo: github.com/alantheflagbearer/python-foundations

**Source of truth note:** Days 1–44 below are the *actual* finished lessons already
in the repo — each title is pulled directly from that day's own build script, not
reconstructed from memory. Days 45–365 are the drafted continuation from
`roadmap-daily-lesson/references/roadmap.md`; those are one-line starting points,
not fixed specs — the real content for each of those days still gets worked out
honestly when that day is actually built, the same way every one of Days 1–44 was.
This file is editable — if a week's theme or a day's topic should change, edit it
directly (and keep `roadmap.md` in sync for the skill that builds each day).

Weeks are not a clean 7-day grid throughout — a few weeks run short or long around
review/portfolio days, exactly as they actually happened. Each week's real day range
is stated in its heading.

---

## Week 1 (Days 1–5): Python Foundations

- **Day 1** — Variables and data types
- **Day 2** — Control flow: conditionals and loops
- **Day 3** — Functions
- **Day 4** — Data structures and comprehensions
- **Day 5** — File handling and exception handling

## Week 2 (Days 6–14): NumPy, Pandas, Statistics & SQL

- **Day 6** — NumPy arrays and indexing
- **Day 7** — NumPy aggregation, reshaping, and broadcasting
- **Day 8** — Pandas Series and DataFrames
- **Day 9** — Filtering, groupby, merging, and missing data
- **Day 10** — Statistics fundamentals
- **Day 11** — SQL basics
- **Day 12** — SQL intermediate
- **Day 13** — Full EDA portfolio project
- **Day 14** — Week 2 review

## Week 3 (Days 15–20): Intro to Machine Learning

- **Day 15** — Intro to machine learning
- **Day 16** — Decision trees and random forests
- **Day 17** — Cross-validation and hyperparameter tuning
- **Day 18** — ROC curves, AUC, and decision thresholds
- **Day 19** — Full classification portfolio project
- **Day 20** — Week 3 review (complete study guide)

## Week 4 (Days 21–27): Feature Engineering & Boosting

- **Day 21** — Feature engineering at scale: complete guide
- **Day 22** — Imputation and custom features inside the pipeline
- **Day 23** — Gradient boosting
- **Day 24** — Handling imbalanced classes
- **Day 25** — XGBoost — industry-standard gradient boosting
- **Day 26** — Ensemble methods: voting and stacking
- **Day 27** — Week 4 review — advanced ensemble techniques (complete guide)

## Week 5 (Days 28–35): Interpretability, Tuning & Deployment Basics

- **Day 28** — Model interpretability with SHAP
- **Day 29** — Visual SHAP: bar, beeswarm, dependence, waterfall
- **Day 30** — Partial dependence and ICE curves
- **Day 31** — RandomizedSearchCV: smarter hyperparameter search
- **Day 32** — Bayesian optimization with Optuna
- **Day 33** — Model persistence with joblib
- **Day 34** — Deploying a model behind a minimal FastAPI endpoint
- **Day 35** — Week 5 review

## Week 6 (Days 36–41): Neural Networks from Scratch (Beginner Edition)

- **Day 36** — Neural network fundamentals (beginner edition)
- **Day 37** — Loss functions and optimizers (beginner edition)
- **Day 38** — Regularization (beginner edition)
- **Day 39** — Softmax, cross-entropy, and mini-batch training (beginner edition)
- **Day 40** — Weight initialization and batch normalization (beginner edition)
- **Day 41** — Convolutional layers (beginner edition)

## Week 7 (Days 42–48): CNNs I & II

- **Day 42** — Deep CNNs: stacking conv+pool blocks (beginner edition)
- **Day 43** — Real RGB images + a three-block CNN (beginner edition)
- **Day 44** — "Same" padding + does depth or width actually win? (beginner edition)
- **Day 45** — Mini-batch gradient descent from scratch: shuffling, batch loops, real noise-vs-full-batch convergence comparison on the junction dataset
- **Day 46** — Learning rate schedules (step decay, cosine annealing); revisit `WideTwoBlockConvNetRGB`'s Day 44 non-convergence with a proper schedule instead of a fixed lr
- **Day 47** — Classic CNN architectures I: recreate a small LeNet-5-style network by hand on the junction dataset
- **Day 48** — Transfer learning concept: freezing early layers, fine-tuning a warm-started network; why it helps, with a real frozen-vs-unfrozen comparison

## Week 8 (Days 49–55): Classic CNN Architectures + Regularization

- **Day 49** — AlexNet-style deeper CNN (ReLU, dropout, larger images) with mini-batches
- **Day 50** — Dropout from scratch (forward/backward); real regularized-vs-unregularized comparison
- **Day 51** — L1/L2 weight regularization revisited on CNNs; real over/underfitting demonstration
- **Day 52** — VGG-style architecture: stacked 3x3 convs; why small stacked kernels beat one large kernel
- **Day 53** — Residual connections (a ResNet block) from scratch; solving vanishing gradients in deep nets, real gradient check
- **Day 54** — Batch normalization inside CNNs (extending Day 40's dense-layer batchnorm to conv layers)
- **Day 55** — Real image data pipeline efficiency: why "dataloader" abstractions exist, batching/shuffling patterns

## Week 9 (Days 56–62): Sequence Data & RNNs I

- **Day 56** — What is sequential data: a real time-series or text example; why feedforward/CNN nets fail on it
- **Day 57** — Vanilla RNN forward pass from scratch on a real toy sequence
- **Day 58** — RNN backward pass (BPTT) from scratch + gradient check
- **Day 59** — Vanishing/exploding gradients in RNNs: real demonstration on a long sequence
- **Day 60** — Gradient clipping from scratch; honest before/after on Day 59's demonstration
- **Day 61** — Many-to-one RNN classifier on a real short-text sentiment dataset
- **Day 62** — Many-to-many RNN (sequence tagging) on a real toy dataset

## Week 10 (Days 63–69): LSTM & GRU

- **Day 63** — The LSTM cell: gates explained, motivated directly from Day 59's vanishing-gradient problem
- **Day 64** — LSTM forward pass from scratch
- **Day 65** — LSTM backward pass from scratch + gradient check
- **Day 66** — GRU: a simpler gated alternative, from-scratch implementation
- **Day 67** — RNN vs LSTM vs GRU: real comparison on Week 9's dataset
- **Day 68** — Stacking + bidirectional RNNs from scratch
- **Day 69** — Sequence-to-sequence (encoder-decoder) from scratch on a toy copy/translation task

## Week 11 (Days 70–76): Attention Mechanism

- **Day 70** — The information-bottleneck problem in seq2seq: real demonstration of it failing on longer sequences
- **Day 71** — Additive (Bahdanau-style) attention from scratch
- **Day 72** — Attention-weight visualization: real alignment heatmap on the toy task
- **Day 73** — Dot-product (Luong) attention from scratch; real comparison to additive attention
- **Day 74** — Self-attention from scratch: attention within one sequence, not across encoder/decoder
- **Day 75** — Scaled dot-product attention; real numerical effect of the scaling factor
- **Day 76** — Multi-head attention from scratch

## Week 12 (Days 77–83): The Transformer

- **Day 77** — Positional encoding from scratch: injecting order without recurrence
- **Day 78** — Transformer encoder block from scratch (self-attention + FFN + residual + layernorm)
- **Day 79** — LayerNorm vs BatchNorm: real comparison, why transformers prefer LayerNorm
- **Day 80** — Transformer decoder block + masked self-attention
- **Day 81** — Full transformer forward pass on a toy copy/translation task
- **Day 82** — Transformer backward pass + gradient check across the whole stack
- **Day 83** — Training a tiny transformer end-to-end; honest real results

## Week 13 (Days 84–90): Tokenization & Embeddings

- **Day 84** — Byte-pair encoding (BPE) from scratch on real text
- **Day 85** — Word embeddings: one-hot vs dense, real vocabulary example
- **Day 86** — Word2Vec (skip-gram) from scratch on a real small corpus
- **Day 87** — Embedding visualization (PCA/t-SNE); real semantic-cluster check
- **Day 88** — Subword embeddings and out-of-vocabulary handling
- **Day 89** — Pretrained embeddings (GloVe-style): loading and using real vectors
- **Day 90** — Building a tiny character-level GPT-style model, combining Weeks 11–13

## Week 14 (Days 91–97): Mini-GPT & Language Modeling

- **Day 91** — Causal language modeling: next-token prediction from scratch
- **Day 92** — Training the mini-GPT on real text; honest perplexity numbers
- **Day 93** — Sampling strategies from scratch: greedy, temperature, top-k, top-p
- **Day 94** — Real generated text from the mini-GPT; honest quality assessment
- **Day 95** — Scaling-laws intuition: real experiment varying model size/data on the mini-GPT
- **Day 96** — Fine-tuning the mini-GPT on a narrower real task
- **Day 97** — Language-model evaluation metrics from scratch: perplexity, BLEU, ROUGE

## Week 15 (Days 98–104): Practical NLP with Pretrained Models

- **Day 98** — Intro to a pretrained-transformers library; using a real pretrained model
- **Day 99** — Fine-tuning a BERT-style model on a real classification dataset
- **Day 100** — Named entity recognition with a pretrained model on a real dataset **(milestone day)**
- **Day 101** — Text summarization: extractive vs abstractive, real example
- **Day 102** — Question answering with a pretrained transformer on a real dataset
- **Day 103** — Semantic search with embeddings: real vector-similarity search
- **Day 104** — Building a tiny retrieval-augmented generation (RAG) pipeline on real documents

## Week 16 (Days 105–111): Classical ML I — Trees

- **Day 105** — Decision trees from scratch: entropy/Gini on a real dataset
- **Day 106** — Decision-tree pruning: real over/underfitting demonstration
- **Day 107** — Random forests from scratch: bagging, real variance-reduction demo
- **Day 108** — Feature importance: real permutation importance vs impurity-based
- **Day 109** — Gradient boosting from scratch (simplified): real sequential error correction
- **Day 110** — A production boosting library in practice; real comparison to the from-scratch version
- **Day 111** — Trees vs neural nets: real head-to-head on tabular data

## Week 17 (Days 112–118): Classical ML II — Kernels & Ensembles

- **Day 112** — Support vector machines from scratch on real linearly separable and non-separable data
- **Day 113** — The kernel trick from scratch: real RBF-kernel demonstration
- **Day 114** — Naive Bayes from scratch: real text classification
- **Day 115** — k-Nearest Neighbors from scratch: real curse-of-dimensionality demonstration
- **Day 116** — Stacking and blending ensembles: real meta-model demo
- **Day 117** — Handling imbalanced data: real SMOTE vs class-weighting comparison
- **Day 118** — Model selection and cross-validation strategies: real k-fold, stratified, and time-series splits

## Week 18 (Days 119–125): Unsupervised Learning I

- **Day 119** — k-Means from scratch: real elbow-method demonstration
- **Day 120** — Hierarchical clustering from scratch: real dendrogram
- **Day 121** — DBSCAN from scratch: real density-based clustering on non-convex data
- **Day 122** — Gaussian mixture models + EM algorithm from scratch
- **Day 123** — PCA from scratch: real variance-explained demonstration
- **Day 124** — SVD and its relationship to PCA: real matrix-factorization demo
- **Day 125** — t-SNE and UMAP: real high-dimensional visualization comparison

## Week 19 (Days 126–132): Unsupervised Learning II — Autoencoders

- **Day 126** — Autoencoders from scratch: real reconstruction demo
- **Day 127** — Denoising autoencoders: real noise-removal demonstration
- **Day 128** — Anomaly detection with autoencoders: real reconstruction-error thresholding
- **Day 129** — Variational autoencoders (VAE): the reparameterization trick from scratch
- **Day 130** — VAE training + real latent-space visualization
- **Day 131** — Contrastive learning basics: real positive/negative pair demonstration
- **Day 132** — Self-supervised learning intuition: real pretext-task demo

## Week 20 (Days 133–139): Generative Models — GANs

- **Day 133** — Generative vs discriminative models: real conceptual comparison
- **Day 134** — A GAN from scratch: generator + discriminator, real minimax training loop
- **Day 135** — GAN training instability: real, honest mode-collapse demonstration
- **Day 136** — A DCGAN-style convolutional GAN generating real images on the junction dataset
- **Day 137** — Conditional GANs: real class-conditional generation
- **Day 138** — Evaluating generative models: real FID/IS-style metric computation
- **Day 139** — Diffusion models intuition: forward noising process from scratch, real demonstration

## Week 21 (Days 140–146): Diffusion Models & Modern Generative AI

- **Day 140** — Reverse diffusion process from scratch: real denoising step
- **Day 141** — Training a tiny diffusion model: real toy image generation
- **Day 142** — GANs vs VAEs vs diffusion: real, honest quality/speed tradeoffs
- **Day 143** — Text-to-image models conceptually: CLIP-style joint embeddings, real demo
- **Day 144** — Prompt engineering for generative models: real experimentation
- **Day 145** — Ethical considerations in generative AI: bias, deepfakes, real case studies
- **Day 146** — Capstone mini-project: a small generative demo end-to-end

## Week 22 (Days 147–153): Time Series Analysis

- **Day 147** — Time-series decomposition: trend/seasonality/residual on a real dataset
- **Day 148** — Stationarity and differencing: real ADF-test demonstration
- **Day 149** — ARIMA from scratch (simplified): real forecasting demo
- **Day 150** — Exponential smoothing methods: real comparison to ARIMA
- **Day 151** — Feature engineering for time series (lags, rolling stats) on a real dataset
- **Day 152** — RNN/LSTM for time-series forecasting: real comparison to classical methods
- **Day 153** — Decomposable forecasting (Prophet-style): real practical example

## Week 23 (Days 154–160): Recommender Systems

- **Day 154** — Collaborative filtering (user-based and item-based) on a real ratings matrix
- **Day 155** — Matrix factorization from scratch: real SVD-based recommender
- **Day 156** — Content-based filtering: real feature-based recommendations
- **Day 157** — Hybrid recommender systems: real combined approach
- **Day 158** — Implicit feedback and real cold-start challenges
- **Day 159** — Evaluating recommenders: real precision@k, recall@k, NDCG
- **Day 160** — Deep learning for recommendations: real neural collaborative filtering

## Week 24 (Days 161–167): Model Evaluation & Statistics Deep Dive

- **Day 161** — Bias-variance tradeoff revisited with real learning curves
- **Day 162** — Hypothesis testing for ML: real A/B-test analysis
- **Day 163** — Confidence intervals and bootstrapping: real resampling demonstration
- **Day 164** — The multiple-comparisons problem: real, honest p-hacking-risk demonstration
- **Day 165** — Calibration of probabilistic predictions: real reliability diagrams
- **Day 166** — Fairness metrics in ML: real demographic-parity/equalized-odds demonstration
- **Day 167** — Explainability: SHAP and LIME from scratch (simplified), real feature attribution

## Week 25 (Days 168–174): Hyperparameter Tuning & Experimentation

- **Day 168** — Grid search vs random search: real efficiency comparison
- **Day 169** — Bayesian optimization from scratch (simplified): real hyperparameter search
- **Day 170** — Learning-rate finders: real practical technique demonstration
- **Day 171** — Early-stopping strategies: real overfitting-prevention demo
- **Day 172** — Cross-validation for time series and grouped data: real leakage demonstration
- **Day 173** — Experiment-tracking practices: real logging/comparison workflow
- **Day 174** — Reproducibility in ML: real seed-control and environment-pinning demonstration

## Week 26 (Days 175–181): MLOps I — Serving Models

- **Day 175** — Saving/loading models: real serialization-format comparison
- **Day 176** — Building a REST API for a model: a real working endpoint
- **Day 177** — Containerizing a model: a real build-and-run demonstration
- **Day 178** — Batch vs real-time inference: real latency/throughput comparison
- **Day 179** — Model versioning: a real practical workflow
- **Day 180** — A/B testing models in production: real conceptual + simulated demonstration
- **Day 181** — Monitoring model performance in production: real drift-detection demo

## Week 27 (Days 182–188): MLOps II — Pipelines & Scaling

- **Day 182** — Data pipelines: a real ETL-style example
- **Day 183** — Feature stores: real conceptual walkthrough with a toy implementation
- **Day 184** — Distributed-training basics: real data-parallel simulation
- **Day 185** — Mixed-precision training: real speed/memory tradeoff demonstration
- **Day 186** — Model quantization: real size/speed/accuracy tradeoff demonstration
- **Day 187** — Model pruning: real sparsity demonstration
- **Day 188** — Knowledge distillation: real teacher-student demonstration

## Week 28 (Days 189–195): Computer Vision Beyond Classification

- **Day 189** — Object detection intuition: real bounding-box demonstration
- **Day 190** — IoU and non-max suppression from scratch
- **Day 191** — A simplified YOLO-style detector: real toy implementation
- **Day 192** — Semantic segmentation intuition: real pixel-wise classification demo
- **Day 193** — A simplified U-Net from scratch: real segmentation demonstration
- **Day 194** — Instance segmentation concepts: real comparison to semantic segmentation
- **Day 195** — Image similarity and retrieval: real embedding-based search

## Week 29 (Days 196–202): Advanced Computer Vision

- **Day 196** — Advanced augmentation strategies (mixup, cutmix): real demonstration
- **Day 197** — Vision transformers (ViT) intuition: real patch-embedding demonstration
- **Day 198** — ViT from scratch (simplified): real training on the junction dataset
- **Day 199** — CNNs vs ViTs: real, honest tradeoffs on real data
- **Day 200** — Self-supervised vision pretraining (SimCLR-style): real contrastive demo **(milestone day)**
- **Day 201** — Transfer learning with pretrained vision models: real fine-tuning workflow
- **Day 202** — Explainability for vision models: a real Grad-CAM implementation

## Week 30 (Days 203–209): Reinforcement Learning I

- **Day 203** — What is reinforcement learning: a real toy environment setup
- **Day 204** — Markov decision processes: real value/policy demonstration
- **Day 205** — Q-learning from scratch: real toy grid-world demonstration
- **Day 206** — Deep Q-Networks (DQN): a real simplified implementation
- **Day 207** — Policy-gradient methods from scratch: real REINFORCE demonstration
- **Day 208** — Actor-critic methods: a real simplified implementation
- **Day 209** — Exploration vs exploitation: real epsilon-greedy vs UCB comparison

## Week 31 (Days 210–216): Reinforcement Learning II & RLHF

- **Day 210** — Reward shaping: real demonstration of its effects
- **Day 211** — Proximal policy optimization (PPO) intuition: a real simplified demo
- **Day 212** — Multi-armed bandits: a real practical application
- **Day 213** — RLHF conceptually: how it fine-tunes language models, real toy demonstration
- **Day 214** — A tiny RLHF-style loop on the Week 14 mini-GPT
- **Day 215** — Simulation environments: a real toy game/environment demo
- **Day 216** — Capstone: training an RL agent on a real toy game end-to-end

## Week 32 (Days 217–223): Graph & Structured Data

- **Day 217** — Graph representations: real adjacency-matrix/list demonstration
- **Day 218** — Graph traversal algorithms in an ML context: real practical example
- **Day 219** — Node embeddings (DeepWalk/node2vec intuition): real small-graph demo
- **Day 220** — Graph neural networks from scratch (simplified): real node classification
- **Day 221** — Message-passing intuition: real step-by-step demonstration
- **Day 222** — Applications of GNNs: a real recommendation or social-network example
- **Day 223** — Knowledge graphs: a real triple-store demonstration

## Week 33 (Days 224–230): Big Data & Data Engineering for ML

- **Day 224** — Working with data too large for memory: real chunked-processing demonstration
- **Day 225** — SQL for ML practitioners: real query-writing practice
- **Day 226** — Distributed data-processing concepts: a real local simulation
- **Day 227** — Data versioning and lineage: a real practical workflow
- **Day 228** — Data quality and validation: real schema-checking demonstration
- **Day 229** — Feature engineering at scale: real efficient-pipeline demonstration
- **Day 230** — Streaming-data basics: real windowed-aggregation demonstration

## Week 34 (Days 231–237): Responsible & Ethical AI

- **Day 231** — Bias in ML systems: real dataset-bias demonstration
- **Day 232** — Privacy-preserving ML: real differential-privacy demonstration (simplified)
- **Day 233** — Federated learning intuition: a real simplified simulation
- **Day 234** — Adversarial examples: a real attack demonstration on a trained network
- **Day 235** — Adversarial training/defenses: real robustness-improvement demonstration
- **Day 236** — Model cards and documentation practices: a real example
- **Day 237** — The AI governance and regulation landscape: a real, non-partisan overview

## Week 35 (Days 238–244): Advanced Optimization

- **Day 238** — Second-order optimization intuition (Newton's method): real comparison to gradient descent
- **Day 239** — Natural gradient / Fisher information intuition: a real simplified demo
- **Day 240** — Adam, AdamW, and weight decay: a real, honest comparison
- **Day 241** — Learning-rate warmup + cosine schedules: real training-stability demonstration
- **Day 242** — Gradient accumulation: real large-effective-batch-size demonstration
- **Day 243** — Curriculum learning: real ordered-vs-random training comparison
- **Day 244** — Meta-learning intuition (learning to learn): a real simplified demonstration

## Week 36 (Days 245–251): Probabilistic ML

- **Day 245** — Bayesian inference basics: real posterior-update demonstration
- **Day 246** — Bayesian linear regression from scratch: real uncertainty quantification
- **Day 247** — Gaussian processes intuition: a real simplified regression demonstration
- **Day 248** — Markov chain Monte Carlo (MCMC) basics: real sampling demonstration
- **Day 249** — Variational inference intuition: a real simplified demonstration
- **Day 250** — Bayesian neural networks conceptually: real uncertainty-estimation demo
- **Day 251** — Probabilistic graphical models intuition: a real small Bayes-net example

## Week 37 (Days 252–258): Applied Project I — Tabular/Business Data

- **Day 252** — Project kickoff: a real business dataset, problem framing
- **Day 253** — EDA and cleaning on the real project dataset
- **Day 254** — Feature engineering for the project dataset
- **Day 255** — Baseline model + classical ML comparison on the project
- **Day 256** — A neural-network approach on the project dataset
- **Day 257** — Model evaluation and error analysis on the project
- **Day 258** — Project writeup and presentation: a real deliverable

## Week 38 (Days 259–265): Applied Project II — Computer Vision

- **Day 259** — Project kickoff: a real image dataset, problem framing
- **Day 260** — Data preprocessing/augmentation pipeline for the project
- **Day 261** — Model architecture selection and training for the project
- **Day 262** — Transfer learning application on the project
- **Day 263** — Model evaluation and error analysis (confusion matrix, misclassified examples)
- **Day 264** — Deployment of the vision project (a simple API/demo)
- **Day 265** — Project writeup and presentation

## Week 39 (Days 266–272): Applied Project III — NLP/Chatbot

- **Day 266** — Project kickoff: a real text dataset or chatbot use case
- **Day 267** — Data preprocessing and tokenization pipeline for the project
- **Day 268** — Model selection (fine-tuning vs from-scratch) for the project
- **Day 269** — Building a retrieval or generation component
- **Day 270** — Evaluation of the NLP project: real qualitative + quantitative
- **Day 271** — Deployment of the NLP project (a simple API/demo)
- **Day 272** — Project writeup and presentation

## Week 40 (Days 273–279): Applied Project IV — End-to-End ML System

- **Day 273** — System design for a full ML product: a real architecture diagram
- **Day 274** — Data-pipeline implementation for the system
- **Day 275** — Model-training pipeline implementation
- **Day 276** — Serving-layer implementation
- **Day 277** — Monitoring and retraining-strategy implementation
- **Day 278** — Load testing and honest performance limits
- **Day 279** — Full system writeup and retrospective

## Week 41 (Days 280–286): Interview Preparation I — ML Fundamentals Review

- **Day 280** — Review: linear/logistic regression, gradient descent — real mock-interview Q&A
- **Day 281** — Review: neural network fundamentals — real mock-interview Q&A
- **Day 282** — Review: CNNs and computer vision — real mock-interview Q&A
- **Day 283** — Review: RNNs, LSTMs, attention, transformers — real mock-interview Q&A
- **Day 284** — Review: classical ML (trees, SVMs, clustering) — real mock-interview Q&A
- **Day 285** — Review: statistics and probability for ML — real mock-interview Q&A
- **Day 286** — Review: model evaluation and MLOps — real mock-interview Q&A

## Week 42 (Days 287–293): Interview Preparation II — Coding & System Design

- **Day 287** — ML coding-interview practice: real from-scratch implementation under time pressure
- **Day 288** — ML system-design interview practice: a real case-study walkthrough
- **Day 289** — SQL interview practice for data roles: real query problems
- **Day 290** — Take-home-assignment simulation: a real end-to-end mini-project under constraints
- **Day 291** — Behavioral interview prep for DS/ML roles: real STAR-method practice
- **Day 292** — Portfolio and resume review: real project narrative-building
- **Day 293** — Mock interview day: a real, full simulated interview

## Week 43 (Days 294–300): Large Language Models Deep Dive

- **Day 294** — LLM architecture review: a real walkthrough tying back to Weeks 12–14
- **Day 295** — Instruction tuning: a real simplified demonstration
- **Day 296** — RLHF for LLMs revisited in depth: a real simplified pipeline
- **Day 297** — In-context learning and few-shot prompting: real experimentation
- **Day 298** — Chain-of-thought prompting: real, honest comparison of prompting strategies
- **Day 299** — Retrieval-augmented generation (RAG) revisited in depth: a real production-style pipeline
- **Day 300** — Evaluating LLM outputs: real rubric-based and automated evaluation **(milestone day)**

## Week 44 (Days 301–307): AI Agents & Tool Use

- **Day 301** — What makes an "agent": a real conceptual framework
- **Day 302** — Tool-calling / function-calling from scratch (simplified): a real demonstration
- **Day 303** — ReAct-style reasoning+acting loops: a real simplified implementation
- **Day 304** — Multi-step planning for agents: a real toy-task demonstration
- **Day 305** — Memory for agents: a real simplified implementation
- **Day 306** — Multi-agent systems intuition: a real toy collaboration demonstration
- **Day 307** — Building a small real agent end-to-end

## Week 45 (Days 308–314): Efficient & Scaled Deep Learning

- **Day 308** — Model parallelism vs data parallelism: real conceptual + simulated demo
- **Day 309** — Gradient checkpointing: real memory/speed tradeoff demonstration
- **Day 310** — LoRA and parameter-efficient fine-tuning from scratch (simplified)
- **Day 311** — Quantization for LLMs: real size/quality tradeoff demonstration
- **Day 312** — Efficient attention variants (sparse/linear attention): a real simplified demo
- **Day 313** — Caching strategies for inference (KV cache): a real demonstration
- **Day 314** — Benchmarking model efficiency: a real, honest measurement methodology

## Week 46 (Days 315–321): Specialized Domains

- **Day 315** — ML for healthcare: a real case study + honest ethical considerations
- **Day 316** — ML for finance: a real case study (e.g. fraud detection)
- **Day 317** — ML for recommendation at scale: a real case study
- **Day 318** — ML for search/ranking: a real learning-to-rank demonstration
- **Day 319** — Audio ML basics: real spectrogram + simple classifier demonstration
- **Day 320** — Speech recognition intuition: a real simplified demonstration
- **Day 321** — Multimodal models (vision+language) intuition: a real simplified demonstration

## Week 47 (Days 322–328): Research Skills

- **Day 322** — Reading ML papers effectively: a real paper walkthrough
- **Day 323** — Reproducing a paper's core result (simplified): a real from-scratch reimplementation
- **Day 324** — Ablation studies: a real demonstration on a from-scratch model
- **Day 325** — Writing up ML experiments clearly: real report-writing practice
- **Day 326** — Statistical rigor in ML research: real significance-testing demonstration
- **Day 327** — Open-source contribution practices: a real workflow walkthrough
- **Day 328** — Staying current: real practice building a personal paper-reading pipeline

## Week 48 (Days 329–335): Capstone Project I — Planning & Data

- **Day 329** — Capstone kickoff: real problem selection and scoping
- **Day 330** — Capstone data collection/sourcing: a real dataset assembly
- **Day 331** — Capstone EDA: a real exploratory analysis
- **Day 332** — Capstone baseline model: a real first working version
- **Day 333** — Capstone architecture design: real decision-making, documented
- **Day 334** — Capstone initial training run: real, honest first results
- **Day 335** — Capstone mid-point review: a real, honest assessment of what's working and what isn't

## Week 49 (Days 336–342): Capstone Project II — Iteration

- **Day 336** — Capstone error analysis: a real failure-mode investigation
- **Day 337** — Capstone feature/architecture improvements: real iteration
- **Day 338** — Capstone hyperparameter tuning: real search results
- **Day 339** — Capstone regularization/robustness pass: real, honest tradeoffs
- **Day 340** — Capstone evaluation against a real held-out benchmark
- **Day 341** — Capstone ablation studies: real component-by-component analysis
- **Day 342** — Capstone second full training run: real final-candidate results

## Week 50 (Days 343–349): Capstone Project III — Deployment & Polish

- **Day 343** — Capstone deployment: a real working demo/API
- **Day 344** — Capstone monitoring setup: real logging/metrics
- **Day 345** — Capstone documentation: a real README and usage guide
- **Day 346** — Capstone testing: real unit/integration tests for the ML system
- **Day 347** — Capstone performance optimization: a real, honest before/after
- **Day 348** — Capstone final polish: a real UI/UX or output-quality pass
- **Day 349** — Capstone final review: a real end-to-end walkthrough

## Week 51 (Days 350–356): Portfolio & Career Prep

- **Day 350** — Portfolio website: a real project-showcase build
- **Day 351** — Technical blog post: a real write-up of a favorite lesson from the roadmap
- **Day 352** — LinkedIn/resume optimization for DS/ML roles: real practice
- **Day 353** — Networking strategy for DS/ML roles: a real practical plan
- **Day 354** — Salary negotiation basics for tech roles: a real practical overview
- **Day 355** — Final mock interview: a real, full simulated round
- **Day 356** — Career roadmap planning: a real personal next-steps plan

## Week 52 (Days 357–363): Full Roadmap Review

- **Day 357** — Review: Weeks 1–7 (foundations through CNNs)
- **Day 358** — Review: Weeks 8–13 (RNNs through embeddings)
- **Day 359** — Review: Weeks 14–20 (LLMs through generative models)
- **Day 360** — Review: Weeks 21–27 (classical ML through MLOps)
- **Day 361** — Review: Weeks 28–36 (advanced CV/RL/probabilistic ML)
- **Day 362** — Review: Weeks 37–44 (applied projects through LLM depth)
- **Day 363** — Review: Weeks 45–51 (efficiency, research, capstone, career)

## Week 53 (Days 364–365): Journey Retrospective & What's Next

- **Day 364** — Full-journey retrospective: a real reflection on growth from Day 1 to Day 363, with concrete before/after comparisons (e.g. Day 1's first model vs the capstone)
- **Day 365** — Closing day: a real synthesis project or personal-choice topic, plus a concrete plan for continued learning beyond Day 365

---

### At a glance

| Range | Theme |
|---|---|
| Days 1–14 | Python, NumPy/Pandas, statistics, SQL |
| Days 15–35 | Classical ML: models, tuning, interpretability, deployment basics |
| Days 36–55 | Neural networks from scratch through classic CNN architectures |
| Days 56–90 | RNNs → LSTM/GRU → attention → the Transformer → embeddings |
| Days 91–104 | Mini-GPT, language modeling, practical NLP with pretrained models |
| Days 105–132 | Classical ML II (kernels/ensembles) + unsupervised learning + autoencoders |
| Days 133–146 | Generative models: GANs and diffusion |
| Days 147–174 | Time series, recommenders, evaluation/statistics, hyperparameter tuning |
| Days 175–188 | MLOps: serving, pipelines, scaling |
| Days 189–202 | Computer vision beyond classification (detection, segmentation, ViTs) |
| Days 203–216 | Reinforcement learning + RLHF |
| Days 217–237 | Graphs/structured data, big data engineering, responsible AI |
| Days 238–251 | Advanced optimization + probabilistic ML |
| Days 252–279 | Four applied end-to-end projects (tabular, vision, NLP, full system) |
| Days 280–293 | Interview preparation (fundamentals, coding, system design) |
| Days 294–314 | LLM deep dive, AI agents, efficient/scaled deep learning |
| Days 315–328 | Specialized domains + research skills |
| Days 329–356 | Capstone project (plan → iterate → deploy) + career prep |
| Days 357–365 | Full roadmap review + retrospective + what's next |
