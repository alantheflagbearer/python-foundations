# Roadmap to AI/ML using Python — From Scratch

A daily-practice log following a 52-week (365-day) roadmap from Python fundamentals through classical ML, deep learning, and into applied AI. Most lessons don't just call a library function — they build the underlying idea (gradient descent, backprop, convolution, dropout...) from scratch in NumPy first, then show the framework equivalent. Each day is a self-contained script; from Week 1 onward, most days add a paired study guide (a concept walkthrough and a line-by-line syntax breakdown, both PDF).

**Currently on Day 50** of 365. Full syllabus: [`full_syllabus_days_1_365.md`](full_syllabus_days_1_365.md).

## How this repo is organized

- **One folder per week** (`week-01-python-foundations/` … `week-08-classic-cnn-architectures/`) — each holds that week's scripts, study-guide PDFs, diagrams, and run outputs, plus its own `README.md` with a day-by-day index. Open a week's `README.md` first; it links to everything inside.
- **`practice-drills/`** — short standalone Python-fundamentals exercises (`pfdsNN.py`), numbered independently of the daily roadmap.
- Root-level `titanic.csv`, `titanic.db`, and `students.csv` are the shared datasets several early weeks use; each is also copied into every week folder whose scripts need it, so any single week folder runs standalone without reaching outside itself.

## Week-by-week index

| Week | Days | Topic | |
|---|---|---|---|
| 1 | 2–5 | Python Foundations | [→ folder](week-01-python-foundations) |
| 2 | 6–14 | NumPy, Pandas, Statistics & SQL | [→ folder](week-02-numpy-pandas-statistics-sql) |
| 3 | 15–20 | Intro to Machine Learning | [→ folder](week-03-intro-to-machine-learning) |
| 4 | 21–27 | Feature Engineering & Boosting | [→ folder](week-04-feature-engineering-boosting) |
| 5 | 28–35 | Interpretability, Tuning & Deployment | [→ folder](week-05-interpretability-tuning-deployment) |
| 6 | 36–41 | Neural Networks from Scratch | [→ folder](week-06-neural-networks-from-scratch) |
| 7 | 42–48 | CNNs I & II | [→ folder](week-07-cnns-i-and-ii) |
| 8 | 49–50 (of 49–55) | Classic CNN Architectures + Regularization — in progress | [→ folder](week-08-classic-cnn-architectures) |

Weeks 9 onward (RNNs, attention, Transformers, tokenization, a mini-GPT, and much further — see the full syllabus) haven't been built yet.

Starting with Week 6, the same small neural-network library — written entirely in NumPy, no framework — is extended week over week: the `net.forward()` / `net.backward()` core built on Day 36 is still what every later script, all the way through Day 50, trains against.

## Recent visuals

| | |
|---|---|
| ![LeNet-5 vs. baseline](week-07-cnns-i-and-ii/lenet_vs_baseline_loss.png) LeNet-5 vs. this series' baseline — Day 47 | ![Transfer strategies](week-07-cnns-i-and-ii/transfer_strategies_diagram.png) Five transfer-learning strategies compared — Day 48 |
| ![AlexNetStyle vs. baseline](week-08-classic-cnn-architectures/alexnet_vs_baseline_loss.png) AlexNet-style vs. this series' baseline — Day 49 | ![Dropout regularization comparison](week-08-classic-cnn-architectures/dropout_regularization_comparison.png) Dropout ON vs. OFF, fully converged — Day 50 |

## Featured project: Titanic EDA (Day 13)

Answers one question: what determined who survived the Titanic disaster, and does it hold up across pandas, statistical tests, and raw SQL?

**Dataset:** 891 passengers, 15 columns after cleaning and feature engineering (`FamilySize`, `IsAlone`, `AgeGroup`, `Title` added; age imputed with median, cabin dropped, embarked imputed with mode).

**Key findings:**
- Survivors paid $48.40 avg fare vs. $22.12 for non-survivors (t = 6.84, p = 2.70e-11) — statistically significant.
- Female survival rate 74.2% vs. male 18.9% (χ² = 260.7, p = 1.20e-58) — statistically significant.
- Survival rate by class: 1st 63.0%, 2nd 47.3%, 3rd 24.2% (χ² = 102.9, p = 4.55e-23) — statistically significant.
- Strongest correlate of `Survived` is `Pclass` (r = -0.34) — correlation, not causation: class is a proxy for cabin location and lifeboat access, not a direct cause.

![Titanic EDA dashboard](week-02-numpy-pandas-statistics-sql/day13_dashboard.png)

Six panels: overall survival, survival by sex, survival by class, age distribution, log-scaled fare distribution, and the full correlation matrix.

**SQL cross-check:** the same survival patterns are reproduced with raw SQL (`GROUP BY`, `CASE WHEN` fare-tier bucketing, and a `RANK()` window function for top fares per class) — see [`day13_eda_portfolio.py`](week-02-numpy-pandas-statistics-sql/day13_eda_portfolio.py).

## What's next

Continuing the roadmap day by day toward Week 53 (deep learning frameworks, deployment, and applied AI projects) — see the full syllabus for the complete week-by-week breakdown.
