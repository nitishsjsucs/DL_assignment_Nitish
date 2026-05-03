# Video Script: Important Classification Metrics (and Regression Metrics)

**Notebook:** `final_important_classification_metrics_tutorial.ipynb`  
**Author:** Nitish Chowdary Ratakonda  
**Estimated read time:** ~35–45 minutes  
**Cells:** 162 total, grouped into 18 Tasks

> This notebook is very large, so the script is organized **Task by Task**, with one line per cell so you can narrate as you scroll. Read the Task intros at normal speed, then a sentence or two per cell, then pause on the outputs.

---

## Intro

Hey everyone, I'm Nitish. This is the largest notebook in my deep-learning portfolio — a comprehensive guide to **classification metrics** (and toward the end, regression metrics too). Accuracy alone almost never tells the full story — we'll cover confusion matrices, precision, recall, F1, ROC-AUC, PR curves, multi-class, multi-label, imbalanced data, MLOps considerations, and even threshold calibration. I'll go task by task, and within each task I'll walk through every cell. Let's start.

---

# Task 1: Setup & The Classification Problem
*(Cells 0–13, plus the Task-1 objective in Cell 15)*

**Task intro:** First we frame the classification problem, build a synthetic medical-diagnosis dataset, and train a simple Logistic Regression baseline that the rest of the notebook will dissect.

- **Cell 0 — markdown:** The Colab badge. Just a link.
- **Cell 1 — markdown:** *"What is Classification?"* — defines it as predicting a categorical label, with a table of real-world examples: medical, email spam, e-commerce, self-driving.
- **Cell 2 — markdown:** *"Types of Classification"* — binary (two classes), multi-class (one label from many), and multi-label (multiple labels at once).
- **Cell 3 — markdown:** *"Environment Setup"* heading — tells us we're about to install libraries.
- **Cell 4 — code:** Imports NumPy, Pandas, scikit-learn's dataset tools, train/test split, LogisticRegression, and all the metric functions we'll use later. Prints library versions.
- **Cell 5 — markdown:** *"Creating Our Dataset"* — we'll synthesize a medical-diagnosis binary dataset: class 0 healthy, class 1 has the condition.
- **Cell 6 — code:** Sets the random seed to 42, calls `make_classification` for 1000 samples with 10 features and a mild class imbalance. Builds a DataFrame with proper column names. Prints shape and class distribution.
- **Cell 7 — code:** Plots the class distribution — a bar chart showing roughly 500 healthy vs 500 condition, and a pie chart showing the percentage split.
- **Cell 8 — markdown:** *"Train-Test Split"* — explains why we split data: training set for fitting, test set for honest evaluation.
- **Cell 9 — code:** Calls `train_test_split` with 80/20 ratio and stratification to preserve class balance. Prints set sizes and class counts in each set.
- **Cell 10 — markdown:** *"Training Our First Classifier"* — we'll use Logistic Regression because it's simple and outputs probabilities.
- **Cell 11 — code:** Instantiates `LogisticRegression`, fits on training data, and produces both hard predictions (`y_pred`) and probability scores (`y_pred_proba`).
- **Cell 12 — code:** Builds a results DataFrame showing Actual, Predicted, Probability, and Correct columns for the first 15 test samples. Prints a readable table so you can see exactly what the model is doing sample-by-sample.
- **Cell 13 — markdown:** Task 1 summary — we set up the problem, trained a baseline, and got both predictions and probabilities. The next task dissects those predictions.
- **Cell 14 — markdown:** Transition divider — introduces Task 2 (confusion matrix).
- **Cell 15 — markdown:** Task 1 learning objectives — note this cell is out of narrative order in the notebook but still belongs to Task 1.

---

# Task 2: The Confusion Matrix — Foundation of All Metrics
*(Cells 16–34)*

**Task intro:** The confusion matrix is the 2×2 table that every classification metric is derived from. We'll name all four quadrants, compute it, visualize it, and discuss the real-world impact of each error type.

- **Cell 16 — markdown:** Big heading: *"Classification Metrics: A Complete Progressive Guide."*
- **Cell 17 — markdown:** *"What is a Confusion Matrix?"* — introduces the 2×2 table of predicted vs actual.
- **Cell 18 — markdown:** *"The Four Fundamental Outcomes"* — defines TP, TN, FP, FN with color-coded cards for each.
- **Cell 19 — markdown:** *"Memory Trick"* — how to remember TP, TN, FP, FN: the first letter is whether the prediction was correct, the second letter is what was predicted.
- **Cell 20 — markdown:** *"Computing the Confusion Matrix"* — we'll use sklearn's `confusion_matrix`.
- **Cell 21 — code:** Imports and calls `confusion_matrix(y_test, y_pred)`. Prints the raw 2×2 matrix.
- **Cell 22 — code:** Extracts TN, FP, FN, TP from `cm.ravel()`. Prints each value with an emoji indicator and computes totals like correct = TP + TN and wrong = FP + FN.
- **Cell 23 — markdown:** *"Visualizing the Confusion Matrix"* — intro for the nice heatmap.
- **Cell 24 — code:** Defines `plot_confusion_matrix_detailed`, which uses seaborn to draw an annotated heatmap with both raw counts and percentages per cell. Calls it on our matrix.
- **Cell 25 — markdown:** *"Type I and Type II Errors"* — connects FP and FN to the classic statistical terminology: Type I is false positive, Type II is false negative.
- **Cell 26 — markdown:** *"Real-World Impact"* — a table of scenarios showing when FN is worse (cancer screening) vs when FP is worse (spam, drug tests, hiring).
- **Cell 27 — markdown:** *"Analyzing Our Model's Errors"* — we'll now look at concrete mistakes.
- **Cell 28 — code:** Builds a results DataFrame tagging each test sample as TP, TN, FP, or FN. Prints the counts of each outcome type.
- **Cell 29 — code:** Prints a few example cases of each outcome type — probabilities and their actual/predicted labels.
- **Cell 30 — code:** Histogram of probability scores split by outcome type. Shows that errors cluster around probability 0.5 — the model's "uncertain" zone.
- **Cell 31 — markdown:** *"The Math"* — fundamental relationships: TP+TN+FP+FN equals total; TP+FN is actual positives; TP+FP is predicted positives.
- **Cell 32 — code:** Verifies those relationships numerically and prints the equations with checkmarks.
- **Cell 33 — markdown:** Another title recap block (formatting artifact).
- **Cell 34 — markdown:** Task 2 summary — you now understand the confusion matrix, the four outcomes, and the asymmetric real-world costs of the two error types.

---

# Task 3: Accuracy & Its Limitations
*(Cells 35–47)*

**Task intro:** Accuracy is the most intuitive metric, but it becomes dangerous on imbalanced datasets. We'll demonstrate the **accuracy paradox** and show how **balanced accuracy** fixes it.

- **Cell 35 — markdown:** Task 3 title and learning objectives.
- **Cell 36 — markdown:** *"What is Accuracy?"* — defined as (TP+TN)/total. Simple and intuitive, but can be misleading.
- **Cell 37 — markdown:** *"Calculating Accuracy"* — intro to the computation.
- **Cell 38 — code:** Computes accuracy three ways: manually from the confusion matrix, using `accuracy_score`, and printing the fraction correct vs total. All three agree.
- **Cell 39 — markdown:** *"When Accuracy Works Well"* — on balanced datasets with equal error costs.
- **Cell 40 — code:** Generates a perfectly balanced 50/50 dataset, trains a model, and computes accuracy. Prints that with balanced classes accuracy is a reasonable metric.
- **Cell 41 — markdown:** *"The Accuracy Paradox"* — a warning: on imbalanced data, a dumb model can get high accuracy.
- **Cell 42 — code:** Creates an extreme 99% / 1% imbalanced fraud-detection dataset. Builds a dummy model that always predicts the majority class. Prints its 99% accuracy — which is worthless because it catches zero fraud.
- **Cell 43 — code:** Visualizes the paradox — bar chart comparing the 99%-accuracy dummy model to a smarter model on multiple metrics. Dummy wins on accuracy, loses on everything else.
- **Cell 44 — markdown:** *"Balanced Accuracy"* — a better alternative, defined as the average of per-class recalls.
- **Cell 45 — code:** Computes balanced accuracy for both the dummy and smart model. Dummy gets 50% (same as random guessing), smart gets something much higher. Shows balanced accuracy correctly ranks the models.
- **Cell 46 — markdown:** *"When to Use and Not Use Accuracy"* — side-by-side guide: use accuracy when classes are balanced and errors are symmetric; avoid when imbalanced or errors are asymmetric.
- **Cell 47 — markdown:** Task 3 summary — accuracy is intuitive but dangerous; always check class balance first.

---

# Task 4: Precision & Recall
*(Cells 48–61)*

**Task intro:** The two most important metrics beyond accuracy. Precision: when I predict positive, am I right? Recall: did I find all the positives? They trade off against each other.

- **Cell 48 — markdown:** Task 4 title and learning objectives.
- **Cell 49 — markdown:** *"Precision"* — defined as TP / (TP+FP). Answers: "of all my positive predictions, what fraction were correct?"
- **Cell 50 — code:** Computes precision manually and via `precision_score`. Prints the value and an interpretation.
- **Cell 51 — markdown:** *"Recall"* — defined as TP / (TP+FN). Answers: "of all actual positives, how many did I catch?"
- **Cell 52 — code:** Computes recall manually and via `recall_score`. Prints the value and its meaning for our medical context.
- **Cell 53 — markdown:** *"Precision vs Recall side-by-side"* — cards comparing them visually.
- **Cell 54 — code:** Visualization comparing precision and recall — bar charts, and a Venn-diagram-style figure showing what TP, FP, and FN represent relative to each metric.
- **Cell 55 — markdown:** *"When to prioritize which"* — high precision for spam filters (FP hurts users), high recall for medical screening (FN kills).
- **Cell 56 — markdown:** *"The precision-recall trade-off in action"* — by shifting the decision threshold, you can trade one for the other.
- **Cell 57 — code:** Sweeps the classification threshold from 0 to 1. At each threshold, plots precision and recall. Shows the classic inverse relationship: as threshold rises, precision rises and recall falls.
- **Cell 58 — code:** A more detailed threshold sweep with annotations showing specific threshold choices: conservative, balanced, aggressive.
- **Cell 59 — markdown:** *"The Precision-Recall Curve"* — the canonical way to visualize this trade-off.
- **Cell 60 — code:** Plots precision vs recall as a curve (not against threshold). Marks the current operating point. This is one of the most useful diagnostic plots.
- **Cell 61 — markdown:** Task 4 summary — precision and recall characterize the two failure modes; always report both.

---

# Task 5: F1, F2, and Fβ Scores
*(Cells 62–73)*

**Task intro:** When you need a single number, you combine precision and recall via the F-score. F1 weighs them equally; Fβ lets you emphasize one over the other.

- **Cell 62 — markdown:** Task 5 title and objectives.
- **Cell 63 — markdown:** *"What is the F1 Score?"* — the harmonic mean of precision and recall, punishing imbalance between them.
- **Cell 64 — code:** Computes F1 manually and via `f1_score`. Prints both — they match.
- **Cell 65 — markdown:** *"Why harmonic mean?"* — visual comparison: harmonic mean drops hard when either component drops, arithmetic mean doesn't.
- **Cell 66 — code:** Plots harmonic mean vs arithmetic mean across a grid of precision/recall values, showing harmonic's stricter behavior.
- **Cell 67 — markdown:** *"F-beta"* — generalization where β>1 weights recall higher (F2) and β<1 weights precision higher (F0.5).
- **Cell 68 — code:** Computes F0.5, F1, F2, and F5 for our model and prints each with interpretation — which ones emphasize precision vs recall.
- **Cell 69 — markdown:** *"When to use each F-score variant"* — F0.5 for precision-critical scenarios, F2 for recall-critical ones.
- **Cell 70 — code:** Bar chart comparing F1, F0.5, F2 across different hypothetical precision/recall combinations.
- **Cell 71 — markdown:** *"F1 vs Accuracy"* — on imbalanced data, F1 is far more informative.
- **Cell 72 — code:** Repeats the imbalanced-dataset comparison but now ranking models by F1 instead of accuracy. F1 correctly identifies the useful model.
- **Cell 73 — markdown:** Task 5 summary — harmonic means, β-tuning, and when to pick F1 vs F0.5 vs F2.

---

# Task 6: ROC Curves and AUC
*(Cells 74–77)*

**Task intro:** ROC-AUC is the standard threshold-free metric. It plots true-positive rate against false-positive rate across all thresholds, giving a single summary number.

- **Cell 74 — markdown:** Task 6 intro and objectives.
- **Cell 75 — markdown:** *"What is ROC?"* — receiver operating characteristic plots TPR vs FPR. AUC = area under that curve. 0.5 is random, 1.0 is perfect.
- **Cell 76 — code:** Computes ROC curve via `roc_curve` and `roc_auc_score`. Plots TPR vs FPR with the diagonal reference line. Annotates the AUC value. Very clean visualization of model quality.
- **Cell 77 — markdown:** Task 6 summary and the important caveat: ROC-AUC can be misleading on highly imbalanced datasets (see Task 7).

---

# Task 7: Precision-Recall Curves
*(Cells 78–80)*

**Task intro:** On imbalanced datasets, PR curves are more honest than ROC curves.

- **Cell 78 — markdown:** Task 7 intro — why PR curves matter more than ROC for imbalance.
- **Cell 79 — code:** Computes and plots the PR curve using `precision_recall_curve` and `average_precision_score`. Shows the trade-off you'd make at each threshold.
- **Cell 80 — markdown:** Task 7 summary: prefer PR curves when the positive class is rare.

---

# Task 8: Multi-class Classification Metrics
*(Cells 81–84)*

**Task intro:** Extending everything to more than two classes. Introduces macro, micro, and weighted averages.

- **Cell 81 — markdown:** Task 8 intro — multi-class setup.
- **Cell 82 — code:** Generates a 3-class dataset and trains a multi-class classifier. Prints the multi-class confusion matrix (N×N).
- **Cell 83 — code:** Computes precision, recall, and F1 with three averaging modes: `macro` (unweighted per-class mean), `micro` (global sum), and `weighted` (weighted by class support). Prints all three and explains the difference with a table.
- **Cell 84 — markdown:** Task 8 summary — pick your averaging mode based on whether all classes matter equally (macro) or the larger classes dominate naturally (micro/weighted).

---

# Task 9: Imbalanced Datasets & Advanced Metrics
*(Cells 85–88)*

**Task intro:** Specialized metrics for imbalanced data — Matthews Correlation Coefficient (MCC) and Cohen's Kappa.

- **Cell 85 — markdown:** Task 9 intro.
- **Cell 86 — code:** Computes MCC and Cohen's Kappa via sklearn. MCC ranges from −1 to +1 and handles imbalance gracefully; Kappa adjusts accuracy for agreement expected by chance.
- **Cell 87 — code:** Visual comparison across many metrics on an imbalanced dataset — shows which metrics reflect the model's actual quality and which are fooled by imbalance.
- **Cell 88 — markdown:** Task 9 summary — MCC is one of the best single-number metrics for imbalanced binary classification.

---

# Task 10: MLOps Pipeline Metrics & Best Practices
*(Cells 89–94)*

**Task intro:** How to pick metrics in production: alignment with business goals, monitoring drift, and calibration.

- **Cell 89 — markdown:** Task 10 intro.
- **Cell 90 — markdown:** *"Aligning metrics to business goals"* — a table of common business objectives and the technical metric that matches.
- **Cell 91 — code:** Demonstrates **calibration** — whether predicted probabilities match observed frequencies. Plots a calibration curve.
- **Cell 92 — code:** Uses `CalibratedClassifierCV` to recalibrate the model and plots before/after calibration curves.
- **Cell 93 — code:** Plots expected calibration error and shows how temperature scaling or Platt scaling can sharpen probabilities.
- **Cell 94 — markdown:** Task 10 summary — in production, track multiple metrics, monitor drift, and calibrate probabilities.

---

# Task 11: Multi-label Classification Metrics
*(Cells 95–111)*

**Task intro:** Multi-label means each sample can have *multiple* labels at once — like tagging images with "beach," "sunset," "person." The metrics change accordingly.

- **Cell 95 — markdown:** Task 11 intro and objectives.
- **Cell 96 — markdown:** *"Multi-label vs multi-class"* — clear side-by-side explanation.
- **Cell 97 — code:** Generates a synthetic multi-label dataset using `make_multilabel_classification`. Prints the label matrix shape — N samples × L labels, where each cell is 0 or 1.
- **Cell 98 — code:** Trains a OneVsRest classifier. Shows predictions — each row is a vector of 0/1 per label.
- **Cell 99 — markdown:** *"Hamming Loss"* — fraction of individual labels that are wrong across all samples.
- **Cell 100 — code:** Computes `hamming_loss` and prints it with interpretation.
- **Cell 101 — markdown:** *"Subset accuracy (Exact Match)"* — harsh metric requiring all labels to match exactly.
- **Cell 102 — code:** Computes exact-match accuracy. Much lower than Hamming loss would suggest — because missing any one of several labels counts as a total failure.
- **Cell 103 — markdown:** *"Label-based precision/recall/F1"* — per-label versions.
- **Cell 104 — code:** Computes per-label precision, recall, F1 with macro and micro averaging.
- **Cell 105 — markdown:** *"Example-based F1"* — averages F1 across samples rather than labels.
- **Cell 106 — code:** Computes example-based (samples-averaged) F1 and compares to macro-averaged F1.
- **Cell 107 — markdown:** *"Ranking metrics for multi-label"* — coverage error, label ranking loss, average precision.
- **Cell 108 — code:** Computes `coverage_error`, `label_ranking_loss`, and `label_ranking_average_precision_score`. Explains each.
- **Cell 109 — code:** Visualization of per-label F1 as a bar chart. Highlights which labels the model does well on and which it struggles with.
- **Cell 110 — code:** Confusion-matrix-like visualization per label — small matrices stacked to show performance on each.
- **Cell 111 — markdown:** Task 11 summary — a full toolbox for evaluating multi-label models.

---

# Task 12: Semi-supervised Classification Metrics
*(Cells 112–122)*

**Task intro:** When you have limited labeled data and lots of unlabeled data, you need special evaluation techniques.

- **Cell 112 — markdown:** Task 12 intro.
- **Cell 113 — markdown:** *"What is semi-supervised learning?"* — small labeled pool plus a large unlabeled pool.
- **Cell 114 — code:** Builds a semi-supervised setup — a small labeled subset, a large unlabeled pool, a held-out test set.
- **Cell 115 — code:** Trains a `LabelPropagation` model from sklearn and evaluates on the test set. Prints accuracy and F1.
- **Cell 116 — markdown:** *"Self-training loop"* — use the model to pseudo-label the unlabeled data, retrain, repeat.
- **Cell 117 — code:** Implements a simple self-training loop. Tracks how test accuracy evolves over iterations.
- **Cell 118 — code:** Plots the pseudo-labeling confidence distribution — helps decide a threshold for which pseudo-labels to trust.
- **Cell 119 — markdown:** *"Special metrics for semi-supervised"* — pseudo-label accuracy, labeled-vs-pseudolabeled F1.
- **Cell 120 — code:** Computes these diagnostic metrics iteration by iteration. Shows when self-training starts hurting (confirmation bias).
- **Cell 121 — code:** Visualizes the trade-off between amount of pseudo-labeled data used vs final test accuracy.
- **Cell 122 — markdown:** Task 12 summary — be careful with confirmation bias; always keep a clean held-out test set.

---

# Task 13: Confidence Scores, Thresholds & Bucketization
*(Cells 123–140)*

**Task intro:** Raw probability scores are gold — you can threshold them, bucket them, and tune the operating point for your business.

- **Cell 123 — markdown:** Task 13 intro.
- **Cell 124 — markdown:** *"What are confidence scores?"* — model probability outputs and what they do and don't guarantee.
- **Cell 125 — code:** Gets `predict_proba` from the classifier. Prints the distribution — ideally bimodal (confident 0s and confident 1s).
- **Cell 126 — code:** Histogram of confidence scores colored by correctness. Errors cluster around 0.5.
- **Cell 127 — markdown:** *"Threshold tuning"* — how shifting the threshold moves the precision/recall operating point.
- **Cell 128 — code:** Sweeps thresholds from 0 to 1 and plots precision, recall, F1, and accuracy as four curves. Marks the threshold that maximizes F1.
- **Cell 129 — code:** Plots TPR, FPR, and TNR across thresholds for completeness.
- **Cell 130 — markdown:** *"Finding the optimal threshold"* — business goals differ; pick based on F1, or a target precision, or cost-weighted.
- **Cell 131 — code:** Finds the threshold that maximizes F1 explicitly; then finds the threshold meeting a 95% precision floor. Prints both.
- **Cell 132 — code:** Computes a **cost-weighted** threshold — assign $ costs to FP and FN and minimize total cost. Very practical for real-world use.
- **Cell 133 — markdown:** *"Bucketization of probabilities"* — group predictions into low/medium/high confidence tiers.
- **Cell 134 — code:** Buckets predictions into low/medium/high confidence using thresholds. Prints the size of each bucket.
- **Cell 135 — code:** For each bucket, prints the actual accuracy — how reliable each confidence tier is. You typically see clean monotonic improvement: high-confidence predictions really are more accurate.
- **Cell 136 — markdown:** *"Reliability diagrams"* — formal calibration visualization.
- **Cell 137 — code:** Plots the reliability diagram — predicted probability on X axis, observed frequency on Y axis. A perfectly calibrated model lies on the diagonal.
- **Cell 138 — code:** Computes Brier score — the mean squared error between predicted probability and actual outcome.
- **Cell 139 — code:** Expected Calibration Error (ECE) — a summary scalar of the reliability diagram.
- **Cell 140 — markdown:** Task 13 summary — always analyze your confidence scores; don't just use the default 0.5 threshold.

---

# Task 14: Regression Metrics Introduction
*(Cells 141–142)*

**Task intro:** Switching gears — regression problems. The notebook now covers metrics for continuous outputs.

- **Cell 141 — markdown:** Task 14 intro — what regression is, with examples (house prices, energy consumption).
- **Cell 142 — code:** Sets up a regression problem using `make_regression` and/or California Housing. Trains a linear regression, makes predictions. Prints a small table of predictions vs actuals.

---

# Task 15: MAE, MSE, RMSE Deep Dive
*(Cells 143–149)*

**Task intro:** The three most common regression error metrics and how they behave differently.

- **Cell 143 — markdown:** Task 15 intro — defines MAE, MSE, RMSE.
- **Cell 144 — code:** Computes MAE, MSE, RMSE via sklearn. Prints all three values.
- **Cell 145 — markdown:** *"Why MSE and RMSE differ from MAE"* — squaring penalizes large errors more.
- **Cell 146 — code:** Demonstrates outlier sensitivity — synthesize residuals with and without outliers and show how MSE jumps while MAE barely moves.
- **Cell 147 — code:** Plots residuals — actual minus predicted — as a histogram. Comments on whether it's roughly Gaussian (ideal) or skewed (model bias).
- **Cell 148 — code:** Plots predicted vs actual scatter with the y=x reference line. Perfect predictions lie on the line.
- **Cell 149 — markdown:** Task 15 summary — MAE is robust to outliers, MSE/RMSE are sensitive; pick based on whether outliers are real or noise.

---

# Task 16: R² and Adjusted R²
*(Cells 150–154)*

**Task intro:** R² measures how much variance the model explains. Adjusted R² penalizes adding meaningless features.

- **Cell 150 — markdown:** Task 16 intro — defines R² = 1 − SS_res/SS_tot.
- **Cell 151 — code:** Computes `r2_score` and prints it with interpretation (fraction of variance explained).
- **Cell 152 — markdown:** *"Adjusted R²"* — penalizes model complexity to prevent gaming R² by adding features.
- **Cell 153 — code:** Computes adjusted R² manually using the formula. Compares plain R² and adjusted R² as you add features — adjusted goes up only when features are genuinely useful.
- **Cell 154 — markdown:** Task 16 summary — report both R² and adjusted R² when comparing models of different complexity.

---

# Task 17: Advanced Regression Metrics
*(Cells 155–156)*

**Task intro:** MAPE, SMAPE, quantile losses, and median absolute error for specialized situations.

- **Cell 155 — markdown:** Task 17 intro — introduces MAPE, SMAPE, quantile loss, and median AE.
- **Cell 156 — code:** Computes all of them and prints with interpretation — MAPE for percentage-error reporting, quantile loss for business decisions where under/over-prediction have different costs.

---

# Task 18: Industry Best Practices — Choosing the Right Metrics
*(Cells 157–161)*

**Task intro:** The final task. Walks through how real teams pick their evaluation metrics in production.

- **Cell 157 — markdown:** Task 18 intro.
- **Cell 158 — markdown:** *"The metric-selection framework"* — questions to ask yourself: is the data balanced, what's the cost of each error, do you need calibrated probabilities, is this multi-label?
- **Cell 159 — code:** Summary bar chart / radar plot of all the metrics we covered, ranked by when they're useful.
- **Cell 160 — markdown:** *"Common mistakes to avoid"* — over-optimizing a single metric, ignoring class imbalance, not calibrating, leaking test-set information into tuning.
- **Cell 161 — markdown:** Final summary — if you remember only one thing: never report just accuracy. Always include precision/recall/F1 or AUC, and always inspect your confusion matrix.

---

## Outro

That's the full tour — 162 cells and 18 tasks of classification and regression metrics. The single biggest takeaway: **accuracy is a starting point, not the finish line.** Always check the confusion matrix, always look at precision and recall (or MAE/RMSE/R² for regression), and always think about what errors cost in your specific application. Thanks for sticking with me through this one — it's the most important notebook in the set because these are the metrics you'll use for the rest of your ML career.
