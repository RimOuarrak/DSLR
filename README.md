# 🧙‍♂️✨ DataScience × Logistic Regression  
### *Harry Potter and a Data Scientist*

> “When the Sorting Hat fails, only a true Data Scientist can save Hogwarts.” 🪄💻  

---

## 🪶 Summary

**Version:** 2  
**Mission:** Write a classifier and save Hogwarts!  
Professor McGonagall needs your help to restore the **Sorting Hat** — and your only weapons are **Python**, **Math**, and **Data Science**.  

You’ll explore, visualize, and classify wizarding data to recreate the magic hat’s intelligence using **Logistic Regression** and **Gradient Descent**. ⚡  

---

## 🧭 Contents

1. [Foreword](#-foreword)  
2. [Introduction](#-introduction)  
3. [Objectives](#-objectives)  
4. [General Instructions](#-general-instructions)  
5. [Mandatory Part](#-mandatory-part)  
6. [Bonus Part](#-bonus-part)  
7. [Submission & Evaluation](#-submission--peer-evaluation)  
8. [Annexes](#-annexes)  

---

## 📖 Foreword

> “It is our choices that show what we truly are, far more than our abilities.” – *Albus Dumbledore* 🪄

This project draws inspiration from the work of **Yann LeCun**, one of the founding fathers of modern AI and deep learning.  
You’ll be walking in the footsteps of great minds — but with a Hogwarts twist. ⚡  

---

## 🪄 Introduction

The Sorting Hat has been **bewitched** and can no longer assign students to their houses!  
Professor McGonagall calls upon *you*, a muggle data scientist, to recreate the Sorting Hat’s intelligence using **Logistic Regression**.  

After a quick *“Digitalis!”* spell, the old spellbook becomes a USB stick filled with wizarding student data...  
Your quest begins. 🧙‍♀️📊  

---

## 🎯 Objectives

Your task is to:
- 🧮 Read, analyze, and clean the dataset  
- 📊 Visualize it using **histograms**, **scatter plots**, and **pair plots**  
- 🤖 Implement a **multi-class logistic regression classifier** (One-vs-All)  
- 🧠 Train your Magic Hat to predict Hogwarts houses with at least **98% accuracy**  

By the end, you’ll have:
- A working **logistic regression model**
- A **machine learning toolkit**
- A well-trained **Sorting Hat 2.0**

---

## ⚙️ General Instructions

- You can use **any programming language**, but Python is recommended 🐍  
- 🚫 You may **not** use built-in summary/statistics functions like:
  - `count()`, `mean()`, `std()`, `min()`, `max()`, `percentile()`, etc.  
- The goal is to **understand and implement everything yourself** — no shortcuts, no magic spells! 🪄  

---

## 📚 Mandatory Part

### 🧾 V.1 Data Analysis
Create a program:
```bash
$> python describe.py dataset_train.csv
```
Output key statistics **(count, mean, std, min, quartiles, max)** for every numeric feature — all computed manually.  
> ❗ No cheating with `pandas.describe()`!

---

### 📊 V.2 Data Visualization
Data visualization helps you *see* magic patterns in your data.  

#### 🪄 Histogram
`histogram.py` — find which course has a homogeneous score distribution between all four houses.

#### 🔮 Scatter Plot
`scatter_plot.py` — show which two features seem most similar.

#### 🧩 Pair Plot
`pair_plot.py` — display all features against each other to spot correlations and select features for your model.

---

### 🧠 V.3 Logistic Regression
The final and most magical part: the **Magic Sorting Hat**! 🎓  

You must implement a **multi-class logistic regression classifier (one-vs-all)**.  

#### 🪄 Training
```bash
$> python logreg_train.py dataset_train.csv
```
Use **gradient descent** to minimize the cost function and save your trained weights.  

#### 🔮 Prediction
```bash
$> python logreg_predict.py dataset_test.csv weights.csv
```
This script predicts each student’s house and outputs:
```csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Slytherin
...
```

---

## 🌟 Bonus Part

Only graded if your mandatory part is **PERFECT** 🧠💫  

Ideas for bonuses:
- ✨ Add more stats in `describe.py`
- 🌀 Implement **Stochastic** or **Mini-Batch** Gradient Descent
- ⚙️ Add **regularization** or advanced optimization (Adam, RMSProp)
- 📉 Add a **loss curve visualization**  

---

## 🧪 Submission & Peer Evaluation

🧾 Submit via your **Git repository**.  
Only the files inside your repo will be evaluated.  

Your model will be graded based on:
- ✅ Correct implementation  
- 🧩 98%+ accuracy on the test dataset  
- 🧠 Your understanding of the math behind logistic regression  

Be ready to **defend** your code and explain your approach like a true data scientist wizard 🪄

---

## 📚 Annexes

### 🧮 Mathematics

Logistic Regression uses:
```math
J(θ) = -1/m * Σ [yᵢ log(hθ(xᵢ)) + (1 - yᵢ) log(1 - hθ(xᵢ))]
```
Where:
- `hθ(x) = g(θᵀx)`
- `g(z) = 1 / (1 + e^(-z))`

Gradient of the loss:
```math
∂J(θ)/∂θⱼ = 1/m * Σ [(hθ(xᵢ) - yᵢ) * xᵢⱼ]
```

---

### 🎨 Visualization Examples

📊 Histogram  
🌌 Scatter Plot  
🧩 Pair Plot  

Each revealing hidden relationships between Hogwarts courses and houses 🏰  

---

## 💫 Credits

👩‍💻 **Created by:** [Amine](https://github.com/mtourham) &  [Rim](https://github.com/RimOuarrak)
🧙‍♂️ Inspired by Hogwarts, Yann LeCun, and the magic of Machine Learning.  

> “Muggle tools, wizard results.” ⚡  

---

## 🪶 License
📝 MIT License — feel free to fork, star ⭐, and cast your own data spells.
