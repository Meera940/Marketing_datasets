# 🛒 Market Basket Analysis Dataset

## 📌 Overview

This repository contains a **Market Basket Analysis** dataset generated using association rule mining techniques. The dataset includes relationships between products that are frequently purchased together and provides important metrics such as **support, confidence, lift, leverage, and conviction**. It is useful for learning data mining concepts, association rule mining, recommendation systems, and retail analytics.

---

## 📂 Dataset Information

| Feature        | Description                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| **antecedent** | The product(s) or item(s) that appear first in an association rule.                                      |
| **consequent** | The product(s) that are likely to be purchased together with the antecedent.                             |
| **support**    | The proportion of transactions containing both antecedent and consequent.                                |
| **confidence** | The probability that the consequent is purchased when the antecedent is purchased.                       |
| **lift**       | Measures how much more likely the consequent is purchased with the antecedent compared to random chance. |
| **leverage**   | Indicates the difference between the observed and expected frequency of the rule.                        |
| **conviction** | Measures the strength of implication between antecedent and consequent.                                  |
| **n_items**    | Total number of items involved in the association rule.                                                  |

---

## 🎯 Objectives

* Analyze customer purchasing patterns.
* Discover frequently bought-together products.
* Generate association rules using market basket analysis.
* Understand rule evaluation metrics such as Support, Confidence, and Lift.
* Build recommendation systems for retail businesses.

---

## 📊 Example Data

| antecedent       | consequent       | support | confidence | lift   |
| ---------------- | ---------------- | ------- | ---------- | ------ |
| wipes            | diapers, formula | 0.18    | 0.7759     | 4.1196 |
| diapers, formula | wipes            | 0.18    | 0.9558     | 4.1196 |
| diapers, wipes   | formula          | 0.18    | 0.9474     | 4.0143 |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📈 Possible Analyses

* Top association rules
* Product recommendation system
* Frequent itemset analysis
* Rule filtering based on support and confidence
* Data visualization of association metrics

---

## 🚀 How to Use

1. Clone this repository.
2. Install the required Python libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
3. Load the dataset:

   ```python
   import pandas as pd

   df = pd.read_csv("13c59a77-3176-4660-9f32-c526893e76c3.csv")
   print(df.head())
   ```
4. Perform exploratory data analysis and association rule evaluation.

---

## 📚 Applications

* Retail Analytics
* E-commerce Product Recommendations
* Inventory Management
* Customer Behavior Analysis
* Cross-selling and Upselling
* Business Intelligence
