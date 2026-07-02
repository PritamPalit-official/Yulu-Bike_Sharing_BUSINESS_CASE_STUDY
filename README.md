<div align="center">

# 🚲 Yulu-Bike Sharing Business Case Study

### Data-Driven Analysis of Rental Demand Patterns for Yulu E-Cycles

*Exploratory Analytics, Variance & Correlation Studies, and Hypothesis Testing (T-Test, ANOVA, Chi-Square, Tukey HSD) to Enable Smarter Marketing & Operational Decisions*

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br>

[📄 View Full Analysis Report (PDF)](Yulu-Bike_Sharing.pdf)

---

</div>

## 📑 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [❓ Business Problem](#-business-problem)
- [📊 Dataset Description](#-dataset-description)
- [🔬 Analysis Workflow](#-analysis-workflow)
- [📐 Statistical Methods Used](#-statistical-methods-used)
- [💡 Key Insights](#-key-insights)
- [📈 Business Recommendations](#-business-recommendations)
- [🛠️ Tools & Technologies](#️-tools--technologies)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [👤 Author](#-author)
- [📄 License](#-license)

---

## 🎯 Project Overview

This project presents a **statistical and hypothesis-driven analysis** of Yulu's bike-sharing demand data. The objective is to identify key factors influencing rental demand and to provide **data-backed recommendations** for revenue recovery and operational optimization.

> Yulu is India's leading micro-mobility service provider, offering unique vehicles for daily commutes. This case study dives deep into their rental data to uncover actionable patterns that drive business growth.

---

## ❓ Business Problem

Yulu has observed a **significant decline in revenue** and seeks to understand how temporal, weather, and operational factors affect customer demand. The analysis focuses on:

- 📉 Identifying **statistically significant drivers** of bike rentals
- 🔍 Validating insights through **rigorous hypothesis testing**
- 📊 Understanding the interplay between **weather, seasonality, and demand**
- 💰 Providing **actionable strategies** for revenue recovery

---

## 📊 Dataset Description

The dataset contains **10,886 records** with **12 features**, capturing comprehensive information about bike rental patterns.

| Feature | Description |
|:---|:---|
| `datetime` | Hourly date and timestamp |
| `season` | Season of the year (1: Spring, 2: Summer, 3: Fall, 4: Winter) |
| `holiday` | Whether the day is a holiday (0: No, 1: Yes) |
| `workingday` | Whether the day is a working day (0: No, 1: Yes) |
| `weather` | Weather condition (1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain/Snow) |
| `temp` | Temperature in Celsius |
| `atemp` | "Feels like" temperature in Celsius |
| `humidity` | Relative humidity |
| `windspeed` | Wind speed |
| `casual` | Count of casual (non-registered) users |
| `registered` | Count of registered users |
| `count` | **Target Variable** — Total rental count (casual + registered) |

---

## 🔬 Analysis Workflow

The analysis follows a structured, end-to-end approach:

```
📥 Data Loading
 │
 ├── 🧹 Step 1: Data Cleaning & Feature Engineering
 │         ├── Datetime conversion
 │         └── Categorical encoding
 │
 ├── 📊 Step 2: Exploratory Data Analysis
 │         ├── Univariate analysis of demand patterns
 │         └── Bivariate analysis across features
 │
 ├── 🔗 Step 3: Correlation Analysis
 │         └── Weather, temperature & rental relationships
 │
 ├── 🧪 Step 4: Hypothesis Testing
 │         ├── Two-sample t-test
 │         ├── One-way ANOVA
 │         └── Chi-square test of independence
 │
 ├── 🔎 Step 5: Post-hoc Analysis
 │         └── Tukey HSD for detailed group comparisons
 │
 └── 💼 Step 6: Business Interpretation
           └── Translating statistical results into actionable insights
```

---

## 📐 Statistical Methods Used

| Method | Purpose |
|:---|:---|
| **Independent Two-Sample T-Test** | Compare rental demand between working days and non-working days |
| **One-Way ANOVA** | Test demand differences across seasons and weather conditions |
| **Tukey HSD Post-Hoc Test** | Pairwise comparisons after significant ANOVA results |
| **Chi-Square Test of Independence** | Examine association between weather and season |
| **Correlation Analysis** | Quantify relationships between continuous variables |

---

## 💡 Key Insights

The analysis confirms several statistically significant findings:

| Finding | Impact |
|:---|:---|
| 🏢 **Working Days** | Significantly higher demand on working days vs. non-working days |
| 🍂 **Fall Season** | Peak rental demand observed during the fall season |
| ☀️ **Clear Weather** | Clear/partly cloudy weather drives the highest rentals |
| 🌧️ **Adverse Weather** | Heavy rain/snow significantly reduces demand |
| 🔗 **Weather–Season Dependency** | Weather patterns are strongly dependent on seasonality |

> **Key Takeaway:** Demand is not random — it is systematically driven by working day patterns, seasonal cycles, and weather conditions, all of which can be leveraged for strategic planning.

---

## 📈 Business Recommendations

| # | Recommendation | Expected Outcome |
|:---:|:---|:---|
| 1 | 🚲 **Increase bike availability** during working days & high-demand seasons | Capture unmet demand and reduce lost revenue |
| 2 | 💲 **Implement dynamic pricing** based on demand & weather forecasts | Maximize revenue during peak periods |
| 3 | 🔧 **Schedule maintenance** during low-demand periods | Minimize service disruption |
| 4 | 📣 **Align marketing campaigns** with seasonal demand trends | Improve ROI on marketing spend |
| 5 | 🏗️ **Plan infrastructure expansion** in regions with favorable weather | Strategic long-term growth |

---

## 🛠️ Tools & Technologies

<div align="center">

| Category | Tools |
|:---|:---|
| **Programming Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Statistical Analysis** | SciPy, StatsModels |
| **Development Environment** | Jupyter Notebook |

</div>

---

## 📁 Project Structure

```
Yulu-Bike_Sharing_BUSINESS_CASE_STUDY/
│
├── 📁 data/
│   └── 📊 yulu_data.csv            # Bike sharing rental dataset
├── 📁 images/                      # Exported EDA and hypothesis visualization plots
├── 📄 app.py                       # Interactive Streamlit dashboard & live hypothesis test sandbox
├── 📄 requirements.txt             # Dependency definitions
├── 📓 Yulu_Hypothesis_Testing.ipynb # Hypothesis testing Jupyter notebook
├── 🐍 Yulu_Hypothesis_Testing.py    # Python counterpart of the analysis notebook
├── 📊 Yulu-Bike_Sharing.pdf        # Complete analysis report with visualizations
└── 📜 LICENSE                      # MIT License
```

---

## 🚀 Getting Started

### 🖥️ Running the Interactive Dashboard (Streamlit)

To view the live interactive web dashboard:

1. **Clone the repository and navigate inside**:
   ```bash
   git clone https://github.com/PritamPalit-official/Yulu-Bike_Sharing_BUSINESS_CASE_STUDY.git
   cd Yulu-Bike_Sharing_BUSINESS_CASE_STUDY
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

This will automatically launch the dashboard in your default browser at `http://localhost:8501`.

### 📓 Running the Notebook Analysis

If you'd like to explore the step-by-step Jupyter Notebook analysis:

1. **Install Jupyter and base requirements**:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy statsmodels jupyter
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```
   Open `Yulu_Hypothesis_Testing.ipynb` to see the complete exploration workflow.

---

## 🛠️ Development & Testing

To maintain production-ready code quality, this repository includes dev dependencies, unit testing configurations, and automated CI pipelines:

### 📦 Setup Developer Dependencies
Install the required development and testing packages:
```bash
pip install -r requirements-dev.txt
```

### 🧪 Run Unit Tests Locally
Run the test suite using Python's built-in `unittest` runner:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### ⚙️ Continuous Integration (CI)
A GitHub Actions workflow is configured in `.github/workflows/ci.yml`. On every `push` and `pull_request` to the repository, it automatically:
1. Provisions an Ubuntu runner with Python 3.10.
2. Installs dependencies from both `requirements.txt` and `requirements-dev.txt`.
3. Runs the test suite to verify code integrity and prevent regressions.

---

## 👤 Author

<div align="center">

**Pritam Palit**

🎓 Electronics and Communication Engineering Graduate

📊 Focus Areas: Data Analytics | Statistics | Business Intelligence

[![GitHub](https://img.shields.io/badge/GitHub-PritamPalit--official-181717?style=for-the-badge&logo=github)](https://github.com/PritamPalit-official)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Pritam_Palit-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/pritam-palit-77b2071b4/)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If you found this project insightful, please consider giving it a star!** ⭐

</div>
