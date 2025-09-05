# 🔥 Motor Temperature Forecasting (PMSM)



## 📌 Project Overview

This project explores and investigates the thermal behavior of a **Permanent Magnet Synchronous Motor (PMSM)** using real-world sensor data and time-series machine learning techniques.  

The operating temperature of electric machines is a key factor for their performance and reliability. Excessive heating increases resistance in windings and accelerates insulation aging. Since direct measurement of these temperatures is often impractical or costly, there is strong motivation to develop **virtual sensors** — machine learning models capable of reliably forecasting the thermal behavior of the motor.

My motivation for this work comes from an interest in **applied machine learning grounded in physical systems**. The project combines:  
- **Physics-informed reasoning** (heat transfer, electrical losses, torque–speed dynamics).  
- **Time-series forecasting methods** (from simple persistence models to advanced neural networks).  
- **Practical engineering considerations** (reproducibility, deployment efficiency, and real-time feasibility).  

The objective is not only to evaluate and compare modeling approaches, but also to design **robust development and deployment pipelines**, integrating **state-of-the-art model optimization techniques**. In doing so, the project aims to demonstrate how machine learning and physics can reinforce each other to address **real industrial challenges** in electric drives.

## 📑 Table of Contents
- [🔥 Motor Temperature Forecasting (PMSM)](#-motor-temperature-forecasting-pmsm)
  - [📌 Project Overview](#-project-overview)
  - [📑 Table of Contents](#-table-of-contents)
  - [📊 Dataset](#-dataset)
    - [Content](#content)
  - [📂 Project Structure](#-project-structure)
  - [⚙️ Installation](#️-installation)
  - [▶️ Usage](#️-usage)
  - [📜 License](#-license)

## 📊 Dataset

The work builds on the **Electric Motor Temperature Dataset (PMSM)** provided by the LEA department at Paderborn University:contentReference[oaicite:0]{index=0}.  
It consists of **185 hours of test bench measurements**, recorded at **2 Hz**, across **69 operating profiles** designed to mimic realistic driving conditions.

### Content
Each record includes:
- **Electrical signals:** `u_d`, `u_q` (voltages), `i_d`, `i_q` (currents).  
- **Mechanical variables:** `motor_speed` [rpm], `torque` [Nm].  
- **Environmental variables:** `ambient` and `coolant` temperatures [°C].  
- **Thermal targets:** `stator_winding`, `stator_tooth`, `stator_yoke`, and `pm` (permanent magnet) temperatures [°C].  
- Data is grouped by **`profile_id`**, corresponding to different drive cycles.

## 📂 Project Structure

This is the projet structure cible, It will be populated as the project advances:

```bash
motor-temp-forecasting/
│── data/ # Raw and processed datasets
│── docs/ # Documentation and bibliography
│── experiments/ # Experiment management
│── notebooks/ # Interactive notebooks (EDA → modeling → results)
│── reports/ # Outputs and documentation
│── src/ # Reusable Python modules│
│── tests/ # Unit tests (data splits, feature generation, model export)│
│── environment.yml # Conda environment
│── LICENSE # Open-source license (MIT)
│── README.md # Project documentation
```

## ⚙️ Installation

**1. Clone the repository**
```bash
   git clone https://github.com/davidmayal23/motor_temperature_forecasting.git
   cd motor_temperature_forecasting
```
**2. Set up the environment**
Use `conda` or `pip` to install the necessary dependencies.
```bash
    conda env create -f environment.yml
    conda activate motor_temp
```
**3. Download the dataset**

Download the [Electric Motor Temperature Dataset](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) and place the main CSV file (measures_v2.csv) in the data/raw/ folder:
```bash
    mkdir -p data/raw data/processed
    mv ~/Downloads/measures_v2.csv data/raw/
```

## ▶️ Usage

Currently available:
- `notebooks/01_exploration.ipynb`  
  Perform exploratory data analysis on the PMSM dataset:
  - Visualize signals (torque, speed, currents, voltages, temperatures).  
  - Inspect correlations between electrical and thermal variables.  
  - Identify trends, outliers, and thermal inertia.  

Run Jupyter Lab and open the notebook:
```bash
jupyter lab notebooks/01_exploration.ipynb
```

Planned next steps:
- Preprocessing & feature engineering (02_preprocessing.ipynb)
- Baseline forecasting models (03_baseline_models.ipynb)
- Deep learning forecasting (04_dl_models.ipynb)
- Model compression / optimization (05_model_compression.ipynb)
- Results dashboard (06_results_dashboard.ipynb)

## 📜 License
This repository is released under the **MIT License**.  
The dataset is distributed under **CC BY-SA 4.0** by Paderborn University.