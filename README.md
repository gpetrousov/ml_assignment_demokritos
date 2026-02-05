# Autofarming: Weed Control Repository

**MSc in Artificial Intelligence** | **NCSR Demokritos & University of Piraeus**
**Course:** Machine Learning
**Student:** Ioannis Petrousov

## 🎯 Repository Purpose
This repository contains a Python-based machine learning pipeline for agricultural image classification. It integrates feature extraction (HOG, LBP, HSV), data augmentation, and classification models to identify crops and weeds in field images.

## 🚀 Key Business Impact
Our optimized models translate statistical performance into real-world agricultural value:
* **Crop Preservation Rate:** The system identifies **91.7%** of all crops, ensuring harvestable products are protected.
* **Weed Suppression Efficiency:** Successfully flags and rejects **85.2%** of weeds, significantly reducing competition for soil nutrients.
* **Operational Reliability:** When a plant is flagged as a weed, the model is **99.5%** accurate, minimizing the risk of accidental crop destruction.

## 📂 Repository Structure
```text
├── notebooks/
│   ├── 00_baseline.ipynb       # Research logic and experimental sandbox
│   └── 01_report.ipynb         # End-to-end classification pipeline & RISE Presentation
├── weedcrop/                   # Module Source package
│   ├── autofarm.py             # Core library logic (Extraction, Training, Eval)
│   └── ...
├── requirements.txt            # Pip dependencies
├── environment.yml             # Conda dependencies
└── README.md                   # Repository documentation
```

## ⚙️ Environment Setup

To reproduce the environment, ensure you have Python 3.10+ installed.

1. Clone the Repository

```bash
git clone "[https://github.com/gpetrousov/ml_assignment_demokritos.git](https://github.com/gpetrousov/ml_assignment_demokritos.git)"
cd "ml_assignment_demokritos"
```

2. Set Python Version (Pyenv)

To match the experimental environment, it is recommended to use Python 3.10.13:

```bash
pyenv install 3.10.13
pyenv local 3.10.13
```

3. Install Dependencies (Pip)

```bash
pip install -r "requirements.txt"
```

4. Create and Activate the Conda Environment

```bash
conda env create -f "environment.yml"
conda activate "ml_clean"
```

5. Launch the Notebook(s)

Run the notebook and ensure the ml_clean kernel is selected:

```bash
jupyter notebook notebooks/01_report.ipynb
```

6. Optional: Enable RISE Presentation

If you need to watch the presentation.

```bash
jupyter-nbextension install rise --py --sys-prefix
jupyter-nbextension enable rise --py --sys-prefix
```

## 📚 Dataset

If you need to run the notebooks, you need to have the dataset in the root directory of the repo.

- https://datasetninja.com/dataset-of-annotated-food-crops-and-weed-images

```text
@article{SUDARS2020105833,
  title = {Dataset of annotated food crops and weed images for robotic computer vision control},
  journal = {Data in Brief},
  volume = {31},
  pages = {105833},
  year = {2020},
  issn = {2352-3409},
  doi = {[https://doi.org/10.1016/j.dib.2020.105833](https://doi.org/10.1016/j.dib.2020.105833)},
  url = {[https://www.sciencedirect.com/science/article/pii/S2352340920307277](https://www.sciencedirect.com/science/article/pii/S2352340920307277)},
  author = {Kaspars Sudars and Janis Jasko and Ivars Namatevs and Liva Ozola and Niks Badaukis},
}
```
