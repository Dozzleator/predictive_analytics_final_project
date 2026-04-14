# Machine Learning Recommender System

An automated machine learning validation and recommendation system. This application allows users to upload raw dataset files, automatically optimises preprocessing and modelling pipelines using Bayesian search, and generates natural language justifications for every mathematical decision using a local Large Language Model.

## Core Features

* **Automated Data Profiling:** Scans uploaded datasets to extract column metadata, missing value counts, unique value ratios, and distribution statistics.
* **Bayesian Pipeline Optimisation:** Utilises Optuna to dynamically search for the highest performing machine learning pipeline, testing various combinations of imputation, scaling, encoding, and base algorithms.
* **Dynamic Target Transformation:** Automatically detects skewed or negative target variables and applies safe non-linear transformations (Log1p or Arcsinh) to prevent mathematical overflow and improve regression accuracy.
* **AI-Powered Explanations:** Integrates with a local Llama 3.2 model via Ollama to generate professional, context-aware justifications for every feature engineering decision based on the dataset's specific metadata.
* **Interactive UI:** Built with Streamlit, featuring dynamic data tables and a custom CSS vertical timeline that visualises the exact feature engineering roadmap for the top-ranked models.

## Project Architecture

The application is strictly separated into modular components to maintain clean logic and UI isolation.

```text
predictive_analytics_final_project/
├── src/
│   ├── config.yaml
│   ├── features/
│   │   ├── ai_explainer.py       
│   │   ├── data_scan.py          
│   │   ├── pipeline_builder.py   
│   │   └── read_config.py        
│   └── ui/
│       ├── app.py                
│       └── timeline_builder.py   
```

## Prerequisites

To run this application locally, you will need the following installed on your machine:

* Python 3.10 or higher
* Ollama (with the `llama3.2:3b` model pulled locally)

## Installation and Setup

Clone the repository and navigate to the project root:

```bash
cd predictive_analytics_final_project
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install requirements.txt
```

Ensure your local LLM is ready. Open a separate terminal and verify Ollama is running and has the correct model installed:

```bash
ollama pull llama3.2:3b
```

## Usage Instructions

Navigate to the `src` directory where the main application resides:

```bash
cd src
```

Launch the Streamlit interface:

```bash
streamlit run ui/app.py
```

Upload your `.csv` dataset through the browser interface, input the exact name of the column you wish to predict, and adjust the target accuracy and maximum trial sliders.

Click **'Generate Optimised Pipeline'** to begin the Bayesian search. Once complete, navigate the tabs to explore the mathematical roadmap and AI justifications for the top three models.

## Configuration Guide (config.yaml)

The optimisation engine is highly customisable. You can easily restrict or expand the mathematical techniques Optuna is allowed to test by editing the `config.yaml` file in the `src` directory.

### Example Configuration Structure

```yaml
imputation:
  - 'mean'
  - 'median'
  - 'most_frequent'

scaling:
  - 'standard'
  - 'robust'
  - 'none'

encoding:
  - 'one_hot'
  - 'ordinal'

models:
  classification:
    - 'random_forest'
    - 'logistic_regression'
  regression:
    - 'random_forest'
    - 'linear_regression'
```

## Modifying the Search Space

**Preprocessing:** If you want to force the pipeline to only use robust scaling to handle extreme outliers, simply remove `'standard'` and `'none'` from the scaling list.

**Algorithms:** To introduce new algorithms to the search, add them to the respective models list. You will also need to define the algorithm wrapper inside `pipeline_builder.py`.

## Technical Details

**Mathematical Safeguards:** The pipeline engine includes custom wrapper functions combined with NumPy clipping to prevent memory overflow errors during inverse transformations on highly volatile datasets.

**Code Style:** All Python backend logic adheres to PEP 8 standards, utilising single quotes for strings where possible to maintain clean readability.
