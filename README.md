Practical Machine Learning Project with Streamlit — Pattern Recognition

This repository presents a complete application for predicting academic dropout in on-campus programs at public Higher Education Institutions (HEIs) using microdata from the INEP/MEC Higher Education Census. The application was developed as part of the Pattern Recognition course (2COP507) and follows the requirements set by Professors Bruno B. Zarpelão and Sylvio Barbon Jr. for the Streamlit practical project. In addition to demonstrating the pipeline for data collection, processing, and modeling, the application includes an interactive Streamlit interface with options for tuning hyperparameters and visualizing results.

Objective

Develop a practical application that incorporates at least one machine learning algorithm studied in the course (regression, classification, or ensembles). The goal is to apply data science techniques to estimate dropout rates in on-campus courses using INEP/MEC microdata and present the results through a user-friendly interface. In this implementation the base model is a Random Forest, but the pipeline allows testing other algorithms (Linear Regression, neural networks with fine-tuning) as required.

Folder Structure
├── dados/                # Dataset(s) used in the project
│   ├── bruto/            # Raw data (INEP microdata) downloaded directly from official sources
│   ├── processado/       # Data after cleaning and preprocessing
│   └── intermediario/      # Temporary data generated during processing
├── modelos/              # Trained models and saved metrics
├── notebooks/            # Exploratory Jupyter notebooks (optional)
├── scripts/              # Python scripts for process automation
│   ├── coleta_dados/     # Microdata collection (e.g., coleta_dados_oficiais.py)
│   ├── processamento/    # Preprocessing and feature engineering
│   ├── modelagem/        # Model training and fine-tuning
│   └── visualizacao/     # Generation of charts and reports
├── app/                  # Streamlit application
│   └── app_evasao.py     # Interactive dashboard with hyperparameter controls
├── requisitos.txt        # Project dependencies
└── README.md             # This document

Here is the requested translation into U.S. English. All folder and script names (e.g., requisitos.txt, scripts/coleta_dados/coleta_dados_oficiais.py) have been preserved in Portuguese as instructed.

⸻

Dependencies
	•	Python 3.8 or higher
	•	Streamlit for the interactive interface
	•	Modeling and visualization libraries: pandas, numpy, scikit-learn, seaborn, matplotlib
	•	tensorflow/keras (optional) for neural networks
	•	See the requisitos.txt file for the complete list. Install dependencies with:
	python3 -m venv .venv
source .venv/bin/activate
pip install -r requisitos.txt


⸻

Project Pipeline

1. Microdata Collection
The official INEP microdata (2009–2023) are downloaded directly from public sources using the script scripts/coleta_dados/coleta_dados_oficiais.py.
This script creates the folder structure under dados/bruto/ and stores the CSV files corresponding to annual censuses.
This collection step needs to be executed only once:
python scripts/coleta_dados/coleta_dados_oficiais.py

2. Preprocessing and Exploratory Analysis
Preprocessing involves cleaning and standardizing the data, selecting only on-campus programs, and consolidating key variables.
Run the script scripts/processamento/pre_processamento.py to unify the years and generate processed files under dados/processado/:
python scripts/processamento/pre_processamento.py

This step includes:
	•	Selection of relevant columns (e.g. taxa_ingresso, taxa_conclusao, taxa_evasao, vagas_totais)
	•	Handling of missing values and type conversions
	•	Generation of dados_cursos_tratado.csv and dados_ies_tratado.csv for modeling
	•	Basic exploratory analysis (histograms, heatmaps), optionally via Jupyter notebooks

3. Data Splitting and Feature Engineering
The scripts under the folder scripts/processamento allow preparing various input sets.
For example, for Transfer Learning or combining additional features (media_geral, taxa_ingresso, taxa_conclusao).
For the Streamlit project, it is recommended to start with the variables taxa_ingresso and taxa_conclusao to predict taxa_evasao.

4. Modeling
Train the machine learning models by running the modeling scripts:
	•	Original model (Random Forest + Linear Regression) — scripts/modelagem/modelagem.py
	•	Model with feature engineering (Random Forest) — scripts/modelagem/treinamento_modelo_Feature-based.py
	•	Model with neural network fine-tuning (ANN) — scripts/modelagem/treinamento_modelo_Fine-tuning.py (optional)
	•	C4.5/J48 decision tree model — scripts/modelagem/treinamento_modelo_C4.5_Tree_J48.py (evaluated as an alternative)

These scripts split the dataset into training and testing subsets, train the models, and log the metrics (MSE, R²) under modelos/resultados_modelos/.
By default, modelagem.py compares Linear Regression with Random Forest and saves the best model.
You can adjust the hyperparameters directly in the scripts or through the Streamlit panel.

5. Evaluation and Visualization
Use scripts/visualizacao/gerar_graficos.py to generate evolution curves of enrollment, completion, and dropout rates, as well as confusion matrices and model comparisons.
The metrics are saved as PNG files under modelos/resultados_modelos/.
For temporal validation and course-specific comparisons, run scripts/analises/validar_modelos_temporais.py and scripts/analises/comparar_predicoes_cursos.py.

6. Streamlit Interface
The interactive application is implemented in app/app_evasao.py.
This panel loads the trained model and allows users to:
	•	Select a course (e.g. Administration, Law, Civil Engineering, Medicine) for analysis
	•	Adjust hyperparameters of the Random Forest model via sliders and drop-down menus (e.g. number of trees n_estimators, maximum depth max_depth, and criterion) — using st.slider and st.selectbox
	•	Display real-time metrics (MSE, R²) and a confusion matrix for binary classification (high vs. low dropout)
	•	Plot line charts comparing actual vs. predicted rates over time

To run the panel:
streamlit run app/app_evasao.py

7. Video and Report
For the final submission:
	•	Record a video (up to 15 minutes) explaining:
(1) project motivation, (2) the constructed pipeline, (3) how the model works, and (4) the Streamlit interface.
	•	Additionally, prepare a short article (up to 4 pages) using the SBC template. The document should describe the choice of algorithms, tested hyperparameters (with justification), and discuss results and future improvements.

⸻

About Model Reuse

Previously trained Random Forest models can serve as a baseline, but it is recommended to adjust them when integrating with the Streamlit app.
Two main reasons:
	1.	Hyperparameters: The scripts currently define fixed values such as n_estimators=100 and max_depth=None.
To meet interaction requirements, expose these parameters in the panel for users to adjust and observe the impact on metrics.
	2.	Data updates: The MVP described in the Streamlit project initially used data from 2023, and later expanded to data from 2009 to 2023.
If you use a different data subset or add new features, the model should be retrained to reflect these changes.

Therefore, reuse the structure of the modeling scripts (training and saving functions), but allow hyperparameter adjustments in the code or interface.
This will ensure the application meets the evaluation criteria (interface, model implementation, documentation) set by the professors.

⸻

Contribution and License

This project was developed by Eduardo Fernandes Bueno as the Machine Learning Practical Project with Streamlit (2COP507 – Recognition of Patterns).
Feel free to clone the repository, test other algorithm combinations, or add new courses.
The license and additional details can be found in LICENCA.

Contributions, Contact, and Citation
	•	Author: Eduardo Fernandes Bueno — developed as the Practical ML Project with Streamlit for course 2COP507 (Recognition of Patterns).
	•	Advisors/Professors: Prof. Bruno B. Zarpelão and Prof. Sylvio Barbon Jr.
	•	Contact: See commit history or collaborators section for contribution and contact details.

How to cite this repository:
Eduardo F. Bueno (2025). Machine Learning Practical Project with Streamlit — Recognition of Patterns (2COP507). GitHub repository. Available at: (insert repository URL).

⸻

Acknowledgments

Special thanks to INEP/MEC for providing the microdata and to the related academic courses for supplying the theoretical and practical foundation for this project.

⸻

Contributions

You are welcome to clone the repository, open issues, and submit pull requests with fixes, new features, or additional course models and algorithms.
If available, see CONTRIBUTING.md for guidelines.

⸻

License

Please see the LICENCA file for terms of use and distribution.

