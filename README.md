# Hum2Tune

**Hum2Tune** is a melody-based music recognition prototype. It allows a user to hum a short melody or upload a humming audio file, then returns the closest matching songs from a prepared reference database.

This project was developed for the Final Year Project:

**Hum2Tune: Melody-Based Music Recognition App Using Frequency Analysis**  
Wong Xuan Kai  
Student Number: 20495251  
BSc Computer Science with Artificial Intelligence  
University of Nottingham Malaysia  
Supervisor: Dr Simon Lau  
Year: 2026

---

## 1. What This Project Does

Hum2Tune is designed for the **Query-by-Humming** problem.

In normal music recognition apps, the user usually records part of the real song. That is much easier because the query still contains the original instruments, vocals, rhythm, and production quality.

Hum2Tune works with a harder situation: the user only hums the melody. A hum is usually:

- short
- imperfect
- sometimes off-key
- different in tempo
- missing instruments and vocals
- affected by microphone quality or background noise

Because of this, Hum2Tune does not use normal audio fingerprinting. Instead, it focuses on melody matching.

The final system uses a **vocal-only Dynamic Time Warping (DTW) retrieval pipeline**. It compares the hummed input with vocal-focused reference features and returns ranked song candidates.

---

## 2. Important: Dataset Is Not Included

The dataset and original song files are **not included** in this submission ZIP.

This is because audio datasets and music files can be large, and some song files may have copyright or redistribution restrictions. To run the full system, users need to download and prepare the data themselves.

The main humming dataset used in this project is available here:

**Query-by-Humming (QBH) Audio Dataset**  
https://www.kaggle.com/datasets/limzhiminjessie/query-by-humming-qbh-audio-dataset

Please download the dataset from Kaggle before running the full project.

You also need to prepare your own original reference songs if you want to test retrieval against a song database.

---

## 3. Main System Features

Hum2Tune includes:

- humming audio input
- audio upload support
- browser-based recording through Streamlit
- audio preprocessing
- pitch extraction using pYIN
- chroma feature extraction
- source-separated vocal reference matching
- Dynamic Time Warping retrieval
- ranked song candidate output
- CNN-LSTM baseline model
- Audio Transformer baseline model
- DualEncoder exploratory retrieval model
- evaluation metrics such as Top-1, Top-3, Top-5, MRR, MAP@10, and NDCG@10
- report figure generation scripts

The main recognition engine is the **vocal-only DTW retrieval pipeline**.

The neural models are included for comparison and analysis. They are not the final 100-song prediction engine.

---

## 4. Project Folder Structure

The project is organised like this:

```text
Hum2Tune_FYP/
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
│
├── config/
│   └── configuration files
│
├── data/
│   ├── Humming Audio/
│   │   └── downloaded humming dataset folders go here
│   │
│   ├── Original Songs/
│   │   └── reference song audio files go here
│   │
│   └── processed/
│       └── datasets/
│           ├── songs.csv
│           ├── queries.csv
│           ├── splits.csv
│           └── classes.json
│
├── demucs_output/
│   └── htdemucs/
│       └── separated vocals and no_vocals stems go here
│
├── models/
│   └── saved model checkpoints
│
├── outputs/
│   └── retrieval outputs and ranked candidate files
│
├── results/
│   ├── evaluations/
│   └── visualizations/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── validate_project.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── hybrid_retrieval.py
│   ├── evaluate_retrieval.py
│   ├── graph.py
│   └── report_generator.py
│
└── src/
    └── fyp_title11/
        ├── data/
        ├── evaluation/
        ├── models/
        └── tokenization/
```

Some folders may be empty when the ZIP is first opened. They will be filled after preparing the dataset, running retrieval, training models, or generating results.

---

## 5. How to Set Up the Dataset

### Step 1: Download the humming dataset

Go to:

```text
https://www.kaggle.com/datasets/limzhiminjessie/query-by-humming-qbh-audio-dataset
```

Download and extract the dataset.

The dataset contains folders named after songs. Each folder contains hummed audio files.

Example:

```text
A Thousand Years/
Back To December/
Sunflower/
All Of Me/
Blank Space/
```

Place these folders inside:

```text
data/Humming Audio/
```

Example:

```text
data/Humming Audio/A Thousand Years/
data/Humming Audio/Back To December/
data/Humming Audio/Sunflower/
```

---

### Step 2: Add original reference songs

The project also needs reference songs for retrieval.

Place your original song files inside:

```text
data/Original Songs/
```

Example:

```text
data/Original Songs/A Thousand Years.wav
data/Original Songs/Back To December.wav
data/Original Songs/Sunflower.wav
```

The song names should match the humming folders as closely as possible.

For the final project evaluation, the database contained 100 reference songs. For basic testing, you can start with a smaller number of songs.

---

### Step 3: Prepare vocal stems with Demucs

Hum2Tune works best with vocal-separated references.

The expected structure is:

```text
demucs_output/htdemucs/<song_name>/vocals.wav
demucs_output/htdemucs/<song_name>/no_vocals.wav
```

Example:

```text
demucs_output/htdemucs/Sunflower/vocals.wav
demucs_output/htdemucs/Sunflower/no_vocals.wav
```

To separate one song using Demucs:

```powershell
demucs -n htdemucs "data/Original Songs/Sunflower.wav" -o demucs_output
```

To separate all songs in a folder:

```powershell
demucs -n htdemucs "data/Original Songs" -o demucs_output
```

This may take some time depending on your computer.

---

## 6. Python Environment Setup

This project was developed using Python and a virtual environment.

Recommended Python version:

```text
Python 3.10 or above
```

Open PowerShell in the project folder:

```powershell
cd "C:\Users\User\Downloads\Hum2Tune_FYP"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If it works, you should see:

```text
(.venv)
```

at the start of your PowerShell line.

---

## 7. Install Required Packages

Install the required packages:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main packages manually:

```powershell
pip install numpy pandas librosa soundfile matplotlib scikit-learn torch streamlit tqdm demucs
```

Optional package:

```powershell
pip install crepe tensorflow
```

CREPE is optional. If it is not installed, you may see this warning:

```text
Warning: CREPE not found. Install 'crepe' and 'tensorflow' for best results.
```

This warning does not stop the main pYIN-based retrieval pipeline from running.

---

## 8. Validate the Project

Before running the system, check that the project structure is correct:

```powershell
python .\scripts\validate_project.py
```

This script checks whether the important folders and files exist.

If validation fails, check that:

- the Kaggle dataset has been downloaded
- humming folders are inside `data/Humming Audio/`
- reference songs are inside `data/Original Songs/`
- Demucs vocal stems exist inside `demucs_output/`
- dataset files such as `songs.csv`, `queries.csv`, and `splits.csv` exist

---

## 9. Prepare the Dataset

Run:

```powershell
python .\scripts\prepare_dataset.py
```

or:

```powershell
python main.py prepare
```

This step prepares the dataset files used by training and evaluation.

Expected files include:

```text
data/processed/datasets/songs.csv
data/processed/datasets/queries.csv
data/processed/datasets/splits.csv
data/processed/datasets/classes.json
```

---

## 10. Run the Main Retrieval Pipeline

The main retrieval mode is:

```text
vocal_only
```

Run:

```powershell
python main.py retrieve --mode vocal_only --top-k 5 --shortlist 10
```

This compares hummed queries against the reference database and returns ranked candidate songs.

To check the available options:

```powershell
python main.py retrieve --help
```

Retrieval outputs are saved in:

```text
outputs/
results/
```

---

## 11. Run the Web Application

Start the Streamlit app:

```powershell
streamlit run app.py
```

If that does not work, use:

```powershell
python -m streamlit run app.py
```

The app should open in your browser.

If it does not open automatically, copy the local URL shown in PowerShell. It usually looks like this:

```text
http://localhost:8501
```

Paste it into your browser.

---

## 12. How to Use the Web App

The app has two input options:

1. Record yourself humming
2. Upload an audio file

### Record mode

1. Click **Record yourself humming**.
2. Click **Start Recording**.
3. Hum for at least 5 seconds.
4. Stop recording.
5. Wait for the result.
6. The app will show ranked song candidates.

### Upload mode

1. Click **Upload an audio file**.
2. Upload a humming file.
3. Wait for the app to process it.
4. Review the ranked candidate songs.

Supported file types usually include:

```text
.wav
.mp3
.m4a
.flac
.ogg
```

The app may reject the input if the recording is too short, too quiet, or cannot be processed.

---

## 13. Recommended Demo Test

For a quick demo, use songs that exist in both the humming dataset and your reference database.

Good examples:

```text
Sunflower
A Thousand Years
Back To December
```

For the cleanest demo:

1. Open the web app.
2. Upload or record a short hum.
3. Show the ranked result list.
4. Explain that the system returns top candidate songs rather than only one fixed prediction.

---

## 14. Train the Neural Baselines

The neural models are included for comparison.

They are not the final 100-song recognition engine.

### Train CNN-LSTM

```powershell
python main.py train --model cnn_lstm
```

### Evaluate CNN-LSTM

```powershell
python main.py evaluate --model cnn_lstm
```

### Train Audio Transformer

```powershell
python main.py train --model audio_transformer
```

### Evaluate Audio Transformer

```powershell
python main.py evaluate --model audio_transformer
```

Model checkpoints are saved in:

```text
models/
```

Evaluation results are saved in:

```text
results/evaluations/
```

Figures are saved in:

```text
results/visualizations/
```

---

## 15. Generate Report Figures

To generate the report graphs:

```powershell
python .\scripts\graph.py
```

Expected outputs include:

```text
results/visualizations/fig_5_1_retrieval_topk_accuracy.png
results/visualizations/fig_5_2_retrieval_ranking_quality.png
results/visualizations/fig_5_3_neural_baseline_comparison.png
```

Training curve figures may also be generated if the training history files are available.

---

## 16. Final Report Results

The final report used:

```text
205 humming queries
100 reference songs
```

The strongest retrieval configuration was:

```text
vocal_only
```

Final reported performance:

```text
Top-1 accuracy: 37.56%
Top-3 accuracy: 61.95%
Top-5 accuracy: 69.76%
MRR: 0.505
```

These results show that the correct song is not always ranked first, but it often appears within the top few candidates. This is why Hum2Tune is designed as a ranked retrieval system.

---

## 17. Common Problems and Fixes

### Problem: PowerShell cannot activate the virtual environment

Try:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

### Problem: `classes.json` is missing

Run:

```powershell
python .\scripts\prepare_dataset.py
```

or:

```powershell
python main.py prepare
```

---

### Problem: no model checkpoint found

If you see an error like:

```text
No checkpoint found at models/cnn_lstm/best_model.pth
```

Train the model first:

```powershell
python main.py train --model cnn_lstm
```

Then evaluate:

```powershell
python main.py evaluate --model cnn_lstm
```

---

### Problem: Streamlit is not recognised

Install Streamlit:

```powershell
pip install streamlit
```

Then run:

```powershell
python -m streamlit run app.py
```

---

### Problem: the app is slow

The app may be slow if reference features are being generated again.

To improve speed:

- make sure Demucs outputs already exist
- make sure reference features are cached
- run retrieval once before using the app
- use a smaller shortlist during demo

Example:

```powershell
python main.py retrieve --mode vocal_only --top-k 5 --shortlist 10
```

---

### Problem: CREPE warning appears

If you see:

```text
Warning: CREPE not found
```

you can ignore it for the main pYIN-based retrieval pipeline.

CREPE is optional.

---

## 18. Limitations

Hum2Tune is a prototype, so it has some limitations:

- it only searches inside the prepared database
- it does not search Spotify, YouTube, or the internet
- it depends on the quality of the hum
- it depends on the quality of vocal separation
- DTW retrieval can become slow for very large databases
- the neural models need more balanced humming data
- the system does not currently use lyrics or metadata

---

## 19. Future Improvements

Future work could improve Hum2Tune by:

- collecting more humming recordings for each song
- using more singers and recording conditions
- improving pitch tracking
- adding better tempo and key handling
- improving the DualEncoder learned retrieval model
- using faster search methods for larger databases
- adding user feedback and correction options
- improving the web app interface

---

## 20. What Is Included in This Submission

This submission includes the source code and project files needed to run Hum2Tune.

It does not include:

- the Kaggle humming dataset
- original song files
- copyrighted reference audio
- large generated audio outputs

Please download the humming dataset manually from Kaggle:

```text
https://www.kaggle.com/datasets/limzhiminjessie/query-by-humming-qbh-audio-dataset
```

Then prepare your own reference songs and place them in the correct folders.

---

## 21. Quick Start

For users who already have the dataset and reference songs prepared:

```powershell
cd "C:\Users\User\Downloads\Hum2Tune_FYP"

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python .\scripts\validate_project.py

python main.py retrieve --mode vocal_only --top-k 5 --shortlist 10

python -m streamlit run app.py
```

Then open the browser app and test a hummed query.

---

## 22. Author

```text
Name: Wong Xuan Kai
Student Number: 20495251
Programme: BSc Computer Science with Artificial Intelligence
Project Title: Hum2Tune: Melody-Based Music Recognition App Using Frequency Analysis
Supervisor: Dr Simon Lau
University: University of Nottingham Malaysia
Year: 2026
```