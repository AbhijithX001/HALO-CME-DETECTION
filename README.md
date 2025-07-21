
# 🛰️ HALO CME Project: Solar Storm Detection Mission

![Solar Flare](https://placehold.co/800x200/000000/FFFFFF?text=HALO+CME+DETECTION+SYSTEM)

**Mission Control:** Welcome, Analyst. This repository contains the operational source code for the Halo Coronal Mass Ejection (CME) Detection Project. Our directive is to automate the surveillance of near-Earth space by processing and analyzing complex satellite telemetry. The system is engineered to ingest heterogeneous data formats from multiple solar observatories, synthesize them into a coherent time-series dataset, and apply a tunable anomaly detection algorithm to identify and flag significant solar storm events.

---

## 🚀 Mission Briefing

A Coronal Mass Ejection (CME) represents one of the most energetic phenomena in our solar system—a colossal expulsion of plasma and magnetic fields from the Sun's corona. When a CME's trajectory intersects with Earth, the resulting geomagnetic storm can disrupt the delicate infrastructure of our modern world, threatening satellites, power grids, and critical communication systems.

The core objective of this project is to create an early-warning system by identifying the tell-tale signatures of a CME's arrival in satellite data. The algorithm is trained to detect coincident, anomalous spikes across multiple independent instruments:

* **Plasma Analyzers:** Monitoring for sudden increases in **proton density and flux**.
* **Magnetometers:** Searching for disturbances in the total magnitude of the **interplanetary magnetic field**.
* **Spectrographs:** Analyzing the integrated energy from **particle detectors**.

By fusing these data streams, the system can achieve a higher confidence in its detections than by relying on any single instrument alone.

---

## 🔭 Project Philosophy & Architecture

This project was built with the realities of scientific data analysis in mind: data is often messy, inconsistent, and comes from a variety of sources.

1.  **Data Ingestion & Normalization:** The first stage is the most complex. The script is designed to be a "universal translator" for the various `.cdf` and `.nc` files provided by different instruments. It automatically detects the correct variable names for time and data, handles multi-dimensional arrays (like spectrograms), and processes massive files in memory-efficient chunks. The end goal is to transform all raw data into a single, standardized "Flux" column indexed by time.
2.  **Signal Processing:** The raw, combined data is noisy. To establish a stable baseline, a rolling mean (`SMOOTHING_WINDOW`) is applied. This smooths out minor fluctuations, allowing the larger, more significant spikes to stand out.
3.  **Anomaly Detection:** The core of the detection logic is a statistical approach. The system calculates the mean and standard deviation of the smoothed data and flags any period where the flux exceeds a user-defined threshold (`FLUX_STD_MULTIPLIER`). It then filters these potential events, keeping only those that persist for a minimum duration (`MIN_EVENT_DURATION_MINUTES`).

---

## 🔧 System Requirements (Dependencies)

Before launch, ensure your analysis environment is equipped with the necessary modules.

```bash
pip install pandas numpy cdflib netCDF4
````

-----

## 🛡️ Data Integrity (`.gitignore`)

This is a data-intensive project. To maintain a clean, lightweight, and reproducible code repository, it is **critical** to exclude raw and processed data from version control.

Create a `.gitignore` file in the root of your project directory with the following contents:

```gitignore
# Ignore all raw data files
raw_data/

# Ignore all processed data and logs
processed_data/
logs/

# Ignore Python virtual environment folders
venv/
*.env

# Ignore Python cache files
__pycache__/
*.pyc
```

This ensures that only the source code is tracked, which is the standard best practice.

-----

## ⚡ Launch Sequence (How to Run)

1.  **Navigate to Mission Control:**
    Open your terminal and navigate to the project folder.

    ```bash
    cd "C:\Path\To\Your\HALO CME PROJECT"
    ```

2.  **Engage Main Thruster:**
    Execute the main Python script.

    ```bash
    python main_code.py
    ```

The system will provide a real-time log, showing each file as it is processed. The full run may take several minutes.

-----

## ⚙️ Tuning the Sensors (Analyst's Task)

The most critical part of this mission is **calibrating the detection algorithm**. A machine can only find what it's told to look for. Your role as the analyst is to fine-tune the parameters in `main_code.py` to match the characteristics of the events you're hunting.

```python
# A lower value will make the detection MORE sensitive.
# Try values like 2.5, 2.2, or 2.0 to find weaker events.
FLUX_STD_MULTIPLIER = 2.5

# A lower value will allow the system to flag shorter events.
# Try values like 4 or 3 to see if the September CMEs were short-lived in the data.
MIN_EVENT_DURATION_MINUTES = 5
```

**Analyst's Recommendation:** Official reports confirm significant CME activity between **September 12-17, 2024**. Our initial run with default settings (`FLUX_STD_MULTIPLIER = 3.0`) did not detect these events, indicating they were not extreme outliers in this dataset.

**Your next mission:**

1.  Begin by lowering the `FLUX_STD_MULTIPLIER` incrementally (e.g., to `2.5`, then `2.2`).
2.  If events are still not detected, consider that they may have been shorter in duration. Lower the `MIN_EVENT_DURATION_MINUTES` to `4` or `3`.
3.  Cross-reference any detected events with the known dates to validate your settings.

This iterative process of tuning and validation is the core of scientific discovery.

-----

## 🌌 Future Missions (Potential Enhancements)

This system provides a robust foundation. Future missions could include:

  * **Data Visualization:** Implementing `matplotlib` or `seaborn` to automatically generate plots of the full dataset and any detected events.
  * **Advanced Detection Models:** Moving beyond statistical thresholds to more advanced machine learning models (e.g., Isolation Forests, LSTMs) for anomaly detection.
  * **Multi-Variable Analysis:** Modifying the script to analyze not just a single "Flux" value, but to look for coincident spikes across proton density, speed, and magnetic field strength independently.

-----

