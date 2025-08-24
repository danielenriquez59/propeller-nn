# Propeller Performance Neural Network Predictor

## Purpose

This project provides a tool to predict the performance characteristics of a propeller based on its geometric properties and operating conditions. The primary purpose is to offer a fast, interactive way to evaluate propeller designs without the need for complex simulations or physical wind tunnel tests.

It uses a PyTorch-based neural network (a 1D Convolutional Neural network) to learn the relationship between propeller geometry (radius, chord, twist) and its resulting performance coefficients (Thrust Coefficient CT, Power Coefficient CP) across a range of advance ratios (J).

Current training data is based on the UIUC APC Propeller Database.

## Key Features

- **Interactive App (Panel + Bokeh)**: A browser-based UI built with Panel and Bokeh that allows real-time manipulation of propeller geometry and immediate visualization of predicted performance.
- **Neural Network Model**: A CNN-based model (`PropNet`) that processes geometric distributions to predict performance curves.
- **Modular Codebase**: The core model logic, training scripts, and GUI are separated for maintainability and clarity.
- **Export to OpenVSP**: Design can be exported to a `.bem` file and optionally opened directly in OpenVSP via its Python API.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages (recommended in a conda/virtual env):**
    ```bash
    pip install -r requirements.txt
    ```

4.  (Optional) **OpenVSP Integration**
    - Install the OpenVSP Python API according to your OpenVSP installation. See the README in your OpenVSP install directory.
    - Ensure the `openvsp` module is importable by Python if you plan to export and launch OpenVSP from the app.

## Usage

### 1. Clean the UIUC Propeller Data

Run the data cleaning/processing to aggregate the UIUC propeller database:

```bash
python data-analysis/process_uiuc_data.py
```

This produces combined datasets used for training.

### 2. Train the Model

To retrain the model with new data, you will need to prepare your data and then run the training script:

```bash
python learn_apc.py
```
This script will read the data, train the `PropNet` model, and save the trained weights and scalers to the `model_weights/` directory.

### 3. Interactive Prediction (Panel GUI)

Serve the browser-based app (Panel + Bokeh):

```bash
panel serve gui_panel.py --autoreload
```

In the app, you can:
- Drag the five control points for chord and twist to shape the geometry.
- Set the propeller's diameter (inches) and number of blades.
- Click "Predict Performance" to plot `CT`, `CP`, and efficiency/10 versus advance ratio `J`.

### 4. Export to OpenVSP

In the "Export / OpenVSP" section of the app:
- Enter a filename (extension `.bem` is added automatically).
- Click "Write BEM" to write the propeller geometry to a BEM file.
- Optionally check "Run OpenVSP after write" to automatically launch OpenVSP and import the BEM file (requires the OpenVSP Python API).

### 5. Command-Line Inference (optional)

To run a single prediction from the command line using the default example data in the script:

```bash
python run_model.py
```

## File Structure

0. `requirements.txt`: A list of all Python dependencies for the project.
1. `data-analysis/process_uiuc_data.py`: Cleans and aggregates the UIUC propeller database.
2. `nn_functions.py`: Core functions for loading the model and making predictions.
3. `learn_apc.py`: The script used to train the neural network model.
4. `model_weights/`: Directory containing trained model weights (`propnet_weights.pt`), metadata (`meta.json`), and scalers (`.joblib`).
5. `run_model.py`: A command-line script to perform a single prediction.
6. `gui_panel.py`: Panel + Bokeh app for interactive design and prediction.
7. `geometry/create_geomety.py`: Writes `.bem` files from chord/twist distributions.
8. `geometry/api_openvsp.py`: OpenVSP Python API integration to import `.bem` and launch the GUI.

## Propeller Design GUI
The Propeller Design GUI provides an intuitive interface for users to design and evaluate propeller performance. With this GUI, users can easily adjust the chord and twist distributions of a propeller by dragging control points on the plot. The interface allows for real-time updates, enabling users to instantly visualize the effects of their modifications on the thrust (`CT`) and power (`CP`) coefficients. Additionally, users can specify the propeller's diameter and the number of blades, making it a versatile tool for various design scenarios. The interactive plot supports zooming, panning, and hovering over data points to display detailed values, enhancing the user experience and providing valuable insights into propeller performance characteristics.

![Propeller Design GUI Screenshot](images/gui_screenshot.png)

