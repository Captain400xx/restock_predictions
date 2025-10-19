# RestockR Predictions

A Streamlit dashboard that provides Pokémon Card restock predictions and visualizes model feature importance.



---

## Features

* **Alert Countdown:** Displays a live countdown to the next data refresh or alert trigger.
* **Model Feature Importance:** Visualizes the key factors driving the model's predictions.
* **Interactive Controls:** A sidebar with controls to filter, sort, or update the data being displayed.
* 4 model predictions

---

## How to Run

Follow these instructions to get a local copy up and running.

### Prerequisites

* Python 3.8+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Captain400xx/restock_predictions].git
    cd [restock_predictions]
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

Once all dependencies are installed, run the following command from your project's root directory:

```bash

streamlit run app2.py
