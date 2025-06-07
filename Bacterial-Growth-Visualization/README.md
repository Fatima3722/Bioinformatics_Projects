# Bacterial-Growth-Visualization

This project demonstrates how to read tabular data from a CSV file representing bacterial growth measurements and generate a line plot visualizing the growth curves of different bacterial strains over time. The plot is created using Python with the `pandas` library for data manipulation, and `matplotlib` with `seaborn` for plotting.

This is a foundational project ideal for showcasing basic data handling, scripting, and visualization skills relevant to biology and bioinformatics.

## Technology Stack

*   **Language:** Python 3
*   **Libraries:**
    *   `pandas`: For reading and structuring the CSV data.
    *   `matplotlib`: For creating the basic plot structure.
    *   `seaborn`: For enhancing the plot aesthetics and simplifying multi-line plotting.

## Files in this Repository

*   `plot_growth.py`: The Python script that reads the data, generates the plot, and saves it.
*   `Bacterial_growth_.csv`: The input data file containing time points, strain identifiers, and optical density (OD600) measurements.
*   `bacterial_growth_plot.png`: The output image file of the generated plot.
*   `README.md`: This documentation file.

## Data Format (`bacterial_growth.csv`)

The input CSV file (`bacterial_growth.csv`) is expected to have the following columns:

1.  `Time_Hours`: Numeric values representing the time in hours at which measurements were taken.
2.  `Strain`: Character or string values identifying the bacterial strain (e.g., "Strain_A", "Strain_B").
3.  `OD600`: Numeric values representing the Optical Density at 600nm, a proxy for bacterial concentration.

## Prerequisites
Before running the script, ensure you have Python 3 installed on your system. You will also need to install the required Python libraries. You can install them using pip:

pip install pandas matplotlib seaborn

## How to Run
1. Clone or Download:
Clone this repository: git clone https://github.com/Fatima3722/bacterial-growth-visualization.git (replace your-username and repository name if different)
Alternatively, download the files (plot_growth.py and bacterial_growth.csv) into a single directory on your computer.
2. Navigate to the Directory:
Open your terminal or command prompt and navigate to the directory where you saved the files.
3. Execute the Script:
Run the Python script from the terminal:
python plot_growth.py
4. View Output:
The script will generate (or overwrite) an image file named bacterial_growth_plot.png in the same directory. The console will also print a confirmation message: "Plot saved as bacterial_growth_plot.png".


