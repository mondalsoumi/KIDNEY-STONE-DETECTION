🫘 Kidney Stone Detection

A Python-based medical imaging project that detects kidney stones in Ultrasound and CT scan images using classical Image Processing and Machine Learning techniques.

📌 Overview

Kidney stones are a common and painful medical condition. Early and accurate detection from medical images can significantly aid diagnosis. This project applies a pipeline of image processing algorithms to identify potential kidney stone regions in grayscale medical scans.

🧪 Techniques Used

Gabor Filtering: texture feature extraction from ultrasound images

Histogram Equalization: contrast enhancement for better visibility

Watershed Segmentation: region-based segmentation to isolate stones

Gaussian Filtering: noise reduction and image smoothing

Median & Laplacian Filtering: edge sharpening and speckle removal


📁 Project Structure

├── Finalcode.py         # Main detection pipeline

├── Gabor_HistEq.py      # Gabor filter + Histogram Equalization

├── WaterShed.py         # Watershed segmentation

├── gaussian.py          # Gaussian smoothing

├── med_lap.py           # Median and Laplacian filtering

├── sg.py                # Supplementary processing

├── read_write.py        # Image I/O utilities

└── Installation.md      # Setup instructions

⚙️ Installation

Check Installation.md for setup instructions.
Dependencies: Python 3.x, OpenCV, NumPy, SciPy, Matplotlib
pip install opencv-python numpy scipy matplotlib

🚀 Usage

python Finalcode.py
Provide the path to your ultrasound or CT scan image when prompted. The output will display the processed image with detected kidney stone regions highlighted.

📚 References

Hindawi – VLSI Journal
IEEE – Kidney Stone Detection
Level Set Method
Data Augmentation Guide


📄 License
This project is open source. Feel free to use and contribute.
