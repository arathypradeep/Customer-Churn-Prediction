# Customer Churn Prediction

## Description
This repository contains a machine learning-based customer churn prediction system. It includes exploratory data analysis (EDA), multiple classification models (KNN, SVM, Decision Tree, Naïve Bayes), and a deployment script for predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Customer-Churn-Prediction.git
   ```
2. Navigate into the project directory:
   ```sh
   cd Customer-Churn-Prediction
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the Jupyter notebooks for EDA and model training using:
```sh
jupyter notebook
```
Run the deployment script:
```sh
python churnnnapp.py
```

## Project Structure
```
📦 Customer-Churn-Prediction
├── churnnnapp.py                     # Deployment script for churn prediction
├── EDA_PROJECT.ipynb                  # Exploratory Data Analysis (EDA)
├── churnnn.pkl                        # Saved churn prediction model
├── KNN,SVM,DT,NB.ipynb                # Machine learning models (KNN, SVM, Decision Tree, Naïve Bayes)
├── images/
│   ├── churn.jpg                      # Visualization image
│   ├── churn_r.jpg                    # Additional visualization
├── README.md                          # Project README
├── requirements.txt                   # Python dependencies
```

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

