# Description of the project
"""
In this project, we are working with the "Smart Meters in London" dataset to predict power consumption.
The goal is to forecast future power usage based on the data provided by smart meters.

Data Source: https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london

Steps taken:
1. **Data Preprocessing**:
   - Loaded the dataset and handled missing values by dropping rows with NaN values.
   - Separated features (X) and target (y) from the dataset, where 'energy_sum' is the column to predict.

2. **Feature Scaling**:
   - Scaled the features using StandardScaler to standardize the data, ensuring that the model works efficiently with features on different scales.

3. **Model Building**:
   - Built a **Neural Network (NN)** model using TensorFlow/Keras with two hidden layers and dropout to prevent overfitting.
   - The model is compiled with the Adam optimizer and mean squared error (MSE) loss.

4. **Training and Evaluation**:
   - The NN model was trained on the training data for 100 epochs, with a batch size of 32 and a validation split of 20%.
   - The performance of the NN model was evaluated using MSE, MAE (mean absolute error), and R2 score.

5. **Alternative Models**:
   - Compared the performance of the NN model with other machine learning models:
     - **Linear Regression**
     - **Support Vector Regressor (SVR)**
     - **Random Forest Regressor**

   - Each model was evaluated using the same metrics (MSE, MAE, R2) to determine the best-performing model.

6. **Visualization**:
   - A bar plot was created to compare the MSE scores of all models, providing a clear visual representation of their performance.

Next steps involve fine-tuning the best-performing model and applying it to real-world data for prediction.
"""
