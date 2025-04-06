Face Detection and Recognition – End-to-End Machine Learning Project
This project is an end-to-end implementation of a face detection application using Histogram of Oriented Gradients (HOG) features and classic machine learning algorithms like K-Nearest Neighbors, Decision Trees, and Random Forests. The entire pipeline includes feature extraction, classification, evaluation, face localization, and deployment using Streamlit.

📁 Dataset
LFW (Labeled Faces in the Wild) for positive face samples.

Various grayscale images from Scikit-Image for negative (non-face) samples.

🚀 Workflow
1. Feature Engineering
Extracted HOG features from positive and negative patches.

Visualized samples of both classes.

Created a dataset X and labels y.

2. Binary Classification
Split the dataset into train and test sets.

Performed Grid Search on:

KNeighborsClassifier

DecisionTreeClassifier

RandomForestClassifier

Evaluated performance and visualized accuracy scores with Seaborn.

3. Model Evaluation
Used accuracy, precision, recall, and f1-score to evaluate the best model on test data.

4. Face Detection on a New Image
Applied sliding window technique on a new image.

Predicted face locations and drew bounding boxes.

5. Model Deployment
Deployed the final model using Streamlit for a simple web interface.

📊 Results
The model with the highest accuracy was selected and used for final predictions.

Achieved effective face detection using classic ML techniques and HOG descriptors.

📦 Requirements
bash
Copier
Modifier
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image streamlit
▶️ Run the Project
To run the detection demo with Streamlit:

bash
Copier
Modifier
streamlit run app.py
📌 Conclusion
This lab provides a hands-on understanding of:

How to engineer features with HOG.

How to train and compare multiple classifiers.

How to implement a face detection system from scratch.

How to deploy ML models using Streamlit.
