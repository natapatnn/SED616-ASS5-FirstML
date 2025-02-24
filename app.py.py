{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "54e79383-4b63-4512-a267-d13d4e2ea510",
   "metadata": {},
   "source": [
    "# Assignment 5:\n",
    "### รหัสนักศึกษา: 67130701702\n",
    "### ชื่อ-นามสกุล: Natapat Na Ubon\n",
    "### หลักสูตร: SED"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "edfb5156-a29e-4c3f-a9a2-e5bce25b38de",
   "metadata": {},
   "source": [
    "## Imbalanced Data Classification & Model Deployment\n",
    "\n",
    "- Understand the challenges of imbalanced classification.\n",
    "- Train different models with various resampling techniques.\n",
    "- Compare model performance using ROC and PR curves.\n",
    "- Deploy the best-performing model using Streamlit.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7000bb50-5d91-4888-ae33-c0add964ca05",
   "metadata": {},
   "source": [
    "#### 1. Install Required Libraries\n",
    "\n",
    "Ensure you have the necessary libraries installed:\n",
    "\n",
    "pip install imbalanced-learn scikit-learn matplotlib seaborn streamlit\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aee2f45c-3681-4b87-916c-eae00678adec",
   "metadata": {},
   "source": [
    "## **2. Load and Explore the Dataset**  \n",
    "Select a dataset from [`imbalanced-learn datasets`](https://imbalanced-learn.org/stable/datasets/index.html). Example: `fetch_datasets` provides multiple datasets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "920e898a-c2bd-4004-87c0-3c0422b82b58",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Target\n",
      "-1    4715\n",
      " 1     183\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "from imblearn.datasets import fetch_datasets\n",
    "import pandas as pd\n",
    "\n",
    "# Load an imbalanced dataset (modify as needed)\n",
    "dataset = fetch_datasets()['wine_quality']  # Example: 'wine_quality'\n",
    "X, y = dataset.data, dataset.target\n",
    "\n",
    "# Convert to DataFrame\n",
    "df = pd.DataFrame(X, columns=[f\"Feature_{i}\" for i in range(X.shape[1])])\n",
    "df['Target'] = y\n",
    "\n",
    "# Check class distribution\n",
    "print(df['Target'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "866ba4df-669a-4bf2-bc13-79684443b757",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# สมมติว่าคุณมีข้อมูลในรูปแบบของ dictionary หรือ DataFrame\n",
    "data = {\n",
    "    \"Feature_0\": 0.03,\n",
    "    \"Feature_1\": -0.02,\n",
    "    \"Feature_2\": 0.00,\n",
    "    \"Feature_3\": 0.00,\n",
    "    \"Feature_4\": 0.00,\n",
    "    \"Feature_5\": 0.00,\n",
    "}\n",
    "\n",
    "# สร้างกราฟ\n",
    "fig, ax = plt.subplots()\n",
    "ax.bar(data.keys(), data.values())\n",
    "\n",
    "# แสดงผลกราฟใน Streamlit\n",
    "st.pyplot(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dfc19be1-f007-442c-a1d1-f18b74ed1005",
   "metadata": {},
   "source": [
    "## **3. Train-Test Split**  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e6649e83-5bd1-4e94-a511-655556c8dc44",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Split data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c6ff86e6-1f2e-48c8-9c74-e10a2f505f87",
   "metadata": {},
   "source": [
    "## **4. Train Models**  \n",
    "\n",
    "### **4.1 Baseline Model (Logistic Regression)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "fbbb2681-ebcb-4c06-b6ff-353e15bdb1b3",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline Model (Logistic Regression) Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "          -1       0.96      1.00      0.98      1415\n",
      "           1       0.50      0.02      0.04        55\n",
      "\n",
      "    accuracy                           0.96      1470\n",
      "   macro avg       0.73      0.51      0.51      1470\n",
      "weighted avg       0.95      0.96      0.95      1470\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "# Train a baseline model (Logistic Regression)\n",
    "base_model = LogisticRegression(random_state=42, max_iter=1000)\n",
    "base_model.fit(X_train, y_train)\n",
    "\n",
    "# Predictions\n",
    "y_pred = base_model.predict(X_test)\n",
    "print(\"Baseline Model (Logistic Regression) Classification Report:\")\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b48afc9b-77d6-425e-bb65-da3fa7a1ef20",
   "metadata": {},
   "source": [
    "### **4.2 Model with Undersampling (Logistic Regression)**  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "df4ef159-bc7d-430d-abe8-96fb9b59cb7c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Undersampling Model (Logistic Regression) Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "          -1       0.98      0.73      0.84      1415\n",
      "           1       0.08      0.60      0.14        55\n",
      "\n",
      "    accuracy                           0.73      1470\n",
      "   macro avg       0.53      0.67      0.49      1470\n",
      "weighted avg       0.95      0.73      0.81      1470\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "\n",
    "# Apply undersampling\n",
    "rus = RandomUnderSampler(random_state=42)\n",
    "X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)\n",
    "\n",
    "# Train Logistic Regression model\n",
    "under_model_lr = LogisticRegression(random_state=42, max_iter=1000)\n",
    "under_model_lr.fit(X_train_rus, y_train_rus)\n",
    "\n",
    "# Predictions\n",
    "y_pred_under = under_model_lr.predict(X_test)\n",
    "print(\"Undersampling Model (Logistic Regression) Classification Report:\")\n",
    "print(classification_report(y_test, y_pred_under))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "214e7597-d1c2-49c9-a955-a3b55583dd78",
   "metadata": {},
   "source": [
    "### **4.3 Model with Oversampling (Logistic Regression)**  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "105b5104-f8bb-4196-a1bd-10859d59c6f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Oversampling Model (Logistic Regression) Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "          -1       0.99      0.76      0.86      1415\n",
      "           1       0.10      0.71      0.18        55\n",
      "\n",
      "    accuracy                           0.76      1470\n",
      "   macro avg       0.54      0.74      0.52      1470\n",
      "weighted avg       0.95      0.76      0.83      1470\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from imblearn.over_sampling import RandomOverSampler\n",
    "\n",
    "# Apply oversampling\n",
    "ros = RandomOverSampler(random_state=42)\n",
    "X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)\n",
    "\n",
    "# Train Logistic Regression model\n",
    "over_model_lr = LogisticRegression(random_state=42, max_iter=1000)\n",
    "over_model_lr.fit(X_train_ros, y_train_ros)\n",
    "\n",
    "# Predictions\n",
    "y_pred_over = over_model_lr.predict(X_test)\n",
    "print(\"Oversampling Model (Logistic Regression) Classification Report:\")\n",
    "print(classification_report(y_test, y_pred_over))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be32d6bf-6c9e-4c78-bb3e-44439a0c8737",
   "metadata": {},
   "source": [
    "### **4.4 Random Forest Model (No Resampling)**  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9d3335d2-80df-4bde-9576-b5c8de9e42af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Model Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "          -1       0.97      1.00      0.98      1415\n",
      "           1       0.85      0.20      0.32        55\n",
      "\n",
      "    accuracy                           0.97      1470\n",
      "   macro avg       0.91      0.60      0.65      1470\n",
      "weighted avg       0.97      0.97      0.96      1470\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Train Random Forest model\n",
    "rf_model = RandomForestClassifier(random_state=42)\n",
    "rf_model.fit(X_train, y_train)\n",
    "\n",
    "# Predictions\n",
    "y_pred_rf = rf_model.predict(X_test)\n",
    "print(\"Random Forest Model Classification Report:\")\n",
    "print(classification_report(y_test, y_pred_rf))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c35653e-acec-491f-83a7-f0b5ef32a1c1",
   "metadata": {},
   "source": [
    "## **5. Compare Model Performance**  \n",
    "\n",
    "### **5.1 Plot ROC Curve & ROC-AUC Score**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a7c53e95-5473-4ca8-b91f-5288fd0d84a1",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ASUS Flip Laptop\\AppData\\Local\\Temp\\ipykernel_21948\\825336759.py:24: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show()\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import roc_curve, auc\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "models = {\n",
    "    \"Logistic Regression\": base_model,\n",
    "    \"Undersampling (Logistic Regression)\": under_model_lr,\n",
    "    \"Oversampling (Logistic Regression)\": over_model_lr,\n",
    "    \"Random Forest\": rf_model\n",
    "}\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "\n",
    "for name, model in models.items():\n",
    "    y_prob = model.predict_proba(X_test)[:, 1]\n",
    "    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)\n",
    "    roc_auc = auc(fpr, tpr)\n",
    "    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')\n",
    "\n",
    "plt.plot([0, 1], [0, 1], linestyle=\"--\", color=\"gray\")\n",
    "plt.xlabel(\"False Positive Rate\")\n",
    "plt.ylabel(\"True Positive Rate\")\n",
    "plt.title(\"ROC Curve\")\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f35dd09-ad10-4c7a-86d6-bee0695a15df",
   "metadata": {},
   "source": [
    "### **5.2 Plot PR Curve & PR-AUC Score**  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "0fc55b5b-0eb7-46cd-8b03-1131765cdcf4",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ASUS Flip Laptop\\AppData\\Local\\Temp\\ipykernel_21948\\534188681.py:15: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show()\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import precision_recall_curve, average_precision_score\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "\n",
    "for name, model in models.items():\n",
    "    y_prob = model.predict_proba(X_test)[:, 1]\n",
    "    precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=1)\n",
    "    pr_auc = average_precision_score(y_test, y_prob)\n",
    "    plt.plot(recall, precision, label=f'{name} (AP = {pr_auc:.2f})')\n",
    "\n",
    "plt.xlabel(\"Recall\")\n",
    "plt.ylabel(\"Precision\")\n",
    "plt.title(\"Precision-Recall Curve\")\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4522f418-0f71-4101-a760-c6ba838fbb04",
   "metadata": {},
   "source": [
    "## **6. Select the Best Model for Deployment**  \n",
    "Choose the best model based on **ROC-AUC and PR-AUC scores**. Assume **oversampling model** performed best.\n",
    "\n",
    "### **6.1 Save the Model**  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3d38ad4a-3dd5-4248-b354-53486f970d26",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['best_model.pkl']"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "# Save the best model\n",
    "joblib.dump(over_model_lr, \"best_model.pkl\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a38edbc7-40b5-41a9-9fb9-29184043fb32",
   "metadata": {},
   "source": [
    "## **7. Deploy Model using Streamlit**  \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "cde8e8d4-c32c-4d18-8b83-665d4b7ad247",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load model\n",
    "model = joblib.load(\"best_model.pkl\")\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"My first ML App (Study on Imbalanced Data Classification by 67xxxx \")\n",
    "\n",
    "# Input fields\n",
    "features = []\n",
    "for i in range(11):  # Adjust based on dataset\n",
    "    value = st.number_input(f\"Feature_{i}\", value=0.0)\n",
    "    features.append(value)\n",
    "\n",
    "# Prediction\n",
    "if st.button(\"Predict\"):\n",
    "    prediction = model.predict([np.array(features)])\n",
    "    st.write(f\"Predicted Class: {prediction[0]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d8241e0a-affb-41dd-bae3-ad2eec5e0135",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (3737097518.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[19], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    streamlit run app.py\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "streamlit run app.py"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84056e61-bef6-43fe-ad36-10064d5d8faa",
   "metadata": {},
   "source": [
    "**Insert link of your App here.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abdedb94-129c-4e32-af3c-6645acbb801f",
   "metadata": {},
   "outputs": [],
   "source": [
    "https://github.com/natapatnn/SED616-ASS5-FirstML.git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "81ad90aa-5446-48c2-9cf7-d195fcee1a34",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
