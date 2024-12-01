
# Smart Irrigation System 🌱  

## Problem Statement  
Water scarcity is a pressing issue in agriculture, especially in arid and semi-arid regions. Traditional irrigation systems often lead to excessive water usage due to inefficiencies and lack of data-driven approaches. Our project aims to optimize water usage by predicting soil moisture levels and determining precise water requirements for crops, reducing water wastage and enhancing agricultural sustainability.  

---

## Project Overview  
The **Smart Irrigation System** is a machine learning-driven solution designed to optimize irrigation practices by predicting soil moisture levels. By leveraging predictive models, real-time data from sensors, and an intuitive user interface, the system helps farmers make informed decisions about water usage.  

Key features of the project include:  
- **Data Analysis:** Visualizing and understanding historical and real-time sensor data.  
- **ML Models:** Comparing the performance of **Random Forest** and **Gradient Boosting** algorithms to predict soil moisture.  
- **Streamlit Dashboard:** A user-friendly interface for data visualization, model comparison, and prediction outcomes.  
- **Cloud Integration:** Storing and analyzing sensor data for better decision-making.  

---

## Project Working  

1. **Data Collection**  
   - Sensor data (e.g., soil moisture, temperature, and humidity) is transmitted to the cloud using NodeMCU.  
2. **Machine Learning Models**  
   - **Random Forest** and **Gradient Boosting** algorithms are trained and evaluated on the same dataset.  
   - The best-performing model is used for predicting water requirements.  
3. **Streamlit Dashboard**  
   - Displays sensor data analysis.  
   - Allows users to input parameters and view predictions.  
4. **Irrigation Recommendations**  
   - Provides precise water usage recommendations to minimize wastage.  
## Tech Stack  

### Backend  
- **Machine Learning:** Python (Gradient Boosting, Random Forest using scikit-learn)  
- **Database:** Firebase for storing sensor and historical data  
- **APIs:** RESTful APIs for sensor data retrieval and prediction  

### Frontend  
- **Streamlit** for interactive dashboards  

### Hardware  
- **NodeMCU** for sensor data collection  

---

## Installation Guide  

### Prerequisites  
- Python 3.7+  
- Firebase account (for database integration)  
- NodeMCU (configured with sensors)  

### Steps to Run  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/your-repo-url  
   cd Smart-Irrigation-System  
   ```  

2. **Install Required Libraries**  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Set Up Firebase**  
   - Configure Firebase and update `firebase_config.json` with your credentials.  

4. **Run the Backend**  
   ```bash  
   python backend_api.py  
   ```  

5. **Launch the Streamlit Dashboard**  
   ```bash  
   streamlit run app.py  
   ```  

6. **Connect Hardware**  
   - Deploy the NodeMCU script for sensor data transmission.  

## License  
[MIT License](LICENSE)  
