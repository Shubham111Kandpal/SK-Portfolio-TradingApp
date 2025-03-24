import streamlit as st
from pathlib import Path

def run():
  # --- Profile Picture ---
  profile_pic = Path("Images/Shubham_pic.png")

  # --- Hero Section ---
  col1, col2 = st.columns([1, 3])
  with col1:
      st.image(profile_pic, width=180)
  with col2:
      st.title("Shubham Kandpal")
      st.markdown("**AI & Data Science Professional | MSc Distinction – University of Surrey**")
      st.markdown("📍 London, UK &nbsp;&nbsp;&nbsp;&nbsp; 📧 [shubham.kandpal@gmail.com](mailto:shubham.kandpal@gmail.com) &nbsp;&nbsp;&nbsp;&nbsp; 📞 +44 7407 844770")
      st.markdown("[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shubham-kandpal-035711165)")
      st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github)](https://github.com/Shubham111Kandpal)")

  st.markdown("---")

  # --- Summary ---
  st.header("👨‍💻 About Me")
  st.write("""
  An innovative and results-driven Data Scientist with 4+ years of experience, specializing in AI and machine learning technologies, having a good problem-solving skillset. 
  Currently working at Outlier.ai, focusing on training AI chatbots using Reinforcement Learning with Human Feedback (RLHF). 
  Recently earned an MSc in Data Science with distinction from The University of Surrey, with a dissertation on Adversarial Machine Learning in Facial Recognition Systems. 
  Expertise includes building end-to-end machine learning solutions including skills in NLP, Time Series Analysis, and cloud computing. 
  Strong background in Python, AWS, Statistics, Algorithms, Docker, and CI/CD pipeline development. 
  Actively seeking opportunities to advance in AI, ML, and data science solutions in dynamic environments.
  """)

  # --- Experience ---
  st.header("💼 Work Experience")

  st.subheader("AI Trainer — Outlier.ai (Sep 2024 – Present)")
  st.write("""
  - Training and optimizing AI chatbots using **RLHF** across multiple use-cases (code generation, math, Q&A).
  - Collaborate with cross-functional teams to enhance AI accuracy and interaction quality for Meta, Gemini, OpenAI, and Claude.
  """)

  st.subheader("Stock Trader — Bombay Stock Exchange (Jun 2022 – Jun 2023)")
  st.write("""
  - Analyzed market trends, technical indicators, and financial statements to make data-driven trading decisions across equity and derivative markets in CNC and commodities.
  - Developed algorithmic trading strategies using Python and tested models to optimize trade execution and risk management. Utilized models like Prophet, ARIMA, and custom LSTMs while leveraging CNN for an unconventional market shock analysis.
  """)

  st.subheader("Full-Stack Developer & Data Miner — TCS (Dec 2018 – May 2022)")
  st.write("""
  - Developed 8+ web applications with Python and React.js, improving user experience and workflow for Singapore Airlines. Used Django and Flask for REST API. I also used the SAP UI5 framework for the ERP modules, such as Material Management, EC-PM, Finance, and HR.
  - Implemented Decision Trees to optimize flight delay predictions based on weather and XGBoost models for classifications in passenger demand forecasting. Built ML-based dashboards to visualize key performance metrics, improving operational efficiency and decision-making.
  - Built full-stack solutions using SAP ABAP, OData, and SAP UI5 within an agile framework for Indonesia (Garura) Airlines.
  - Led client interaction, ensuring successful project delivery and high customer satisfaction along with building POCs and KPI dashboards.
  - Earned certifications in ‘SAP ABAP’ and ‘Data Science and Machine Learning’.
  - Built a secure login portal for GXP users, enhancing security and user experience.       
  """)

  # --- Projects ---
  st.header("📌 Key Projects")
  st.markdown("""
  - **[Adversarial Facial Recognition](https://github.com/Shubham111Kandpal/Adversarial-Analysis-of-Facial-Recognition-Model)**  
    Attacks and defenses on ML-based facial recognition systems.
    
  - **[ILP + ML Comparison](https://github.com/Shubham111Kandpal/Machine_Learning_-_Data_Mining)**  
    Aleph ILP vs ML models on churn and synthetic datasets.
    
  - **[Customer Churn Prediction in R](https://github.com/Shubham111Kandpal/Practical_Business_Analytics)**  
    Used XGBoost, NNs, SMOTE, and EDA for churn modeling.
    
  - **[Rental Bikes Web App](https://github.com/Shubham111Kandpal/WebApp_BikeRental)**  
    End-to-end app with MySQL, FastAPI, React.js, Docker, CI/CD.
    
  - **[NLP Sequence Labeling](https://github.com/Shubham111Kandpal/NLP/tree/main)**  
    LSTM, CNN, BERT + embeddings (GloVe, Word2Vec).
    
  - **[Trading App with AWS + GCP](https://github.com/Shubham111Kandpal/Trading_API)**  
    Serverless Monte Carlo simulations, EC2, Lambda, S3.
  """)

  # --- Skills ---
  st.header("🛠 Technical Skills")

  st.markdown("""
  **Languages**: Python, R, SQL, JS, HTML, CSS, SAP ABAP  
  **ML/DS**: Regression, Classification, NLP, LSTM, Transformers, Time Series  
  **Libraries**: TensorFlow, PyTorch, Keras, Scikit-learn, XGBoost, FastAPI, Flask, Spacy  
  **Tools**: Docker, Git, Streamlit, Tableau, Jupyter, VS Code  
  **Cloud & DevOps**: AWS (Lambda, EC2, S3), GCP, GitLab CI/CD  
  **Other**: Agile, CRISP-DM, Jira, SCRUM
  """)

  # --- Education ---
  st.header("🎓 Education")
  st.subheader("MSc Data Science – University of Surrey (2023–2024) [Distinction]")
  st.write("Dissertation: Adversarial Analysis of ML-based Facial Recognition Systems")

  st.subheader("B.Tech Electrical Engineering – BTKIT Dwarahat (2014–2018)")

  # --- Certifications ---
  st.header("📜 Certifications")
  st.markdown("""
  - SAP ABAP (TCS Verified)  
  - Foundations of DS/ML in SAP (SAP Verified)  
  - Python 3 (Udemy Verified)
  """)

  # --- Interests ---
  st.header("🌱 Interests")
  st.markdown("""
  - Artificial Intelligence & Machine Learning  
  - Adversarial Robustness in AI  
  - Data Science & Analytics  
  """)