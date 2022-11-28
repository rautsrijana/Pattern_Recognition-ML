# Core Package
import streamlit as st 

# Packages for EDA 
import pandas as pd 
import numpy as np 

# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Packages for Data Visualization
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

# DB
from database import *

# For Password 
def generate_hashes(password):
	"""
	Create the password in hased format
	"""
	return hashlib.sha256(str.encode(password)).hexdigest()

# Verify password
def verify_hashes(password,hashed_text):
	"""
	Verify the password saved in hashed format
	"""
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False

# Our 14 best features
feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']

# Based on our dataset
gender_dict = {"male":1,"female":2}
feature_dict = {"No":1,"Yes":2}

def get_value(val,my_dict):
	"""
	To get the dict value 
	"""
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	"""
	To get the dict key
	"""
	for key,value in my_dict.items():
		if val == key:
			return key

def get_fvalue(val):
	"""
	get feature values
	"""
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value 

# Load ML Models
def load_model(model_file):
	"""
	to load our model according to our choice
	"""
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# ML Interpretation
import lime
import lime.lime_tabular

def main():
	"""Application For Mortality Prediction """
	st.title("Hepatitis Mortality Prediction")
	
	menu = ["Home","Login","Signup","About"]
	submenu = ["Plot","Prediction"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.write("Hepatitis means inflammation of the liver.---Need to write more about hepatitis")

	elif choice == "Login":
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			#from database
			create_usertable()
			hashed_pswd = generate_hashes(password)
			result = login_user(username,verify_hashes(password,hashed_pswd))

			if result:
				st.success("Welcome {}".format(username))
				
				activity = st.selectbox("Activity", submenu)
				if activity == "Plot":
					st.subheader("Data Visualization Plot")
					df = pd.read_csv("data/clean_hepatitis_dataset.csv")
					st.dataframe(df)
					st.bar_chart(df["class"].value_counts())
					# Freq Dist Plot
					freq_df = pd.read_csv("data/freq_df_hepatitis_dataset.csv")
					st.bar_chart(freq_df['count'])

					if st.checkbox("Area Chart"):
						all_columns = df.columns.to_list()
						feat_choices = st.multiselect("Choose a Feature",all_columns)
						new_df = df[feat_choices]
						st.area_chart(new_df)    # this is coming from sqlite
				

				elif activity == "Prediction":
					st.subheader("Predictive Analysis")

					# Setting the Rules According to our 14 Best features
					age = st.number_input("Age",7,78) # min age is 7 and max is  78 in our dataset
					sex = st.radio("Sex",tuple(gender_dict.keys()))
					# For numerical data
					steroid = st.radio("Do You Take Steroids?",tuple(feature_dict.keys()))
					antivirals = st.radio("Do You Take Antivirals?",tuple(feature_dict.keys()))
					fatigue = st.radio("Do You Have Fatigue",tuple(feature_dict.keys()))
					spiders = st.radio("Presence of Spider Naeve",tuple(feature_dict.keys()))
					ascites = st.selectbox("Ascities",tuple(feature_dict.keys()))
					varices = st.selectbox("Presence of Varices",tuple(feature_dict.keys()))
					bilirubin = st.number_input("bilirubin Content",0.0,8.0)
					alk_phosphate = st.number_input("Alkaline Phosphate Content",0.0,296.0)
					sgot = st.number_input("Sgot",0.0,648.0)
					albumin = st.number_input("Albumin",0.0,6.4)
					protime = st.number_input("Prothrombin Time",0.0,100.0)
					histology = st.selectbox("Histology",tuple(feature_dict.keys()))
					feature_list = [age,get_value(sex,gender_dict),get_fvalue(steroid),get_fvalue(antivirals),get_fvalue(fatigue),get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology)]
					st.write(len(feature_list))
					st.write(feature_list)

					pretty_result = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
					st.json(pretty_result)
					single_sample = np.array(feature_list).reshape(1,-1)

					# Machine LearninG Part
					model_choice = st.selectbox("Select Model",["Logistic Regression","KNN","DecisionTree"])
					if st.button("Predict"):
						if model_choice == "KNN":
							loaded_model = load_model("models/knn_hepB_model_nov27_2022.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						elif model_choice == "DecisionTree":
							loaded_model = load_model("models/decision_tree_clf_hepB_model_nov27_2022.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						else:
							loaded_model = load_model("models/logistic_regression_hepB_model_nov27_2022.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						st.write(prediction)

						if prediction == 1:
							st.warning("Patient Dies")
							pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
							st.subheader("The Prediction Probability Score using {}".format(model_choice)) #passing the model choice
							st.json(pred_probability_score)
							
							
						else:
							st.success("Patient Lives")
							pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)

			else:
				st.warning("Incorrect Username or Password")
	
	elif choice == "Signup":
		new_username = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		confirm_password = st.text_input("Confirm Password",type='password')
		if new_password == confirm_password:
			st.success("Password Confirmed !!")
		else:
			st.warning("Passwords donot Matched!!")
		

		if st.button("Submit"):
			# function is from database.py file
			create_usertable()
			hashed_new_password = generate_hashes(new_password)  # above function 
			add_userdata(new_username,hashed_new_password)
			st.success("Thank You!! You have successfully created a new account")
			st.info("Please Login to Get Started")

	elif choice == "About":
		st.write("This is our Graduate Project of Pattern Recognition System and Machine Learning(CSC-588). Our model classify whether the person live or die based upon the given parameters from the dataset.We use three Machine Learning Algorithms to make our model i.e are Logistic Regression, KNN and Decision Tree.Joblib was used for storing and using the trained model in the website. Sqlite3 is the database that is used to store encrypted form of password.Streamlit is mainly used for the front-end development for our Website.")
		st.caption('Submitted To: Dr. VijayaLaxmi Saravanan')
		st.caption('Created by: Shreedhar Dahal(),Srijana Raut(101134199)')
	


if __name__ == '__main__':
	main()

