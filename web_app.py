import streamlit as st
import pandas as pd
import os, json, joblib, requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')


def authenticate(username:str, password:str) -> str:
  """
  authenticating username and password
  """
  load_dotenv()
  AUTH_URL = os.getenv('AUTH_URL')
  VALIDATE_TOKEN = os.getenv('VALIDATE_TOKEN')

  # send a request to the authentication service
  response = requests.post(AUTH_URL, json={'username':username, 'password':password})

  if response.status_code==200:
    # get acccess token
    token = response.json().get("access_token")

    # valdiate access token
    response_validation = requests.get(f"{VALIDATE_TOKEN}{token}")
    token_is_valid = response_validation.json().get("token_valid")
    if token_is_valid:
      return token
  else:
    st.error('Invalid username or password')
    return None

def visual_projection(df_train:pd.DataFrame, x: float, y: float) -> None:
  """
  Visualize trained data in feature engineering projections with user input data
  point to support prediction result's confidence.

  Args:
      df_train (pd.DataFrame): trained data
      x (float): sepal_width-coordinate of the user input data.
      y (float): petal_width-coordinate of the user input data.

  Returns:
      None
  """

  # Existing data
  fig = px.scatter(
      df_train,
      x=df_train.columns[0],
      y=df_train.columns[1],
      color=df_train.columns[2],
      title='Visual Projection'
  )
  
  # Prediction
  new_point = go.Scatter(
      x=[x],  
      y=[y],  
      mode='markers',
      marker=dict(symbol='x', size=12, color='black'),
      name='User Input Data'  
  )
  fig.add_trace(new_point)
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def login_page():
  """
  Login page
  """
  
  st.set_page_config(page_title='Login Page', layout='wide')
  st.title("Login Page")
  st.write("Please login into your account before you can access web app")

  username = st.text_input("Username")
  password = st.text_input("Password", type="password")

  if st.button("Login"):
    access_token = authenticate(username, password)

    if access_token:
      st.success("Logged in successfully!")
      st.session_state.access_token = access_token
      st.experimental_rerun()

  footer = st.container()
  with footer:
    st.write('<hr>', unsafe_allow_html=True)
    st.write('<h6 style="text-align:center";><i>Author: <a href="https://www.m-nanda.github.io">Muhammad Nanda</a></i></h6>', unsafe_allow_html=True)

def main_page():
  """
  Main page for prediction
  """
  # load credential
  load_dotenv()
  DATA_SOURCE = os.getenv('DATA_SOURCE')
  API_1 = os.getenv('ML_API_1')
  API_2 = os.getenv('ML_API_2')

  # Load existing data from pipeline
  pipeline = joblib.load('pipeline.bin')
  data_train = pipeline['trained_data']
  features = pipeline['feature_engineering']
  label = pipeline['trained_label']
  df_train = pd.DataFrame(data_train, columns=features)
  df_train['label'] = label

  st.set_page_config(page_title='Iris-Classification', layout='wide')

  st.title('Predicting Iris Flower using Machine Learning')
  st.markdown("""<div style="text-align: justify;"> The Iris flower dataset was first introduced in 1936 by the British statistician Ronald Fisher in his paper 
  <a href=https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-1809.1936.tb02137.x>"The Use of Multiple Measurements in Taxonomic Problems."</a> Fisher collected the data as part of his work on linear discriminant analysis, a statistical method for predicting the class of an object based on its measurements. Nowadays, this dataset is commonly used in machine learning, particularly for classification tasks. The dataset contains measurements of the sepal length, sepal width, petal length, and petal width of three species of iris flowers: Iris setosa, Iris versicolor, and Iris virginica. The goal of the classification task is to predict the species of iris flower based on these measurements. The dataset is small, with only 150 instances, making it a good dataset for testing and comparing the performance of different classification algorithms. Additionally, the dataset is well-balanced, with 50 instances of each species, ensuring that each class is equally represented in the data. This web app predicts Iris Flower as a case of machine learning deployment in production.</div>""", unsafe_allow_html=True)
  st.image('https://content.codecademy.com/programs/machine-learning/k-means/iris.svg', 'Iris Flower (image source: kaggle.com/code/necibecan/iris-dataset-eda-n)')

  tab1, tab2 = st.tabs(['Custom', 'Existing (by index)'])

  with tab1:
    st.write('Fill Characteristics')

    # get input data from user
    sepal_length = st.number_input('Sepal Length (cm):', step=0.01, min_value=0.0, format='%.f')
    sepal_width = st.number_input('Sepal Width (cm):', step=0.01, min_value=0.0, format='%.f') 
    petal_length = st.number_input('Petal Length (cm):', step=0.01, min_value=0.0, format='%.f')
    petal_width = st.number_input('Petal Width (cm):', step=0.01, min_value=0.0, format='%.f') 
    
    submitted = st.button('Predict')
    if submitted:
      
      # call API
      access_token = st.session_state.access_token
      headers = {'Auth': f'{access_token}'}
      results = requests.post(f"{API_1}{sepal_length}/{sepal_width}/{petal_length}/{petal_width}", headers=headers)
      
      # show result from API
      results = json.loads(results.text)
      for key, item in results['result'].items():
        st.write(f'{key}:')
        st.info(f'{item}')
      visual_projection(df_train,
                        results['preprocessed_data']['1'],
                        results['preprocessed_data']['2'])

      # clear result
      reset = st.button('Clear Results')	
      if reset:      
        submitted=False
        del results

  with tab2:
    st.write('Choose index')
    idx = st.selectbox('Index data:', (range(150)))

    try:
      col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
      data_tab2 = pd.read_csv(DATA_SOURCE, names=col_names)
      data_tab2 = data_tab2.loc[[int(idx)]]
      input_data_tab2 = data_tab2.drop(columns='class')
    except Exception as e:
      st.exception(e)

    submitted_tab2 = st.button(label='Predict ', key='Predict_by_idx')
    if submitted_tab2:
      st.write('Input Data:')
      st.dataframe(data_tab2)

      # call API
      access_token = st.session_state.access_token
      headers = {'Auth': f'{access_token}'}
      results = requests.post(f"{API_2}{idx}", headers=headers)

      # show result from API
      results = json.loads(results.text)
      for key, item in results['result'].items():
        st.write(f'{key}:')
        st.info(f'{item}')
      visual_projection(df_train,
                        results['preprocessed_data']['1'],
                        results['preprocessed_data']['2'])

      # clear result
      reset_tab2 = st.button('Clear Results')	
      if reset_tab2:      
        submitted_tab2=False
        del results

  footer = st.container()
  with footer:
    st.write('<hr>', unsafe_allow_html=True)
    st.write('<h6 style="text-align:center";><i>Author: <a href="https://www.m-nanda.github.io">Muhammad Nanda</a></i></h6>', unsafe_allow_html=True)
    
# Streamlit app flow
if 'access_token' not in st.session_state:
  login_page()
else:
  main_page()