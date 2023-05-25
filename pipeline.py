import pandas as pd
import os, time, joblib, warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as acc, \
                            precision_score as prec, \
                            recall_score as rec, \
                            f1_score as f1, \
                            confusion_matrix as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
warnings.filterwarnings('ignore')

class Pipeline:
  def __init__(self, rs=None, compare_scaling_effect=True, show=False):
    self.start_ts = datetime.now()
    self.time_metadata = {}
    self.random_state = rs
    self.show = show
    self.compare_scaling_effect = compare_scaling_effect

  def run_pipeline(self):
    """
    Runs a machine learning pipeline from getting data until generating 
    the validation report.
    """
    print(f'{" STARTING PIPELINE ":=^42s}')
    self.get_data()
    self.prepare_data()
    self.build_model()
    self.test_model()
    self.export_file()
    self.generate_report()
    print(f'{" PIPELINE FINISH ":=^42s}')

  def get_data(self):
    """
    Gets the data from a remote server (the UCI Machine Learning Repository)
    and loads it into a pandas DataFrame.

    Raises:
        ConnectionError: If the server cannot be reached.
        ValueError: If the data cannot be loaded into a DataFrame.
    """
    print(f'{"Getting Data ":.<28s}', end=' ')
    start = time.time()
    
    try:
      self.col_names = ['sepal_length', 'sepal_width', 'petal_length', 
                        'petal_width', 'class']
      self.data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                              names=self.col_names)
      end = time.time()
      self.time_metadata['Get Data Time'] = f'{round((end-start),2)} s'
      print(f'done ({self.time_metadata["Get Data Time"]})')
    except ConnectionError:
      print('Error: Could not connect to server.')
    except ValueError:
      print('Error: Could not load data into DataFrame.')
    
    if self.show: print(self.data.describe())

  def prepare_data(self):
    """
    Preprocesses the data by encoding the class labels, splitting the data 
    into training and validation sets, and scaling the training data.
    """
    print(f'{"Preparing Data ":.<28s}', end=' ')
    start = time.time()
    self._encode_labels()
    self._split_data()
    self.x_train = self._feature_engineering(data=self.x_train)
    self._scale_data()
    end = time.time()
    self.time_metadata['Preprocessing Time'] = f'{round((end-start),2)} s'
    print(f'done ({self.time_metadata["Preprocessing Time"]})')

  def _encode_labels(self):
    """
    Encodes the class labels in the DataFrame.
    """
    self.label_encoder = LabelEncoder()
    self.label_encoder.fit(self.data['class'])
    self.data['class_label'] = self.label_encoder.transform(self.data['class'])

  def _split_data(self):
    """
    Splits the data into training and validation sets.
    """
    X = self.data[self.col_names[:-1]] 
    y = self.data['class_label'].copy() 
    self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                                                          X, y, test_size=.2, stratify=y, 
                                                          random_state=self.random_state
                                                         )

    # get index of splitted data
    self.train_index = self.x_train.index.tolist()
    self.val_index = self.x_val.index.tolist()

  def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the given dataset.

    Returns:
    - (pd.DataFrame): data with feature engineering only.
    """
    self.fe_source_columns = {
      'sepal_area': ['sepal_width', 'sepal_length'],
      'petal_area': ['petal_width', 'petal_length']
    }
    self.fe_columns = ['sepal_area', 'petal_area']
    
    # generate new features
    data['sepal_area'] = data[self.fe_source_columns['sepal_area']].prod(axis=1) 
    data['petal_area'] = data[self.fe_source_columns['petal_area']].prod(axis=1)
    
    # take only new feature for training
    data = data[self.fe_columns]

    return data
    
  def _scale_data(self):
    """
    Scales the training data using sklearn.preprocessing.StandardScaler.
    """
    if self.compare_scaling_effect:
      self._visualize_data('Before Scaling')

    self.scaler = StandardScaler()
    self.scaler.fit(self.x_train)
    self.x_train = self.scaler.transform(self.x_train)
    
    if self.compare_scaling_effect:
      self._visualize_data('After Scaling')

  def _visualize_data(self, title: str):
    """
    Function to plot features for scaling effect comparison purpose.
    """
    plt.style.use('ggplot')
    plt.figure(figsize=(5,5))
    if isinstance(self.x_train, pd.DataFrame):
      plt.scatter(self.x_train['sepal_area'], self.x_train['petal_area'], c='orange')
    else:
      plt.scatter(self.x_train[:,0], self.x_train[:,1], c='orange')
    plt.axhline(0, linestyle='--', c='k')
    plt.axvline(0, linestyle='--', c='k')
    plt.title(f"{title}")
    plt.xlabel('sepal_area')
    plt.ylabel('petal_area')
    plt.savefig(f'result/{title}.jpg', dpi=200)
    plt.close()

  def build_model(self):
    """
    build machine learning model (with 3 models variation)
    """    
    print(f'{"Building Model ":.<28s}', end=' ')
    start = time.time()
    
    model_lr = LogisticRegression(random_state=self.random_state)
    model_lr.fit(self.x_train, self.y_train)
    self.lr = model_lr
    
    model_dtc = DecisionTreeClassifier(random_state=self.random_state)
    model_dtc.fit(self.x_train, self.y_train)
    self.dtc = model_dtc

    model_svc = SVC(random_state=self.random_state, probability=True)
    model_svc.fit(self.x_train, self.y_train)
    self.svc = model_svc

    end = time.time()
    self.time_metadata['Build Model Time'] = f'{round((end-start),2)} s'
    print(f'done ({self.time_metadata["Build Model Time"]})')

  def test_model(self):
    """
    validate model with unseen data
    """    
    print(f'{"Testing Model ":.<28s}', end=' ')
    start = time.time()
    
    # prepare validation data
    self.x_val = self._feature_engineering(data=self.x_val)
    self.x_val = self.scaler.transform(self.x_val) 
    
    # make predictions and calculate metrics
    results = {} 
    models = [self.lr, self.dtc, self.svc]
    for model in models:
      y_pred = model.predict(self.x_val)
      results[type(model).__name__] = {
        'accuracy': acc(y_true=self.y_val, y_pred=y_pred),
        'precision': prec(y_true=self.y_val, y_pred=y_pred, average='macro'),
        'recall': rec(y_true=self.y_val, y_pred=y_pred, average='macro'),
        'f1-score': f1(y_true=self.y_val, y_pred=y_pred, average='macro'),
        'confusion_matrix': cm(y_true=self.y_val, y_pred=y_pred)
      }

    # save result
    df_results = pd.DataFrame().from_dict(results, orient='index')
    df_results.reset_index(inplace=True)
    df_results.rename(columns={'index': 'model_name'}, inplace=True)
    self.val_result = df_results

    end = time.time()
    self.time_metadata['Test Model Time'] = f'{round((end-start),2)} s'
    print(f'done ({self.time_metadata["Test Model Time"]})')

  def export_file(self):
    """
    Export important file for production
    """
    print(f'{"Exporting Pipeline File ":.<28s}', end=' ')
    start = time.time()
    
    self.pipeline_object = {
      'scaler': self.scaler,
      'label_encoder': self.label_encoder,
      'model_lr': self.lr,
      'model_dtc': self.dtc,
      'model_svc': self.svc,
      'trained_data': self.x_train,
      'trained_label': self.data.loc[self.train_index]['class'].tolist(),
      'feature_engineering': self.fe_source_columns
    }

    joblib.dump(self.pipeline_object, './pipeline.bin' )
    
    end = time.time()
    self.time_metadata['Export File Time'] = f'{round((end-start),2)} s'
    print(f'done ({self.time_metadata["Export File Time"]})')
    
  def generate_report(self):
    """
    Generates a report of the validation results, including metrics comparison 
    & confusion matrices. The report is saved as an Excel file and a PNG image.
    """
    print(f'{"Generating Report ":.<28s}', end=' ')
    start = time.time()
    
    # check &/ create report directory
    report_dir = 'result'
    if not os.path.exists(report_dir): 
      os.makedirs(report_dir)
    
    # plot setting
    plt.style.use('seaborn')
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 18.5
    plt.rcParams['ytick.labelsize'] = 18.5
    plt.rcParams['legend.fontsize'] = 16

    # plotting    
    fig = plt.figure(figsize=(16,20), layout='constrained')
    gs = GridSpec(5,3, figure=fig, height_ratios=[.8,.8, 1, .75, 1])

    # compare trained vs validation data input
    ax00 = fig.add_subplot(gs[0,:2])
    ax10 = fig.add_subplot(gs[1,:2])
    ax0_hist = [ax00, ax10]
    for i, ax in enumerate(ax0_hist):
      sns.histplot(data=self.x_train[:,i], label='Train', stat='percent', 
                   element='step', color='red', ax=ax, legend=True) 
      sns.histplot(data=self.x_val[:,i], label='Validation', stat='percent', 
                   element='step', color='blue', ax=ax, legend=True)
      ax.set_title(self.fe_columns[i], x=.82)
      ax.set_xlabel('cm$^2$')
      ax.legend(loc='best')#, frameon=True, shadow=True)
    df_comparison = pd.DataFrame({
      self.fe_columns[0]: self.x_train[:,0].tolist() + self.x_val[:,0].tolist(), 
      self.fe_columns[1]: self.x_train[:,1].tolist() + self.x_val[:,1].tolist(), 
      'label': ['Train']*self.x_train.shape[0] + ['Validation']*self.x_val.shape[0]
    })
    ax0_box1 = fig.add_subplot(gs[0,2])
    sns.boxplot(data=df_comparison, x=df_comparison.columns[0], y='label', 
                ax=ax0_box1, palette='Set1') 
    ax0_box1.set_ylabel('')
    ax0_box1.set_xlabel('cm$^2$')
    ax0_box2 = fig.add_subplot(gs[1,2])
    sns.boxplot(data=df_comparison, x=df_comparison.columns[1], y='label', 
                ax=ax0_box2, palette='Set1')
    ax0_box2.set_ylabel('')
    ax0_box2.set_xlabel('cm$^2$')
    ax0_title = fig.add_subplot(gs[0, :3])
    ax0_title.set_title('Input Comparison', fontsize=26, fontweight='bold', y=1.1)
    ax0_title.axis('off')

    # metrics
    metrics = self.val_result[self.val_result.columns[:-1]]
    metrics[metrics.columns[1:]] = metrics[metrics.columns[1:]] * 100
    ax2 = fig.add_subplot(gs[2:4,:])
    metrics.plot(kind='bar', ylim=(0,100), ax=ax2)
    ax2.set_xticklabels(metrics.model_name, rotation=0, fontstyle='italic', 
                        fontweight='bold')
    ax2.set_title('Metrics Comparison', fontweight='bold')
    ax2.set_ylabel('Score (%)', fontweight='bold')
    ax2.legend(shadow=True, frameon=True)

    # confusion matrix
    ax40 = fig.add_subplot(gs[4:,0])
    ax41 = fig.add_subplot(gs[4:,1])
    ax42 = fig.add_subplot(gs[4:,2])
    ax4 = [ax40, ax41, ax42]
    for i, ax in enumerate(ax4):
      ax.set_title(metrics.model_name.loc[i], fontstyle='italic')
      sns.heatmap(self.val_result.confusion_matrix.loc[i], annot=True,
                  ax=ax, fmt='d', cbar=False, cmap='Greens', center=0, 
                  yticklabels=True, xticklabels=True, annot_kws={'size':20}
                  )
      label_names = self.label_encoder.inverse_transform(list(range(3)))
      label_names = [lbl.replace('-','\n') for lbl in label_names]
      ax.set_xticklabels(label_names, rotation=0) 
      ax.set_yticklabels(label_names, rotation=0) 
      ax.set_xlabel('Prediction')
      ax.set_ylabel('Actual')
    ax4_title = fig.add_subplot(gs[4, :])
    ax4_title.set_title('Confusion Matrix', fontsize=26, fontweight='bold', 
                        y=1.12)
    ax4_title.axis('off')

    # gs.update(hspace=.0375, wspace=.01)#, top=.2)#, wspace=.025)
    fig.savefig(f'{report_dir}/{str(datetime.now().date()).replace("-","_")}_metrics_result.png', 
                bbox_inches='tight', dpi=150)

    end = time.time()
    self.time_metadata['Genereate File Time'] = f'{round((end-start),2)} s'
    self.end_ts = datetime.now()

    self.time_metadata['Start Pipeline at'] = self.start_ts
    self.time_metadata['Finish Pipeline at'] = self.end_ts
    time_df = pd.DataFrame().from_dict(self.time_metadata, orient='index')
    

    # extract pipeline object info
    obj_info = {key:type(value).__name__ for key, value in self.pipeline_object.items()}
    df_obj_info = pd.DataFrame().from_dict(obj_info, orient='index', columns=['Object Type'])
    df_obj_info.reset_index(inplace=True)
    df_obj_info.rename(columns={'index': 'Pipeline_Object_Name'}, inplace=True)
    
    # extract trained data
    trained_data = pd.DataFrame(self.x_train, columns=self.fe_columns)
    
    # exporting validation result file and object info
    with pd.ExcelWriter(f'{report_dir}/{str(datetime.now().date()).replace("-","_")}_pipeline_result.xlsx') as writer:
      df_obj_info.to_excel(writer, sheet_name='pipeline_info')
      trained_data.describe().to_excel(writer, sheet_name='trained_data_statistics')
      self.val_result.to_excel(writer, sheet_name='validation_metric_result', index=False)
      time_df.to_excel(writer, sheet_name='time_metadata')

    print(f'done ({self.time_metadata["Genereate File Time"]})')
  
if __name__ == '__main__':
  ml_pipeline = Pipeline(949672)
  ml_pipeline.run_pipeline()