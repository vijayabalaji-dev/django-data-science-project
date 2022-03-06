from django.shortcuts import render
import pandas as pd
import numpy as np 
import redis as rd
import json
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import seaborn as sns
import urllib
import io
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report


# Create your views here.
def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer,format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def index(request):

    """
          connecting to redis with strict to get rid of Bstrings(B prefixed with words)
    """
    r = rd.StrictRedis(host="localhost", port=6379, charset="utf-8", decode_responses=True)
    data = r.get('foo')
    mylist = r.lrange('cids',0,-1)
    k = list()
    for i in mylist:
        d = r.hgetall(i)
        k.append(d)
    """
         creating data frame from list of dictionaries
    """
    dataf = pd.DataFrame.from_dict(k, orient='columns')
    """
         sending the head to html
    """
    head = dataf.head()
    json_records = head.reset_index().to_json(orient='records')
    tabled = json.loads(json_records)
    des = dataf.describe().to_html(classes='table table-striped')

    #print(r.hgetall(mylist[0]))

    """
         Data pre-processing converting string to numbers 
    """

    lableTrain = dataf.copy()
    le = LabelEncoder()
    cols = dataf.select_dtypes(include=object).columns
    for j in cols:
        lableTrain[j] = le.fit_transform(lableTrain[j])


    #necessary to switch backend to show data in html
    """
        creating first chart in django 

        #pie chart of churn values count
    """
    plt.switch_backend('AGG')
    plt.title("Pie Chart")
    lableTrain['Churn'].value_counts().plot(kind='pie',legend=True,explode=[0,0.09],autopct="%3.1f%%",shadow=True)
    graph = get_graph()
    plt.clf()

    """
         #Bar chart of phone service and total charges based on churn    
    """
    sns.barplot(lableTrain['PhoneService'],lableTrain['TotalCharges'],hue=lableTrain['Churn'])
    gh = get_graph()
    plt.clf()

    """
         Bar chart of payment method and churn   
    """
    plt.figure(figsize=(10,8))
    plt.title("Bar Graph representaion of payment method and count")
    sns.countplot(dataf['PaymentMethod'],hue=lableTrain['Churn'])
    payment_method = get_graph()  
    plt.clf()
    """ pie chart of payment method and churn is yes"""

    plt.figure(figsize=(8,5))
    dataf[dataf['Churn'] == 'Yes']['PaymentMethod'].value_counts().plot(kind='pie', explode=[0.09,0,0,0], legend=True)
    churn_yes_payment_method = get_graph()
    plt.clf()

    """ pie chart of payment method and churn is no"""

    plt.figure(figsize=(8,5))
    dataf[dataf['Churn'] == 'No']['PaymentMethod'].value_counts().plot(kind='pie', explode=[0.09,0,0,0], legend=True)
    churn_no_payment_method = get_graph()
    plt.clf()
    
    """ data for modeling """
    
    data_modeling = lableTrain[['tenure','PhoneService','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','Churn']]
    
    #sending to html

    data_modeling_html = data_modeling.head().to_html(classes='table table-striped')



    x = data_modeling[['tenure','PhoneService','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges']]
    y = data_modeling['Churn']

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
   
    log_model = LogisticRegression()

    log_score = cross_val_score(estimator=log_model,X=x_train,y=y_train,cv=5,scoring='roc_auc')

    print(str(log_score)+ " mean : "+ str(log_score.mean()))

    m_log_score = log_score.mean()

    #Decesion Tree

    decision_tree = DecisionTreeClassifier()
    
    des_score = cross_val_score(estimator=decision_tree,X=x_train,y=y_train,cv=5,scoring='roc_auc')

    print(str(des_score)+ " mean : "+ str(des_score.mean()))

    m_des_score = des_score.mean()

    #Random Forest
    
    random_forest = RandomForestClassifier(criterion='gini')

    ran_score = cross_val_score(estimator=random_forest,X=x_train,y=y_train,cv=5,scoring='roc_auc')
    
    print(str(ran_score)+ " mean : "+ str(ran_score.mean()))
    
    m_ran_score = ran_score.mean()
    
    """
          Random Forest gives us best Result to we will be building model using random forest
    """
    
    #feature importance 
     
    random_forest.fit(x_train,y_train)

    for score , name in sorted(zip(random_forest.feature_importances_,x_train.columns),reverse=True):
        print(str(name)+ "  "+ str(score*100)+ " %")

    y_train_pred = random_forest.predict(x_train)

    y_pred = random_forest.predict(x_test)

    """
          Model Evaluation

    """

    confusion = pd.DataFrame(confusion_matrix(y_test,y_pred))

    confusion.index = ['Actual Negative','Actual Positive']

    confusion.columns = ['Predicted Negative','Predicted Positive']

    confusion_to_html = confusion.to_html(classes='table table-striped')
    
    print(confusion_to_html)

    # Accuracy Score
    accuracy = accuracy_score(y_test,y_pred)
    print("Accuracy Score "+str(accuracy_score(y_test,y_pred)))

    #precision score
    precision = precision_score(y_test,y_pred)
    print("Precesion Score "+str(precision_score(y_test,y_pred)))

    #Recall Score
    recall = recall_score(y_test,y_pred)
    print("Recall Score "+str(recall_score(y_test,y_pred)))
    
    result = 0
    
    if request.method == "POST":
        tenure =  request.POST['tenure']
        phoneservice = request.POST['phoneservice']
        contract = request.POST['contract']
        paperlessbilling = request.POST['paperlessbilling']
        paymentmethod = request.POST['paymentmethod']
        monthlycharges = request.POST['monthlycharges']

        datalist = {'tenure' : [tenure],'PhoneService' : [phoneservice],'Contract' : [contract],'PaperlessBilling' : [paperlessbilling],'PaymentMethod' : [paymentmethod],'MonthlyCharges' : [monthlycharges]}

        df = pd.DataFrame(datalist)

        print(df)
        
        converter = LabelEncoder()
        for j in df.columns:
            df[j] = converter.fit_transform(df[j])
        
        res = random_forest.predict(df)
        result = ''

        if res[0] == 1:
            result = "oops ! he will leave"
        else: 
            result = "hey ! he will stay with you"



    ######  rendering to html ######## 
    return render(request,'index.html',{
        "text" : data,
        "t" : tabled,
        "des"  : des,
        "grf" : graph,
        "gf":gh,
        "payment" : payment_method,
        "churn_yes_payment_method" : churn_yes_payment_method,
        "churn_no_payment_method" : churn_no_payment_method,
        "data_modeling_html" : data_modeling_html, 
        "log_score" : m_log_score,
        "des_score" : m_des_score,
        "ran_score" : m_ran_score,
        "confusion" : confusion_to_html,
        "accuracy" : accuracy,
        "recall" : recall,
        "precision" : precision,
        "result":  result
         })