import streamlit as st
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set page config
st.set_page_config(
    page_title="Healthcare ChatBot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for colors
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
    }
    .sub-header {
        color: #ff7f0e;
        font-size: 1.5em;
    }
    .success-text {
        color: #2ca02c;
        font-weight: bold;
    }
    .warning-text {
        color: #d62728;
        font-weight: bold;
    }
    .info-text {
        color: #17becf;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border: 2px solid #1f77b4;
    }
    .stNumberInput>div>div>input {
        border: 2px solid #1f77b4;
    }
    .stSelectbox>div>div>select {
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

import os

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
training = pd.read_csv(os.path.join(script_dir, 'Data', 'Training.csv'))
testing = pd.read_csv(os.path.join(script_dir, 'Data', 'Testing.csv'))
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
model = SVC()
model.fit(x_train, y_train)

# Calculate accuracies
dt_accuracy = clf.score(x_test, y_test)
svm_accuracy = model.score(x_test, y_test)

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}
for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}
for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp) + 1) > 13):
        return "You should take the consultation from doctor. "
    else:
        return "It might not be that bad but you should take precautions."

def getDescription():
    global description_list
    with open(os.path.join(script_dir, 'MasterData', 'symptom_Description.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open(os.path.join(script_dir, 'MasterData', 'symptom_severity.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open(os.path.join(script_dir, 'MasterData', 'symptom_precaution.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv(os.path.join(script_dir, 'Data', 'Training.csv'))
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def get_prediction(disease_input):
    tree_ = clf.tree_
    feature_name = [
        cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(cols).split(",")
    symptoms_present = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                return recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                return recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            return present_disease, list(symptoms_given)

    return recurse(0, 1)

# Load dictionaries
getSeverityDict()
getDescription()
getprecautionDict()

# Streamlit app
st.sidebar.title("🏥 Healthcare ChatBot Info")
st.sidebar.markdown("**Model Accuracies:**")
st.sidebar.markdown(f"- Decision Tree: <span class='success-text'>{dt_accuracy:.2%}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"- SVM: <span class='success-text'>{svm_accuracy:.2%}</span>", unsafe_allow_html=True)
st.sidebar.markdown("**Total Diseases:**")
st.sidebar.markdown(f"<span class='info-text'>{len(le.classes_)}</span>", unsafe_allow_html=True)
st.sidebar.markdown("**Total Symptoms:**")
st.sidebar.markdown(f"<span class='info-text'>{len(cols)}</span>", unsafe_allow_html=True)
st.sidebar.markdown("**Current Step:**")
if 'state' not in st.session_state:
    st.session_state.state = 'start'
step_info = {
    'start': '👋 Welcome - Enter Name',
    'symptom_input': '🔍 Symptom Input',
    'symptom_select': '📋 Symptom Selection',
    'days': '📅 Duration Input',
    'questions': '❓ Symptom Confirmation',
    'results': '📊 Diagnosis Results'
}
current_step = step_info.get(st.session_state.state, 'Unknown')
st.sidebar.markdown(f"<span class='info-text'>{current_step}</span>", unsafe_allow_html=True)

st.markdown('<div class="main-header">Healthcare ChatBot</div>', unsafe_allow_html=True)

if 'state' not in st.session_state:
    st.session_state.state = 'start'

if st.session_state.state == 'start':
    st.markdown('<div class="sub-header">Welcome to the Healthcare ChatBot</div>', unsafe_allow_html=True)
    name = st.text_input("Your Name?", key="name_input")
    if st.button("Start Chat"):
        if name:
            st.session_state.name = name
            st.session_state.state = 'symptom_input'
            st.rerun()
        else:
            st.error("Please enter your name.")

elif st.session_state.state == 'symptom_input':
    st.write(f"Hello, {st.session_state.name}")
    symptom = st.text_input("Enter the symptom you are experiencing", key="symptom_input")
    chk_dis = ",".join(cols).split(",")
    if st.button("Submit Symptom"):
        conf, cnf_dis = check_pattern(chk_dis, symptom)
        if conf == 1:
            st.session_state.cnf_dis = cnf_dis
            st.session_state.state = 'symptom_select'
            st.rerun()
        else:
            st.error("Enter a valid symptom.")

elif st.session_state.state == 'symptom_select':
    cnf_dis = st.session_state.cnf_dis
    if len(cnf_dis) > 1:
        selected = st.selectbox("Select the one you meant", cnf_dis, key="select_symptom")
        if st.button("Confirm Selection"):
            st.session_state.disease_input = selected
            st.session_state.state = 'days'
            st.rerun()
    else:
        st.session_state.disease_input = cnf_dis[0]
        st.session_state.state = 'days'
        st.rerun()

elif st.session_state.state == 'days':
    days = st.number_input("From how many days?", min_value=1, step=1, key="days_input")
    if st.button("Submit Days"):
        st.session_state.num_days = days
        present_disease, symptoms_given = get_prediction(st.session_state.disease_input)
        st.session_state.present_disease = present_disease
        st.session_state.symptoms_given = symptoms_given
        st.session_state.symptoms_exp = []
        st.session_state.question_index = 0
        st.session_state.state = 'questions'
        st.rerun()

elif st.session_state.state == 'questions':
    symptoms_given = st.session_state.symptoms_given
    question_index = st.session_state.question_index
    progress = (question_index) / len(symptoms_given)
    st.progress(progress)
    st.write(f"Question {question_index + 1} of {len(symptoms_given)}")
    if question_index < len(symptoms_given):
        sym = symptoms_given[question_index]
        st.write(f"Are you experiencing {sym}?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key=f"yes_{question_index}"):
                st.session_state.symptoms_exp.append(sym)
                st.session_state.question_index += 1
                st.rerun()
        with col2:
            if st.button("No", key=f"no_{question_index}"):
                st.session_state.question_index += 1
                st.rerun()
    else:
        # Done with questions
        st.session_state.state = 'results'
        st.rerun()

elif st.session_state.state == 'results':
    symptoms_exp = st.session_state.symptoms_exp
    present_disease = st.session_state.present_disease
    num_days = st.session_state.num_days

    second_prediction = sec_predict(symptoms_exp)
    condition = calc_condition(symptoms_exp, num_days)

    st.markdown(f"<span class='warning-text'>{condition}</span>", unsafe_allow_html=True)

    if present_disease[0] == second_prediction[0]:
        st.markdown(f"<span class='success-text'>You may have {present_disease[0]}</span>", unsafe_allow_html=True)
        st.write(description_list[present_disease[0]])
    else:
        st.markdown(f"<span class='warning-text'>You may have {present_disease[0]} or {second_prediction[0]}</span>", unsafe_allow_html=True)
        st.write(description_list[present_disease[0]])
        st.write(description_list[second_prediction[0]])

    precution_list = precautionDictionary[present_disease[0]]
    st.markdown('<div class="sub-header">Take following measures:</div>', unsafe_allow_html=True)
    for i, j in enumerate(precution_list):
        st.write(f"{i+1}) {j}")

    if st.button("Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()