from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data (same as before)
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)
clf = DecisionTreeClassifier().fit(x_train, y_train)
model = SVC().fit(x_train, y_train)

severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

def getDescription():
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getSeverityDict():
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                severityDictionary[row[0]] = int(row[1])
        except:
            pass

def getprecautionDict():
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

def calc_condition(exp, days):
    if not exp:
        return "No symptoms provided."
    sum_sev = sum(severityDictionary.get(item, 0) for item in exp)
    if (sum_sev * days) / (len(exp) + 1) > 13:
        return "You should take the consultation from doctor."
    else:
        return "It might not be that bad but you should take precautions."

def check_pattern(dis_list, inp):
    inp = inp.replace(' ', '_')
    regexp = re.compile(inp)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return len(pred_list) > 0, pred_list

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier().fit(X_train, y_train)
    symptoms_dict_local = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict_local))
    for item in symptoms_exp:
        if item in symptoms_dict_local:
            input_vector[symptoms_dict_local[item]] = 1
    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def get_prediction(disease_input):
    tree_ = clf.tree_
    feature_name = [cols[i] if i != -2 else "undefined!" for i in tree_.feature]
    symptoms_present = []
    def recurse(node, depth):
        if tree_.feature[node] != -2:
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
            red_cols = training.groupby(training['prognosis']).max().columns
            symptoms_given = red_cols[training.groupby(training['prognosis']).max().loc[present_disease].values[0].nonzero()]
            return present_disease, list(symptoms_given)
    return recurse(0, 1)

getSeverityDict()
getDescription()
getprecautionDict()

class ChatBotApp(App):
    def build(self):
        print("Building app...")
        self.state = 'start'
        self.user_name = ''
        self.disease_input = ''
        self.num_days = 0
        self.symptoms_given = []
        self.symptoms_exp = []
        self.question_index = 0
        self.present_disease = []
        self.chk_dis = [col.replace('_', ' ') for col in cols]

        Window.size = (400, 700)  # Larger height for better display

        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        self.scroll = ScrollView(size_hint=(1, 0.9))
        self.content = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None)
        self.content.bind(minimum_height=self.content.setter('height'))
        self.scroll.add_widget(self.content)
        self.layout.add_widget(self.scroll)

        self.update_ui()
        print("App built successfully")
        return self.layout

    def update_ui(self):
        self.content.clear_widgets()
        if self.state == 'start':
            self.content.add_widget(Label(text='Healthcare ChatBot', font_size=32, color=(0, 0.5, 1, 1), size_hint_y=None, height=50))
            self.content.add_widget(Label(text='Enter your name:', font_size=24, size_hint_y=None, height=30))
            self.name_input = TextInput(multiline=False, font_size=24, size_hint_y=None, height=40)
            self.content.add_widget(self.name_input)
            btn = Button(text='Start Chat', background_color=(0, 0.5, 1, 1), font_size=24, size_hint_y=None, height=50)
            btn.bind(on_press=self.start_chat)
            self.content.add_widget(btn)
        elif self.state == 'symptom_input':
            self.content.add_widget(Label(text=f'Hello, {self.user_name}', font_size=24, size_hint_y=None, height=30))
            self.content.add_widget(Label(text='Enter the symptom you are experiencing:', font_size=20, size_hint_y=None, height=30))
            self.symptom_input = TextInput(multiline=False, font_size=24, size_hint_y=None, height=40)
            self.content.add_widget(self.symptom_input)
            btn = Button(text='Submit Symptom', background_color=(0, 0.5, 1, 1), font_size=24, size_hint_y=None, height=50)
            btn.bind(on_press=self.submit_symptom)
            self.content.add_widget(btn)
        elif self.state == 'symptom_select':
            self.content.add_widget(Label(text='Select the one you meant:', font_size=20, size_hint_y=None, height=30))
            for i, sym in enumerate(self.cnf_dis):
                btn = Button(text=sym, background_color=(0.5, 0.5, 0.5, 1), font_size=20, size_hint_y=None, height=40)
                btn.bind(on_press=lambda instance, s=sym: self.select_symptom(s))
                self.content.add_widget(btn)
        elif self.state == 'days':
            self.content.add_widget(Label(text='From how many days?', font_size=20, size_hint_y=None, height=30))
            self.days_input = TextInput(multiline=False, input_filter='int', font_size=24, size_hint_y=None, height=40)
            self.content.add_widget(self.days_input)
            btn = Button(text='Submit Days', background_color=(0, 0.5, 1, 1), font_size=24, size_hint_y=None, height=50)
            btn.bind(on_press=self.submit_days)
            self.content.add_widget(btn)
        elif self.state == 'questions':
            if self.question_index < len(self.symptoms_given):
                sym = self.symptoms_given[self.question_index]
                self.content.add_widget(Label(text=f'Are you experiencing {sym}?', font_size=20, size_hint_y=None, height=30))
                hbox = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=None, height=50)
                yes_btn = Button(text='Yes', background_color=(0, 1, 0, 1), font_size=20)
                yes_btn.bind(on_press=self.answer_yes)
                no_btn = Button(text='No', background_color=(1, 0, 0, 1), font_size=20)
                no_btn.bind(on_press=self.answer_no)
                hbox.add_widget(yes_btn)
                hbox.add_widget(no_btn)
                self.content.add_widget(hbox)
            else:
                self.show_results()
        elif self.state == 'results':
            self.show_results()

    def start_chat(self, instance):
        self.user_name = self.name_input.text
        if self.user_name:
            self.state = 'symptom_input'
            self.update_ui()

    def submit_symptom(self, instance):
        symptom = self.symptom_input.text
        conf, cnf_dis = check_pattern(self.chk_dis, symptom)
        if conf:
            self.cnf_dis = cnf_dis
            self.state = 'symptom_select'
            self.update_ui()
        else:
            self.content.add_widget(Label(text='Enter a valid symptom.', color=(1, 0, 0, 1)))

    def select_symptom(self, sym):
        self.disease_input = sym.replace(' ', '_')
        self.state = 'days'
        self.update_ui()

    def submit_days(self, instance):
        try:
            self.num_days = int(self.days_input.text)
            present_disease, symptoms_given = get_prediction(self.disease_input)
            self.present_disease = present_disease
            self.symptoms_given = symptoms_given
            self.symptoms_exp = []
            self.question_index = 0
            self.state = 'questions'
            self.update_ui()
        except:
            self.content.add_widget(Label(text='Enter a valid number.', color=(1, 0, 0, 1)))

    def answer_yes(self, instance):
        self.symptoms_exp.append(self.symptoms_given[self.question_index])
        self.question_index += 1
        self.update_ui()

    def answer_no(self, instance):
        self.question_index += 1
        self.update_ui()

    def show_results(self):
        second_prediction = sec_predict(self.symptoms_exp)
        condition = calc_condition(self.symptoms_exp, self.num_days)
        self.content.add_widget(Label(text=condition, color=(1, 0.5, 0, 1)))
        if self.present_disease[0] == second_prediction[0]:
            self.content.add_widget(Label(text=f'You may have {self.present_disease[0]}', color=(0, 1, 0, 1)))
            self.content.add_widget(Label(text=description_list.get(self.present_disease[0], '')))
        else:
            self.content.add_widget(Label(text=f'You may have {self.present_disease[0]} or {second_prediction[0]}', color=(1, 0, 0, 1)))
            self.content.add_widget(Label(text=description_list.get(self.present_disease[0], '')))
            self.content.add_widget(Label(text=description_list.get(second_prediction[0], '')))
        prec = precautionDictionary.get(self.present_disease[0], [])
        self.content.add_widget(Label(text='Take following measures:'))
        for i, p in enumerate(prec):
            self.content.add_widget(Label(text=f'{i+1}) {p}'))
        btn = Button(text='Start Over', background_color=(0, 0.5, 1, 1))
        btn.bind(on_press=self.reset)
        self.content.add_widget(btn)

    def show_results(self):
        second_prediction = sec_predict(self.symptoms_exp)
        condition = calc_condition(self.symptoms_exp, self.num_days)
        self.content.add_widget(Label(text=condition, color=(1, 0.5, 0, 1), font_size=18, size_hint_y=None, height=30))
        if self.present_disease[0] == second_prediction[0]:
            self.content.add_widget(Label(text=f'You may have {self.present_disease[0]}', color=(0, 1, 0, 1), font_size=20, size_hint_y=None, height=30))
            self.content.add_widget(Label(text=description_list.get(self.present_disease[0], ''), font_size=16, text_size=(380, None), halign='left', valign='top', size_hint_y=None, height=100))
        else:
            self.content.add_widget(Label(text=f'You may have {self.present_disease[0]} or {second_prediction[0]}', color=(1, 0, 0, 1), font_size=18, size_hint_y=None, height=30))
            self.content.add_widget(Label(text=description_list.get(self.present_disease[0], ''), font_size=16, text_size=(380, None), halign='left', valign='top', size_hint_y=None, height=100))
            self.content.add_widget(Label(text=description_list.get(second_prediction[0], ''), font_size=16, text_size=(380, None), halign='left', valign='top', size_hint_y=None, height=100))
        prec = precautionDictionary.get(self.present_disease[0], [])
        self.content.add_widget(Label(text='Take following measures:', font_size=18, size_hint_y=None, height=30))
        for i, p in enumerate(prec):
            self.content.add_widget(Label(text=f'{i+1}) {p}', font_size=16, size_hint_y=None, height=30))
        btn = Button(text='Start Over', background_color=(0, 0.5, 1, 1), font_size=24, size_hint_y=None, height=50)
        btn.bind(on_press=self.reset)
        self.content.add_widget(btn)