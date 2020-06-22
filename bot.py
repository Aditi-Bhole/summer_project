import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def pie(confidence_level):
    cl = []
    cl.append(confidence_level)
    cl.append(100-confidence_level)
    cll = ['May have','May not have']
    fig = plt.figure(figsize =(10, 7)) 
    plt.pie(cl, labels = cll) 
    plt.show()

def print_disease(node,le):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero() 
    #print(val)
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree, feature_names, le, reduced_data):
    tree_ = tree.tree_
    #print(tree_)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))
    symptoms_present = []
    symptoms_ask = []
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("  "+name + " ?")
            symptoms_ask.append(name)
            ans = input().lower()
            if ans == 'yes':                    
                val = 1
            else :
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            pd=" "
            present_disease = print_disease(tree_.value[node],le)
            pd = pd.join(present_disease)
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            sg = set(symptoms_given)
            sa = set(symptoms_ask)
            sa = sg.difference(sa)
            symptoms_task = list(sa)
            if pd == "Type 1" or pd =="Type 2":
                symptoms_task = symptoms_task[:20]
            for i in symptoms_task:
                print("  "+i+" ?")
                ans = input().lower()
                if ans == 'yes':
                    symptoms_present.append(i)
            confidence_level = ((1.0*len(symptoms_present))/len(symptoms_given))*100
            if confidence_level > 15:
                print( "\n  You may have " + pd +" diabetes!")
                print("\n  Symptoms present : ")
                for i in symptoms_present:
                    print("    "+i)
                pie(confidence_level)
                print("\n  Confidence level is " + str(confidence_level)+"%")
            else:
                print("  You don't have diabetes!")
    recurse(0, 1)

def mop():
    training = pd.read_csv('data/Training.csv')
    testing  = pd.read_csv('data/Testing.csv')
    cols     = training.columns
    cols     = cols[:-1]
    x        = training[cols]
    y        = training['prognosis']
    y1       = y
    
    reduced_data = training.groupby(training['prognosis']).max()
    
    #mapping strings to numbers
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    testx    = testing[cols]
    testy    = testing['prognosis']  
    testy    = le.transform(testy)

    clf1  = DecisionTreeClassifier()
    clf = clf1.fit(x_train,y_train)
    #print(clf.score(x_train,y_train))
    #print ("cross result========")
    #scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=3)
    #print (scores)
    #print (scores.mean())
    #print(clf.score(testx,testy))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = cols

    #feature_importances
    #for f in range(10):
    #    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))

    tree_to_code(clf,cols,le,reduced_data)

def female():
    training = pd.read_csv('data/Training_f.csv')
    testing  = pd.read_csv('data/Testing_f.csv')
    cols     = training.columns
    cols     = cols[:-1]
    x        = training[cols]
    y        = training['prognosis']
    y1       = y
    
    reduced_data = training.groupby(training['prognosis']).max()
    
    #mapping strings to numbers
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    testx    = testing[cols]
    testy    = testing['prognosis']  
    testy    = le.transform(testy)

    clf1  = DecisionTreeClassifier()
    clf = clf1.fit(x_train,y_train)
    #print(clf.score(x_train,y_train))
    #print ("cross result========")
    #scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=3)
    #print (scores)
    #print (scores.mean())
    #print(clf.score(testx,testy))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = cols

    #feature_importances
    #for f in range(10):
    #    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))

    tree_to_code(clf,cols,le,reduced_data)

def getReply(a):
    if 'who' in a and 'you' in a:
        print("  Hello, I'm Diabot.")
    elif 'hi' in a or 'hello' in a:
        print("  Hi! What is your name?")
    elif 'my name' in a or 'name' in a:
        print("  What is your age?")
    elif 'my age is' in a or 'i am' in a:
        print("  Select your gender:\n    1) Male\n    2) Female\n    3) Other\n    4) Prefer not to say.")
    elif 'female' in a:
        print("  Please answer the following in Yes or No only...")
        female()
        print("  Do you need a report ?")
    elif 'male' in a or 'other' in a or 'prefer' in a:
        print("  Please answer the following in Yes or No only...")
        mop()
        print("  Do you need a report ?")
    elif 'yes' in a or 'yeah' in a or 'sure' in a:
        print("  Can you give us your e-mail address?")
    elif 'no' in a:
        print("  Ok!\n  Please visit a doctor for further consultation.\n  How will you rate us?\n  Are there any suggestions?")
    elif 'email' in a or '@' in a:
        print("  You will get the report in some time.\n  Please visit a doctor for further consultation.\n  How will you rate us?\n  Are there any suggestions?")
    elif 'rate' in a or 'suggestion' in a:
        print("  Thank you!")
    elif 'bye' in a:
        print("  Bye! Thanks for visiting.")
    else:
        print("  Sorry I don't understand what you are saying!")

print("Type bye to terminate and return to home.")
print("Hi! Welcome to Diabot.\nPlease fill the following information...\n\n  What is your name? \n  And your age?")
a = ''
while ('bye' not in a):
    a = input()
    a = a.lower()
    getReply(a)
