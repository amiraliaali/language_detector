import pandas as pd
import numpy as np
def mL():
    dataSet = pd.read_csv("Language Detection.csv")
    print(dataSet.head())                                   # We have an overview of the first four lines of the DataSet
    print("Total rows in the DS = " + str(dataSet.size))    # 20674 rows
    print("=================================")
    print("If there is any 'Language' row with Nan data: " + str(dataSet[dataSet["Language"].isnull()]))       #Empty Data Frame
    print("If there is any 'Text' row with Nan data: " + str(dataSet[dataSet["Text"].isnull()]))       #Empty Data Frame
    print("=================================")
    print(str(dataSet["Language"].nunique()) + " total different languages")     # 17 different languages
    print(dataSet["Language"].unique())                     # ['English' 'Malayalam' 'Hindi' 'Tamil' 'Portugeese' 'French' 'Dutch'
                                                            # 'Spanish' 'Greek' 'Russian' 'Danish' 'Italian' 'Turkish' 'Sweedish'
                                                            # 'Arabic' 'German' 'Kannada']
    print("English = " + str(dataSet[dataSet["Language"] == "English"]["Language"].size))       # 1385 rows for english
    print("German = " + str(dataSet[dataSet["Language"] == "German"]["Language"].size))         # 470 rows for german
    print(dataSet["Language"].value_counts())               # Overall count of every language

    # Start of the Machine Learning
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    x = dataSet["Text"]
    y = dataSet["Language"]

    cv = CountVectorizer()
    X = cv.fit_transform(x)
    results = []
    bestRandomState = 0
    bestResult = 0
    for i in range(30, 60):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model = MultinomialNB()
        model.fit(X_train,y_train)
        if model.score(X_test,y_test) > bestResult:
            bestRandomState = i
            bestResult = model.score(X_test,y_test)
        results.append( model.score(X_test,y_test) )
    print(len(results))

    # Now we check for the best parameters
    import seaborn as sns
    import matplotlib.pyplot as plt
    #sns.lineplot(x = [int(i) for i in range(30, 60)], y = results)      #turns out random_state = 58 gives the best result

    #Final machine trainings
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=bestRandomState)
    model = MultinomialNB()
    model.fit(X_train,y_train)
    results.append( model.score(X_test,y_test) )
    plt.show()
    return model, cv


if __name__ == '__main__':
    model, cv = mL()
    user = input("Enter a Text: ")
    data = cv.transform([user])
    output = model.predict(data)
    print("The detected language of the input text is : " + output[0])
