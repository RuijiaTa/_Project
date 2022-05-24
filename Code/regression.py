import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.linear_model import Lasso

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

'''
Step1: Prepare the Data.
'''
### Open the data set, and read the content of the house prices.
df = pd.read_fwf("housing.csv")
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print("Read Housing Prices:")
print(df)

### Divide the whole data into train set and test set. Pick 20% to be test set and the rest to be trained by models.
print("Get the shape of Train Set and Test Set.(X:Inputs; Y:MEDV)")
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
print("X_train.shape:", X_train.shape)
print("Y_train.shape:", Y_train.shape)
print("X_test.shape:", X_test.shape)
print("Y_test.shape:", Y_test.shape)

##############################################################

'''
Step2: Analyse the data set. 
'''

def analyse_data(price):
    ### Analyse the content of the data. Check the number of data, the mean, std, min,max...
    print("Analyse the Data Set：")
    print(price.describe())

    ### The red line represents the normal distribution, and the blue line represents the sample data. The closer the blue
    # is to the red reference line, the more it conforms to the expected distribution (normal distribution).
    print("(Generate the whole Displot and Porbplot)")
    sns.distplot(np.log(df["MEDV"]), fit=norm)
    plt.title("Estimation Diagram")
    plt.figure()
    stats.probplot(np.log(df["MEDV"]), plot=plt)
    plt.show()

    ###Analyze the displot and scatter plot by single inputs
    # print("(Generate Displots and Scatter of 13 Inputs)")
    # for item in price.columns:
    #     if item != "MEDV":
    #         plt.figure(1, figsize=(8, 4))
    #         sns.scatterplot(x=item, y="MEDV", data=df)
    #         plt.show()
    #         fig = plt.figure()
    #         sns.distplot(price[item])
    #         plt.title(item)
    #         plt.show()

    # ### Three inputs which affect the MEDV most. And plot the scatter plot of them.
    # print("(Seperate three most important factors' scatter plots)")
    # plt.figure(figsize=(10, 5))
    # plt.subplot(131)
    # sns.scatterplot(x='ZN', y='MEDV', data=df)
    # plt.subplot(132)
    # sns.scatterplot(x='RM', y='MEDV', data=df)
    # plt.subplot(133)
    # sns.scatterplot(x='B', y='MEDV', data=df)
    # plt.tight_layout()

    ### Judge the correlation between different factors. And generate the heatmap to plot the result.
    print("(Generate Heatmap of Correlation)")
    df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(price.corr(), annot=True)
    plt.title("Correlation HeatMap")
    plt.show()
    return

analyse_data(df)

##############################################################

'''
Step3: Analyse the models.
'''
'''1. Linear Regression'''


def linear_regression(x, y):
    ### Use the model of OLS to pridicate the house price.
    reg1 = sm.OLS(y, sm.add_constant(x)).fit()
    result_OLS = reg1.summary()
    print("OLS Result:", result_OLS)
    print("###########################################################################")

    ### Using Linear Regression to Define the sequence factors's importance'.
    reg1.params.iloc[1:].plot(kind='barh')
    plt.title("Importance of Histogram under Linear Regression")
    plt.show()
    print("###########################################################################")

    ### Get the Linear Regression's RMSE and Score the model of Linear Regression.
    pred1 = reg1.predict(sm.add_constant(X_test))
    print("-Linear Regression Result:")
    print("RMSE:", np.sqrt(mean_squared_error(Y_test, pred1)))
    # The fitting standard deviation of regression system is a parameter corresponding to
    # the original data and the predicted data in linear regression model.
    X = df.drop(['MEDV'], axis=1).values
    Y = np.log(df['MEDV'].values)
    reg1 = LinearRegression(fit_intercept=True).fit(X, Y)
    print("Score:", reg1.score(X, Y))  # Use the cross-validation model evaluation tool
    plt.plot(Y_test, pred1, 'o')
    plt.plot([0, 50], [0, 50], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('MEDV')
    plt.ylabel('Prediction')
    plt.title("Scatter plot of Linear Regression")
    plt.show()
    # Use line chart to check the result of Linear Regression.
    plt.plot(range(len(pred1)), pred1, 'b', label="Predict data")
    plt.plot(range(len(Y_test)), Y_test, 'r', label="Real data")
    plt.legend(loc=2)
    plt.title("Line chart of Linear Regression")
    plt.show()
    print("(Generate the Scatter Plot and Line Chart to visualize the result of Linear Regression)")
    print("###########################################################################")

    ### Pick Losso Regression as the optimization model
    # Lasso Regression （Least Absolute Shrinkage and Selection Operator）
    # Lasso also penalizes the absolute value of its regression coefficient.
    # This makes it possible for the value of the punishment to go to zero
    lasso = Lasso()
    lasso.fit(x, y)
    y_predict_lasso = lasso.predict(X_test)
    plt.plot(Y_test, pred1, 'o', label="OLS")
    plt.plot(Y_test, y_predict_lasso, 'o', label="Lasso")
    plt.legend()
    plt.plot([0, 50], [0, 50], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.title("Comparison of Lasso and Linear Regression")
    plt.show()
    print("Lasso Regression Result:")
    print('RMSE:', np.sqrt(mean_squared_error(Y_test, y_predict_lasso)))  # RMSE(标准误差)
    print('SCORE:', lasso.score(X_test, Y_test))  # 模型评分
    print("(Compare the result of Linear Regression and Lasso Regression)")
    print("###########################################################################")
    return pred1


# linear_regression(X_train,Y_train)

'''2. KNN Regression'''


def knn_regression(x, y):
    ### Use the KNN Regression as the model to predict. And judge the result of it by RMSE and Score.
    reg2 = KNeighborsRegressor(n_neighbors=2).fit(x, y)
    pred2 = reg2.predict(X_test)
    print("-KNN Regression Result:")
    print("RMSE:", np.sqrt(mean_squared_error(Y_test, pred2)))
    print("SCORE:", reg2.score(X_test, Y_test))

    ### Get the result of the KNN Regression and compare with the real data.
    plt.plot(Y_test, pred2, 'o')
    plt.plot([0, 50], [0, 50], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('MEDV')
    plt.ylabel('Prediction')
    plt.title("Scatter plot of KNN Regression")
    plt.show()
    print("(Generate the Scatter Plot of the result of KNN Regression)")
    print("###########################################################################")

    ### Use MSE which can better reflect the actual situation of predicted value error. To compare with the neighbors.
    knn_res = []
    for idx in range(2, 20):
        reg2 = KNeighborsRegressor(n_neighbors=idx).fit(x, y)
        pred2 = reg2.predict(X_test)
        knn_res.append(np.abs((Y_test - pred2)).mean())
    plt.plot(range(2, 20), knn_res)
    plt.xticks(range(2, 20), range(2, 20))
    plt.xlabel('n_neighbors')
    plt.ylabel('MAE')
    plt.title("n_neighbors affect MAE")
    plt.show()
    print("(Find the number of neighbors which can get the best accuracy of prediction by line chart.)")
    print("###########################################################################")
    return pred2


# knn_regression(X_train,Y_train)

'''3. SVM Regression'''


def svm_regression(x, y):
    ### Use the SVM Regression as the model to predict. And judge the result of it by RMSE and Score.
    reg3 = SVR().fit(x, y)
    pred3 = reg3.predict(X_test)
    print("-SVM Regression Result:")
    print("RMSE:", np.abs((Y_test - pred3)).mean())
    print("SCORE:", reg3.score(X_test, Y_test))

    ### Get the result of the SVM Regression and compare with the real data.
    plt.plot(Y_test, pred3, 'o')
    plt.plot([0, 50], [0, 50], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('MEDV')
    plt.ylabel('Prediction')
    plt.title("Scatter of SVM Regression")
    plt.show()
    print("(Generate the Scatter Plot of the result of SVM Regression)")
    print("###########################################################################")
    return pred3


# svm_regression(X_train,Y_train)

'''4. RandomForest'''


def random_forest_regression(x, y):
    ### Use the RandomForest Regression as the model to predict. And judge the result of it by RMSE and Score.
    reg4 = RandomForestRegressor().fit(x, y)
    pred4 = reg4.predict(X_test)
    print("-RandomForest Regression Result:")
    print("RMSE:", np.abs((Y_test - pred4)).mean())
    print("SCORE:", reg4.score(X_test, Y_test))

    ### Using RandomForest Regression to Define the sequence factors's importance'.
    plt.barh(range(13), reg4.feature_importances_)
    plt.yticks(range(13), x.columns[:])
    plt.title("Importance of Histogram under RandomForest Regression")
    plt.show()
    print("###########################################################################")

    ### Get the result of the RandomForest Regression and compare with the real data.
    plt.plot(Y_test, pred4, 'o')
    plt.plot([0, 50], [0, 50], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('MEDV')
    plt.ylabel('Prediction')
    plt.title("Scatter of RandomForest Regression")
    plt.show()
    print("(Generate the Scatter Plot of the result of RandomForest Regression)")
    print("###########################################################################")

    ### Use MSE which can better reflect the actual situation of predicted value error. To compare with the max_depth.
    rf_res = []
    for idx in range(2, 20):
        reg4 = RandomForestRegressor(max_depth=idx).fit(x, y)
        pred4 = reg4.predict(X_test)
        rf_res.append(np.abs((Y_test - pred4)).mean())
    plt.plot(range(2, 20), rf_res)
    plt.xticks(range(2, 20), range(2, 20))
    plt.xlabel('max_depth')
    plt.ylabel('MAE')
    plt.title("max_depth affect MAE")
    plt.show()
    print("(Find the best max depth of tree which can get the best accuracy of prediction by line chart.)")

    ### Use MSE which can better reflect the actual situation of predicted value error. To compare with the number of estimator.
    rf_res = []
    for idx in range(20, 100, 10):
        reg4 = RandomForestRegressor(n_estimators=idx).fit(x, y)
        pred4 = reg4.predict(X_test)
        rf_res.append(np.abs((Y_test - pred4)).mean())

    plt.plot(range(20, 100, 10), rf_res)
    plt.xticks(range(20, 100, 10), range(20, 100, 10))
    plt.xlabel('n_estimators')
    plt.title("n_estimators affect MAE")
    plt.ylabel('MAE')
    plt.show()
    print("###########################################################################")
    return pred4


# random_forest_regression(X_train,Y_train)

##############################################################

'''
Step4: Comparing 4 Regression Models
'''


def compare_4_models(p1, p2, p3, p4):
    plt.plot(Y_test, p1, 'o', label="OLS")
    plt.plot(Y_test, p2, 'o', label="KNN")
    plt.plot(Y_test, p3, 'o', label="SVM")
    plt.plot(Y_test, p4, 'o', label="RF")
    plt.plot([0, 50], [0, 50], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.legend()
    plt.title("Comparison of 4 models")
    plt.show()
    print("Finish the Boston House Price Prediction!")
    print("The Best Choice is RandomForest Regression")


compare_4_models(linear_regression(X_train, Y_train), knn_regression(X_train, Y_train),
                 svm_regression(X_train, Y_train), random_forest_regression(X_train, Y_train))
