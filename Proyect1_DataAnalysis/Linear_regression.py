from Database_connection import data_preprocesing, df_country
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as LG

def linearRegression_model(x_train, y_train):
    #Regression with Scikitlearn
    reg = LG().fit(x_train, y_train)
    #valores predichos para Y, realizados por la regresion
    xtrain_Prediction = reg.predict(x)
    
    #R^2 Score
    print('R^2 = ',reg.score(x_train, y_train,))

    print('Slope:', reg.coef_)
    print('y - Intercept:', reg.intercept_)
    print('\nLine equation: ', 'LifeExpentancy (GNP) =', str(np.round(reg.coef_[0][0],3)) ,'*','ln(GNP)',' + ',np.round(reg.intercept_[0],3),'\n')
    plt.title('GNP Porcentual increment vs life expectancy')
    plt.xlabel('Log(GNP per country')
    plt.ylabel('Life expectancy')
    plt.plot(x_train,y_train, '.r'), plt.plot(x_train, xtrain_Prediction)
    plt.show()


x, y, kmeans_data, CountryName = data_preprocesing(df_country)
linearRegression_model(x,y)