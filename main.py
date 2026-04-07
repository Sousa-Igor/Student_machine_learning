#%% 
import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb

#%% 
df = pd.read_csv("data/StudentPerformanceFactors.csv")
df.head()
df.shape
# %%
target = df["Exam_Score"]
target = pd.Series(target)
features = df.columns[:-1]
X = df.drop(target.name, axis = 1)


# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, target,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    )

print("Taxa variável resposta geral:", target.mean())
print("Taxa variável resposta Treino:", y_train.mean())
print("Taxa variável resposta Test:", y_test.mean())
# %%
X_train.isna().sum().sort_values(ascending = False) 

# %%
# Criando variaveis dummies

X_train = pd.get_dummies(X_train,
                        dtype=int,
                        dummy_na=True)
X_train.columns


# %%
X_test = pd.get_dummies(X_test,
                        dtype=int,
                        dummy_na=True)
X_train, X_test = X_train.align(X_test,join="left",axis=1,fill_value=9999999999)
# %%
X_train.sum().sort_values(ascending=False)
# %%

X_test.sum().sort_values(ascending=False)
# %%
arvore = tree.DecisionTreeRegressor()
arvore.fit(X_train,y_train)

#%%
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)

#%%
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
#%%
y_arvore_predict = arvore.predict(X_test)
y_reg_predict = reg.predict(X_test)
y_rf_predict = rf.predict(X_test)
# %%
print("MSE da arvore:",metrics.mean_squared_error(y_test,y_arvore_predict))
print("MSE da regressão:",metrics.mean_squared_error(y_test,y_reg_predict))
print("MSE do random forest:",metrics.mean_squared_error(y_test,y_rf_predict))


# %%
pd.DataFrame({"y_test":y_test,
             "y_preidto":y_reg_predict.round(0)})

# %%
sb.regplot(x = y_test,y = y_reg_predict, line_kws={"color": "red"}, scatter_kws={"color": "blue"})