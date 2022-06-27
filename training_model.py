import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_merge.csv')
df = df.set_index("session_id")

cluster_assign = pd.read_csv('cluster_assign.csv')
cluster_assign = cluster_assign[['item_id2', 'Cluster_pred']]

number_of_clusters = 4

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=40)
len(train_set),len(test_set)

cat_vars = ["first_prod", "last_prod", "time_first_prod", "most_common_cat", "most_seen_cluster", "first_item_cluster", "last_item_cluster"]
num_vars = ["prod_count", "time_diff", "time_per_prod", "count_num_unique_cat"] + [str(i) for i in range(0,number_of_clusters)]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train_set_num = train_set[num_vars]

num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

from sklearn.preprocessing import OneHotEncoder

train_set_cat = train_set[cat_vars]

#Establish values for each category -> otherwise the encoder does not work for the test set
items=cluster_assign.item_id2.to_list()
cats=list(range(1,74))
clusts=list(range(0,number_of_clusters))
times=['día','madrugada','tarde','noche']

cat_pipeline = Pipeline([
        ('one_hot_encoder', OneHotEncoder(categories=[items,items,times,cats,clusts,clusts,clusts], handle_unknown='ignore'))
    ])

from sklearn.compose import ColumnTransformer

num_attribs = train_set_num.columns
cat_attribs = train_set_cat.columns

full_pipeline_train = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

train_set_prepared = full_pipeline_train.fit_transform(train_set)
train_set_prepared

from sklearn.ensemble import RandomForestClassifier

forest_class = RandomForestClassifier(n_estimators=20, random_state=42)
forest_class.fit(train_set_prepared, train_set[["item_purch"]].to_numpy().ravel())

test_set_prepared = full_pipeline_train.fit_transform(test_set)

def mean_reciprocal_rank(model, set_to_test, rr = 0, cont = 0):

    predictions = model.predict_proba(set_to_test)

    pred_df_test = pd.DataFrame(predictions)
    pred_df_test.columns = model.classes_

    pred_df_test["session_id"] = test_set.index
    pred_df_test = pred_df_test.merge(df[["item_purch"]], how='inner', on='session_id')
    pred_df_test = pred_df_test[["session_id", "item_purch"] + list(pred_df_test.columns[:-2])]

    for index, row in pred_df_test.iterrows():
        item_purch_act = int(row.iloc[1])
        row = row.iloc[2:]
        row_sorted = row.sort_values(ascending=False)
        items = row_sorted.index.to_list()
        if item_purch_act in items:
            rank = int(items.index(item_purch_act)) + 1
            if rank <= 100:
                rr += 1/rank
        cont += 1

    mrr = rr/cont
    print("The mean reciprocal rank for the " + str(model) + " is " + str(mrr))

mean_reciprocal_rank(forest_class, test_set_prepared)

