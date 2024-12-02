import numpy as np
np.random.seed(1881)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from geopy.distance import geodesic
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False) 
pd.set_option('display.float_format', '{:.4f}'.format)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('D:/Users/sefa.erkan/Desktop/Credit_Card/credit_card_transactions.csv')
df.head()

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

data_types = pd.DataFrame(df.dtypes).reset_index()
data_types.columns = ['Feature Name', 'Data Type']
data_types

missing_counts = df.isnull().sum()
missing_counts_df = pd.DataFrame(missing_counts).reset_index()
missing_counts_df.columns = ['Feature','Missing Value Count']
missing_counts_df

unique_counts = df.nunique()
unique_counts_df = pd.DataFrame(unique_counts).reset_index()
unique_counts_df.columns = ['Feature', 'Number of Unique Values']
unique_counts_df

print(f'Number of duplicated rows: {df.duplicated().sum()}')

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='is_fraud', data=df, palette='viridis')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=10, color='black')
plt.title('is_fraud Distribution')
plt.xlabel('is_fraud')
plt.ylabel('Count')
plt.show()

categorical_columns = ['category','gender']

for column in categorical_columns:
    plt.figure(figsize=(12,6))
    ax = sns.countplot(y=column, hue='is_fraud',data=df,palette='viridis')
    plt.title(f'{column} Distribution by is_fraud')
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.legend(title='credit approval', loc='upper right', bbox_to_anchor=(1.15,1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 10, p.get_y() + p.get_height() / 2, f'{int(width)}',ha='left', va='center')

    plt.show()    

# Aykırı değer tespiti için kullanabileceğim tek sayısal değer işlem tutarıdır (amt).
plt.figure(figsize=(10,6))
plt.scatter(df.index, df['amt'], color='red')
plt.title("Scatter Plot of 'amt'")
plt.xlabel('Index')
plt.ylabel('amt')
plt.show()

# 2500-3000 noktaları sonrasında dağılmalar başlamıştır.
outlier_threshold = 2700
outliers = df['amt'] > outlier_threshold
outlier_count = np.count_nonzero(outliers)
total_count = len(df)
outlier_percentage = (outlier_count / total_count) * 100

plt.figure(figsize=(10,6))
plt.scatter(df.index[outliers], df['amt'][outliers], color='lightcoral', label='Outliers')
plt.scatter(df.index[~outliers], df['amt'][~outliers], color='red', label='Value Points')
plt.axhline(y=outlier_threshold, color='black', linestyle='--', label=f'Outlier Treshhold: {outlier_threshold}')
plt.title('Outlier Analysis Using Scatter Plot')
plt.xlabel('Index')
plt.ylabel('amt Values')
plt.legend()
plt.show()

print(f'Number of outliers: {outlier_count}')
print(f'Outlier percentage: % {outlier_percentage}')

# Veriden outliers'leri kaldıracağım
# 430 değer kaybolacak ve bu verinin 0,03'ü demek.
df = df[~outliers]

# Gereksiz kolonları kaldırıyorum
drop_columns =  ['Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 'zip', 'trans_num','unix_time','merch_zipcode']
df = df.drop(columns=drop_columns)

# Yeni kolonlar (featurelar) ekliyoruz
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_year'] = df['trans_date_trans_time'].dt.year
df['trans_month'] = df['trans_date_trans_time'].dt.month
df['trans_day'] = df['trans_date_trans_time'].dt.day
df['trans_season'] = df['trans_date_trans_time'].dt.month % 12 // 3 + 1
df['trans_weekday'] = df['trans_date_trans_time'].dt.weekday
df['trans_hour'] = df['trans_date_trans_time'].dt.hour
df['trans_minute'] = df['trans_date_trans_time'].dt.minute
df['trans_second'] = df['trans_date_trans_time'].dt.second

df = df.drop(columns=['trans_date_trans_time'])

# Kart sahbinin işlemin gerçekleştiği andaki yaşını hesaplayalım
df['dob'] = pd.to_datetime(df['dob'])
df['birth_year'] = df['dob'].dt.year
df['card_holder_age'] = df['trans_year'] - df['birth_year']
df = df.drop(columns=['dob', 'birth_year'])

# Geopy kütüphanesini kullanarak enlem ve boylam koordinatlarını kullanarak
# iki nokta arasındaki (işlem yapılan yerin, kart sahibinin konumu) coğrafi mesafeyi hesaplayacaktır.
def calculate_distance(row):
    point_a = (row['lat'], row['long'])
    point_b = (row['merch_lat'], row['merch_long'])
    return geodesic(point_a, point_b).kilometers

df['distance'] = df.apply(calculate_distance, axis=1)

# Kategorik değerleri sayısal değerlere dönüştürmek
def encode_categorical_columns(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

cat_features = ['cc_num', 'merchant', 'category', 'gender', 'job']
df = encode_categorical_columns(df, cat_features)
print(df.head())

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1881)

print("y_train class distribution:")
print(y_train.value_counts(normalize=True))

print("\ny_test class distribution:")
print(y_test.value_counts(normalize=True))

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

models = {
    "Catboost": CatBoostClassifier(
        depth=10,
        learning_rate=0.2,
        n_estimators=2000,
        min_child_samples=10,
        subsample=0.7,
        l2_leaf_reg=8,
        cat_features=cat_features,
        random_state=1881,
        eval_metric='F1',
        loss_function='Logloss',
        bootstrap_type='Bernoulli',
        class_weights=class_weight_dict,
        task_type='GPU',
        verbose=False
    ),
    "XGBoost": XGBClassifier(
        max_depth=7,
        learning_rate=0.2,
        n_estimators=2000,
        min_child_weight=10,
        subsample=0.8,
        reg_lambda=1,
        reg_alpha=3,
        scale_pos_weight=class_weight_dict[1] / class_weight_dict[0],
        objective='binary:logistic', 
        eval_metric='logloss',
        tree_method='gpu_hist', 
        random_state=1881,
        verbose=False
    ),
    "LGBM": LGBMClassifier(
        max_depth=8,
        num_leaves=64,
        learning_rate=0.03,
        n_estimators=2000,
        min_child_weight=10,
        subsample=0.9,
        reg_lambda=3,
        reg_alpha=1,
        scale_pos_weight=class_weight_dict[1] / class_weight_dict[0],
        objective='binary',  
        metric='binary_logloss',  
        random_state=1881,
        device="gpu",
        verbose=-1
    )
}

results = {
    model_name: {
        "y_tests" : [],
        "y_preds" : [],
        "f1_scores" : [],
        "precisions" : [],
        "recalls" : [],
    }
    for model_name in models.keys()
}

# Çarpraz Doğrulama 
# Bu strateji, özellikle sınıflar arasında dengeli bir veri dağılımı sağlamak için kullanılır.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1881)

for model_name, model in models.items():
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_hold = model.predict(X_test_fold)

        f1 = f1_score(y_test_fold, y_pred_hold)
        precision = precision_score(y_test_fold, y_pred_hold)

        recall = recall_score(y_test_fold, y_pred_hold)
        
        results[model_name]["y_tests"].append(y_test_fold)
        results[model_name]["y_preds"].append(y_pred_hold)
        results[model_name]["f1_scores"].append(f1)
        results[model_name]["precisions"].append(precision)
        results[model_name]["recalls"].append(recall)

average_results = {
    "Model": [],
    "Mean F1 Score": [],
    "Mean Precision": [],
    "Mean Recall": []
}      

for model_name, metrics in results.items():
    average_results["Model"].append(model_name)
    average_results["Mean F1 Score"].append(sum(metrics["f1_scores"]) / len(metrics["f1_scores"]))
    average_results["Mean Precision"].append(sum(metrics["precisions"]) / len(metrics["precisions"]))
    average_results["Mean Recall"].append(sum(metrics["recalls"]) / len(metrics["recalls"]))

df_results = pd.DataFrame(average_results)
df_results


voting_clf = VotingClassifier(
    estimators=[
        ('Catboost', models["Catboost"]),
        ('XGBoost', models["XGBoost"]),
        ('LGBM', models["LGBM"])
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

metrics = {
    'Metric': ['F1 Score', 'Precision', 'Recall'],
    'Value': [f1, precision, recall]
}

metrics_df = pd.DataFrame(metrics)
metrics_df

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', cbar=True, linewidths=1, linecolor='black')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

clf_report = classification_report(y_test, y_pred, output_dict=True)
rep_df = pd.DataFrame(clf_report).transpose()
rep_df

catboost_importance = models['Catboost'].feature_importances_
xgboost_importance = models['XGBoost'].feature_importances_
lgbm_importance = models['LGBM'].feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'CatBoost': catboost_importance,
    'XGBoost': xgboost_importance,
    'LGBM': lgbm_importance
})
importance_df['Average'] = importance_df[['CatBoost', 'XGBoost', 'LGBM']].mean(axis=1)
top_features = importance_df.nlargest(10, 'Average')

plt.figure(figsize=(10, 6))
sns.barplot(x='Average', y='Feature', data=top_features, palette='viridis')
plt.title('Top 10 Important Features')
plt.xlabel('Average Importance')
plt.ylabel('Feature')
plt.show()