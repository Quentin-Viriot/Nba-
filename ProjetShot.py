import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib as plt

shot_location = pd.read_csv('NBA Shot Locations 1997 - 2020.csv', sep=',')

top_players = ['LeBron James', 'Kobe Bryant', 'Tim Duncan', "Shaquille O'Neal", 'Stephen Curry', 'Kevin Durant', 'Dwyane Wade', 'Giannis Antetokounmpo', 'Kevin Garnett', 'Dirk Nowitzki', 'Kawhi Leonard', 'Allen Iverson', 'Steve Nash', 'Tony Parker', 'Damian Lillard', 'Paul Pierce', 'Jason Kidd', 'Russel Westbrook', 'Ray Allen', 'James Harden']
shot_location.info()
shot_loc_top_players = shot_location[shot_location['Player Name'].isin(top_players)]

variables_selection = ['Game Date','Action Type','Shot Made Flag', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Shot Distance', 'X Location', 'Y Location', 'Season Type','Player Name']

shot_loc = shot_loc_top_players[variables_selection]
shot_loc.rename(columns={'Player Name': 'Player'}, inplace=True)

print(shot_loc)
shot_loc.info()
shot_loc.isna().sum()
shot_loc.head()

shot_loc['Game Date'] = shot_loc['Game Date'].astype(str)
shot_loc['Year'] = shot_loc['Game Date'].str[:4]
shot_loc['Year'] = pd.to_numeric(shot_loc['Year'])

shot_loc = shot_loc[(shot_loc['Year'] >= 2000) & (shot_loc['Year'] <= 2020)]

print(shot_loc)
shot_loc.info()

#Visualisation distribution
import matplotlib.pyplot as plt

categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']

for var in categorical_variables:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=var, data=shot_loc)
    plt.title(f"Distribution de {var}")
    plt.xticks(rotation=45)
    plt.show()

plt.figure(figsize=(10, 6))
sns.stripplot(x='Action Type', data=shot_loc, jitter=True, size=5)
plt.title("Distribution de Action Type")
plt.xticks(rotation=45)
plt.show()

numerical_variables = ['Shot Distance', 'X Location', 'Y Location', 'Year']

for var in numerical_variables:
    plt.figure(figsize=(8, 5))
    sns.histplot(shot_loc[var], bins=20, kde=True)
    plt.title(f"Distribution de {var}")
    plt.xlabel(var)
    plt.ylabel("Count")
    plt.show()

# Filtre des 5 occurrences les plus représentées dans la colonne "Action Type"
top_action_types = shot_loc['Action Type'].value_counts().nlargest(5).index
shot_loc_filtered = shot_loc[shot_loc['Action Type'].isin(top_action_types)]

legend_ax = plt.gca()
legend_ax.set_axis_off()
legend_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.figure(figsize=(12, 8))
sns.relplot(data=shot_loc_filtered, x='Shot Distance', y='Action Type', hue='Player', col='Player', col_wrap=4, height=3, aspect=1.5, palette='Set1')
plt.xlabel('Type d\'action')
plt.ylabel('Type de tir')
plt.suptitle("Relation entre le type de tir et le type d'action pour chaque joueur (Top 5)")
plt.tight_layout()
plt.show()

shot_loc_filtered_players = shot_loc[shot_loc['Player'].isin(top_players[:5])]
top_action_types = shot_loc_filtered_players['Action Type'].value_counts().nlargest(5).index

for player_name in top_players[:5]:
    shot_loc_filtered = shot_loc_filtered_players[(shot_loc_filtered_players['Player'] == player_name) & (shot_loc_filtered_players['Action Type'].isin(top_action_types))]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=shot_loc_filtered, x='Shot Distance', y='Action Type', hue='Player', palette='Set1')
    plt.title(f"Type de tir par rapport à la distance de tir pour {player_name}")
    plt.xlabel('Distance de tir')
    plt.ylabel('Type d\'action')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

for player_name in top_players[:5]:
    shot_loc_filtered = shot_loc_filtered_players[(shot_loc_filtered_players['Player'] == player_name) ]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=shot_loc_filtered, x='Shot Distance', y='Shot Zone Basic', hue='Player', palette='Set1')
    plt.title(f"Type de tir par rapport à la distance de tir pour {player_name}")
    plt.xlabel('Distance de tir')
    plt.ylabel('Type d\'action')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

class_counts = shot_loc['Shot Made Flag'].value_counts()
print(class_counts)
total_samples = len(shot_loc)
class_proportions = class_counts / total_samples
print(class_proportions)

sns.countplot(x='Shot Made Flag', data=shot_loc)
plt.title("Répartition des classes")
plt.xlabel("Shot Made Flag")
plt.ylabel("Nombre d'occurrences")
plt.show()


#Modélisation 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

shot_loc_active_players = shot_loc[shot_loc['Player'].isin(top_players)]

shot_loc_regular_season = shot_loc_active_players[shot_loc_active_players['Season Type'] == 'Regular Season']
shot_loc_playoffs = shot_loc_active_players[shot_loc_active_players['Season Type'] == 'Playoffs']

variables_selection = ['Game Date', 'Action Type', 'Shot Made Flag', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Shot Distance', 'X Location', 'Y Location', 'Season Type', 'Player']
shot_loc_regular_season = shot_loc_regular_season[variables_selection]
shot_loc_playoffs = shot_loc_playoffs[variables_selection]

numeric_columns = shot_loc_active_players[['Shot Distance', 'X Location', 'Y Location']]
scaler = StandardScaler()
scaled_numeric_columns = scaler.fit_transform(numeric_columns)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_numeric_columns)
shot_loc_active_players['PCA Component 1'] = pca_result[:, 0]
shot_loc_active_players['PCA Component 2'] = pca_result[:, 1]

categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']
shot_loc_regular_season_encoded = pd.get_dummies(shot_loc_regular_season, columns=categorical_variables)
shot_loc_playoffs_encoded = pd.get_dummies(shot_loc_playoffs, columns=categorical_variables)

X_reg_season = shot_loc_regular_season_encoded.drop(columns=['Shot Made Flag'])
y_reg_season = shot_loc_regular_season_encoded['Shot Made Flag']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_season, y_reg_season, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']
shot_loc_regular_season_encoded = pd.get_dummies(shot_loc_regular_season, columns=categorical_variables)
shot_loc_playoffs_encoded = pd.get_dummies(shot_loc_playoffs, columns=categorical_variables)

X_reg_season = shot_loc_regular_season_encoded.drop(columns=['Shot Made Flag'])
y_reg_season = shot_loc_regular_season_encoded['Shot Made Flag']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_season, y_reg_season, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

#Algorithme regression logistique SR
logistic_model_reg = LogisticRegression()
logistic_model_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg_logistic = logistic_model_reg.predict(X_test_reg_scaled)

print("Metrics for Regular Season - Logistic Regression:")
print("-----------------------------------------------")
print("Accuracy:", accuracy_score(y_test_reg, y_pred_reg_logistic))
print("Recall:", recall_score(y_test_reg, y_pred_reg_logistic))
print("Precision:", precision_score(y_test_reg, y_pred_reg_logistic))
print("F1-score:", f1_score(y_test_reg, y_pred_reg_logistic))
print(classification_report(y_test_reg, y_pred_reg_logistic))

confusion_mat_reg = confusion_matrix(y_test_reg, y_pred_reg_logistic)

print("Matrice de confusion - Saison régulière")
print("--------------------------------------")
print(confusion_mat_reg)

#Algorithme RandomForest SR

rf_model_reg = RandomForestClassifier()
rf_model_reg.fit(X_train_reg_scaled, y_train_reg)

y_pred_reg_rf = rf_model_reg.predict(X_test_reg_scaled)

print("Metrics for Regular Season - Random Forest:")
print("-----------------------------------------")
print("Accuracy:", accuracy_score(y_test_reg, y_pred_reg_rf))
print("Recall:", recall_score(y_test_reg, y_pred_reg_rf))
print("Precision:", precision_score(y_test_reg, y_pred_reg_rf))
print("F1-score:", f1_score(y_test_reg, y_pred_reg_rf))
print(classification_report(y_test_reg, y_pred_reg_rf))

confusion_reg_rf = confusion_matrix(y_test_reg, y_pred_reg_rf)
print("Matrice de confusion - Saison régulière")
print("--------------------------------------")
print(confusion_reg_rf)

#Algorithme KNN SR

knn_model_reg = KNeighborsClassifier()
knn_model_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg_knn = knn_model_reg.predict(X_test_reg_scaled)

print("Metrics for Regular Season - KNN:")
print("--------------------------------")
print("Accuracy:", accuracy_score(y_test_reg, y_pred_reg_knn))
print("Recall:", recall_score(y_test_reg, y_pred_reg_knn))
print("Precision:", precision_score(y_test_reg, y_pred_reg_knn))
print("F1-score:", f1_score(y_test_reg, y_pred_reg_knn))
print(classification_report(y_test_reg, y_pred_reg_knn))

confusion_mat_knn = confusion_matrix(y_test_reg, y_pred_reg_knn)
print("Matrice de confusion - Saison régulière")
print("--------------------------------------")
print(confusion_mat_knn)

#Playoffs
numeric_columns1 = shot_loc_playoffs[['Shot Distance', 'X Location', 'Y Location']]
scaler = StandardScaler()
scaled_numeric_columns = scaler.fit_transform(numeric_columns1)
pca = PCA()
pca_result = pca.fit_transform(scaled_numeric_columns)
explained_variances = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variances) + 1), np.cumsum(explained_variances), marker='o')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulative')
plt.title('Courbe du coude - PCA')
plt.xticks(np.arange(1, len(explained_variances) + 1))
plt.grid(True)
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_numeric_columns)
shot_loc_playoffs['PCA Component 1'] = pca_result[:, 0]
shot_loc_playoffs['PCA Component 2'] = pca_result[:, 1]

categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']
shot_loc_playoffs_encoded = pd.get_dummies(shot_loc_playoffs, columns=categorical_variables)

X_playoffs = shot_loc_playoffs_encoded.drop(columns=['Shot Made Flag'])
y_playoffs = shot_loc_playoffs_encoded['Shot Made Flag']
X_train_playoffs, X_test_playoffs, y_train_playoffs, y_test_playoffs = train_test_split(X_playoffs, y_playoffs, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_playoffs_scaled = scaler.fit_transform(X_train_playoffs)
X_test_playoffs_scaled = scaler.transform(X_test_playoffs)

logistic_model_playoffs = LogisticRegression()
logistic_model_playoffs.fit(X_train_playoffs_scaled, y_train_playoffs)
y_pred_playoffs_logistic = logistic_model_playoffs.predict(X_test_playoffs_scaled)

print("Metrics for Playoffs - Logistic Regression:")
print("-------------------------------------------")
print("Accuracy:", accuracy_score(y_test_playoffs, y_pred_playoffs_logistic))
print("Recall:", recall_score(y_test_playoffs, y_pred_playoffs_logistic))
print("Precision:", precision_score(y_test_playoffs, y_pred_playoffs_logistic))
print("F1-score:", f1_score(y_test_playoffs, y_pred_playoffs_logistic))
print(classification_report(y_test_playoffs, y_pred_playoffs_logistic))

confusion_mat_playoffs = confusion_matrix(y_test_playoffs, y_pred_playoffs_logistic)

print("Matrice de confusion - Playoffs")
print("---------------------------------")
print(confusion_mat_playoffs)

# Algorithme RandomForest Playoffs
rf_model_playoffs = RandomForestClassifier()
rf_model_playoffs.fit(X_train_playoffs_scaled, y_train_playoffs)
y_pred_playoffs_rf = rf_model_playoffs.predict(X_test_playoffs_scaled)

print("Metrics for Playoffs - Random Forest:")
print("-------------------------------------")
print("Accuracy:", accuracy_score(y_test_playoffs, y_pred_playoffs_rf))
print("Recall:", recall_score(y_test_playoffs, y_pred_playoffs_rf))
print("Precision:", precision_score(y_test_playoffs, y_pred_playoffs_rf))
print("F1-score:", f1_score(y_test_playoffs, y_pred_playoffs_rf))
print(classification_report(y_test_playoffs, y_pred_playoffs_rf))

# Algorithme KNN Playoffs
knn_model_playoffs = KNeighborsClassifier()
knn_model_playoffs.fit(X_train_playoffs_scaled, y_train_playoffs)
y_pred_playoffs_knn = knn_model_playoffs.predict(X_test_playoffs_scaled)

print("Metrics for Playoffs - KNN:")
print("---------------------------")
print("Accuracy:", accuracy_score(y_test_playoffs, y_pred_playoffs_knn))
print("Recall:", recall_score(y_test_playoffs, y_pred_playoffs_knn))
print("Precision:", precision_score(y_test_playoffs, y_pred_playoffs_knn))
print("F1-score:", f1_score(y_test_playoffs, y_pred_playoffs_knn))
print(classification_report(y_test_playoffs, y_pred_playoffs_knn))

from sklearn.model_selection import KFold

n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

accuracy_scores = []
recall_scores = []
precision_scores = []
f1_scores = []

for train_index, test_index in kf.split(X_reg_season, y_reg_season):
   
    X_train, X_test = X_reg_season.iloc[train_index], X_reg_season.iloc[test_index]
    y_train, y_test = y_reg_season.iloc[train_index], y_reg_season.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

mean_accuracy = np.mean(accuracy_scores)
mean_recall = np.mean(recall_scores)
mean_precision = np.mean(precision_scores)
mean_f1 = np.mean(f1_scores)

print("Moyenne des scores de validation croisée (K-Fold) :")
print("Accuracy:", mean_accuracy)
print("Recall:", mean_recall)
print("Precision:", mean_precision)
print("F1-score:", mean_f1)


from sklearn.model_selection import ShuffleSplit

shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

accuracy_scores = []

for train_index, test_index in shuffle_split.split(X_playoffs):
    X_train, X_test = X_playoffs.iloc[train_index], X_playoffs.iloc[test_index]
    y_train, y_test = y_playoffs.iloc[train_index], y_playoffs.iloc[test_index]

    rf_model_playoffs.fit(X_train, y_train)
    y_pred = rf_model_playoffs.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
mean_accuracy = np.mean(accuracy_scores)
print("Moyenne des scores de validation croisée (Shuffle-Split) :")
print("Accuracy:", mean_accuracy)

