
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

#Pre-Process, choix des 20 joueurs pour boxscore et games by season

score=pd.read_csv('boxscore.csv', sep=',')
score.isna().sum()
score.head()
games_season=pd.read_csv('games.csv',sep=',')
games_season.isna().sum()
games_season.info()


top_players = ['LeBron James', 'Kobe Bryant', 'Tim Duncan', "Shaquille O'Neal", 'Stephen Curry', 'Kevin Durant', 'Dwyane Wade', 'Giannis Antetokounmpo', 'Kevin Garnett', 'Dirk Nowitzki','Kawhi Leonard', 'Allen Iverson', 'Steve Nash', 'Tony Parker', 'Damian Lillard','Paul Pierce', 'Jason Kidd', 'Russell Westbrook', 'Ray Allen', 'James Harden']

# Afficher les noms des joueurs sélectionnés présents dans le DataFrame score
for player_name in top_players:
    player_row = score[score['playerName'] == player_name]
    print(player_row['playerName'].iloc[0])

score_top_players = score[score['playerName'].isin(top_players)]
score_top_players.head(20)
score_top_players.info()

#Création de final_df et gestion des variables
final_df = pd.merge(score_top_players, games_season[['game_id', 'seasonStartYear']], on='game_id', how='left')
print(final_df)
final_df.info()
final_df['FG'] = pd.to_numeric(final_df['FG'], errors='coerce')
final_df['FGA'] = pd.to_numeric(final_df['FGA'], errors='coerce')
final_df['3P'] = pd.to_numeric(final_df['3P'], errors='coerce')
final_df['3PA'] = pd.to_numeric(final_df['3PA'], errors='coerce')
final_df['FT'] = pd.to_numeric(final_df['FT'], errors='coerce')
final_df['FTA'] = pd.to_numeric(final_df['FTA'], errors='coerce')
cols_to_convert = ['ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS','MP', '+/-']
final_df[cols_to_convert] = final_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
#Création des variables 2 points et des pourcentages 3P et FT%
final_df['FG%'] = (final_df['FG'] / final_df['FGA']).round(2)
final_df['3P%'] = (final_df['3P'] / final_df['3PA']).round(2)
final_df['FT%'] = (final_df['FT'] / final_df['FTA']).round(2)
final_df['2PFG'] = final_df['FG'] - final_df['3P']
final_df['2PFA'] = final_df['FGA'] - final_df['3PA']
final_df['2P%'] = (final_df['2PFG'] / final_df['2PFA']) * 100
final_df['2P%'] = final_df['2P%'].round(2)

final_df.rename(columns={'playerName': 'Player','teamName':'Team','game_id':'GameId','isStarter': 'Starter', 'seasonStartYear': 'Year'}, inplace=True)
final_df=final_df.drop(['ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF'], axis=1)
final_df.set_index('GameId', inplace=True)
final_df = final_df[['Team','Year','Player','Starter','MP','FG','FGA','FG%','2PFG','2PFA','2P%','3P','3PA','3P%','FT','FTA','FT%','PTS','+/-']]
final_df.info()
final_df.head()
final_df.isna().sum()
final_df.drop('MP',axis=1, inplace=True)
final_df = final_df[(final_df['Year'] >= 2000) & (final_df['Year'] <= 2020)]


# Supprimer les lignes où tous les joueurs n'ont pas joué (c'est-à-dire que toutes les variables statistiques sont NaN pour un GameId donnée)
final_df.dropna(subset=['FG', 'FGA', 'FG%', '2PFG', '2PFA', '2P%', '3P', '3PA', 'FT', 'FTA', 'FT%', 'PTS', '+/-'], how='all', inplace=True)

# Réinitialiser l'index après la suppression des lignes
final_df.reset_index(drop=True, inplace=True)

#Certains nans conservés car pertinents pour l'analyse, nan= 0 shoot pris de cette zone 
final_df.isna().sum()
print(final_df)
final_df.info()
final_df[final_df['Player'] == "Allen Iverson"]

#Visualisation des fréquences de tirs par joueurs et par saison  
plt.figure(figsize=(15, 20)) 

for i, player_name in enumerate(top_players):
    player_data = final_df[final_df['Player'] == player_name]
    player_data.reset_index(drop=True, inplace=True)
    
    plt.subplot(5, 4, i+1)  
    sns.lineplot(x='Year', y='FGA', data=player_data, marker='o', label='Tirs pris')  
    plt.xlabel('Saison')
    plt.ylabel('Tirs pris (FGA)')
    plt.title(player_name)
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

#Visualisation de l'efficacité de tirs par joueurs et par saison 
plt.figure(figsize=(15, 20))  

for i, player_name in enumerate(top_players):
    player_data = final_df[final_df['Player'] == player_name]
    
    player_data.reset_index(drop=True, inplace=True)
    
    plt.subplot(5, 4, i+1)  
    sns.lineplot(x='Year', y='FG', data=player_data, marker='o', label='Tirs réussis')  
    plt.ylabel('Tirs réussis (FG)')
    plt.title(player_name)
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 20)) 
for i, player_name in enumerate(top_players):
    player_data = final_df[final_df['Player'] == player_name]
    
    player_data.reset_index(drop=True, inplace=True)
    plt.subplot(5, 4, i+1) 
    sns.lineplot(x='Year', y='PTS', data=player_data, marker='o', label='Moyenne de Points')
    sns.lineplot(x='Year', y='3PA', data=player_data, marker='o', label='Tirs tentés à 3PTS')
    
    plt.xlabel('Saison')
    plt.ylabel('Nombre de tirs')
    plt.title(player_name)
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 20)) 
for i, player_name in enumerate(top_players):
    player_data = final_df[final_df['Player'] == player_name]
    
    player_data.reset_index(drop=True, inplace=True)
    plt.figure()  # Créer une nouvelle figure pour chaque joueur
    
    sns.lineplot(x='Year', y='PTS', data=player_data, marker='o', label='Moyenne de Points')
    sns.lineplot(x='Year', y='3PA', data=player_data, marker='o', label='Tirs tentés à 3PTS')
    
    plt.xlabel('Saison')
    plt.ylabel('Nombre de tirs')
    plt.title(player_name)
    plt.legend(loc='upper left')
    plt.tight_layout()

plt.show()


plt.figure(figsize=(15, 20)) 
for i, player_name in enumerate(top_players):
    player_data = final_df[final_df['Player'] == player_name]
    
    player_data.reset_index(drop=True, inplace=True)
    plt.figure() 
    
    sns.lineplot(x='Year', y='PTS', data=player_data, marker='o', label='Moyenne de Points')
    sns.lineplot(x='Year', y='3PA', data=player_data, marker='o', label='Tirs tentés à 3PTS')
    sns.lineplot(x='Year', y='2PFA', data=player_data, marker='o', label='Tirs tentés à 2PTS')
    sns.lineplot(x='Year', y='FTA', data=player_data, marker='o', label='Tirs tentés aux lancers francs')
    plt.xlabel('Saison')
    plt.ylabel('Nombre de tirs')
    plt.title(player_name)
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()



plt.figure()
plt.figure(figsize=(12, 8))
sns.boxplot( y='Player', x ='FG%', data= final_df)
plt.axvline(final_df['FG%'].mean(), color='red', linestyle='--', label='Moyenne générale')
plt.xlabel('Moyenne de pourcentage aux tirs par joueur sur toutes les saisons')
plt.ylabel('Joueur')
plt.title('Comparaison des moyennes de pourcentage aux tirs pour tous les joueurs')
plt.show()


    
plt.figure(figsize=(15, 20)) 
player_data.reset_index(drop=True, inplace=True)
plt.subplot(5, 4, i+1) 
sns.lineplot(x='Year', y='3PA', data=player_data, marker='o', label='Tirs à 3PTS')
sns.lineplot(x='Year', y='3P', data=player_data, marker='o', label='Tirs réussis à 3PTS')
    
plt.xlabel('Saison')
plt.ylabel('Nombre de tirs')
plt.title(player_name)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

#Visualisation de distribution 
numeric_columns = final_df.select_dtypes(include='number')
# Tracer l'histogramme multiple
numeric_columns.hist(bins=20, figsize=(12, 8))
plt.suptitle("Histogrammes des variables continues", fontsize=16)
plt.show()

# Moyenne de points par top players
plt.figure(figsize=(12, 8))
sns.boxplot( y='Player', x ='PTS', data= final_df)
plt.axvline(final_df['PTS'].mean(), color='red', linestyle='--', label='Moyenne générale')
plt.xlabel('Moyenne de points par joueur sur toutes les saisons')
plt.ylabel('Joueur')
plt.title('Comparaison des moyennes de points pour top players')
plt.show()

#Test 

from scipy.stats import kstest

for column in numeric_columns.columns:
    # Extraire les données de la colonne pour le test
    data = numeric_columns[column].dropna()
    
    # Effectuer le test de Kolmogorov-Smirnov
    statistic, p_value = kstest(data, 'norm')
    
    # Afficher les résultats du test
    print(f"Variable : {column}")
    print(f"Statistique du test : {statistic}")
    print(f"P-valeur : {p_value}")
    
    # Interprétation des résultats
    significance_level = 0.05
    if p_value > significance_level:
        print("La variable suit une distribution normale.")
    else:
        print("La variable ne suit pas une distribution normale.")
    print("")


from sklearn.preprocessing import StandardScaler

# Classe StandardScaler
scaler = StandardScaler()

# Appliquer la standardisation aux données numériques
scaled_data = scaler.fit_transform(numeric_columns)

# Remplacer les données numériques dans final_df par les données standardisées
final_df[numeric_columns.columns] = scaled_data

# Afficher les premières lignes du DataFrame final_df
print(final_df.head())

#Calcul Matrice de corrélation Pearson 
correlation_matrix = numeric_columns.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation de Pearson')
plt.show()

#Calcul coeff de corrélation Pearson 
from scipy.stats import pearsonr

# Boucler sur toutes les paires de variables numériques pour calculer la corrélation de Pearson et la valeur p
for var1 in numeric_columns.columns:
    for var2 in numeric_columns.columns:
        if var1 != var2:
            # Extraire les données pour les deux variables et éliminer les valeurs manquantes
            data1 = final_df[var1].dropna()
            data2 = final_df[var2].dropna()
            
            # Vérifier si les deux variables ont des données valides pour effectuer le test
            if len(data1) == len(data2):
                # Calculer la corrélation de Pearson et la valeur p
                correlation_coefficient, p_value = pearsonr(data1, data2)
                
                # Afficher les résultats pour chaque paire de variables
                print(f"Corrélation entre {var1} et {var2} : {correlation_coefficient:.2f}, p-valeur : {p_value:.5f}")

#Algorithme 1

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.impute import SimpleImputer

features = ['FG', '2PFA', '3PA', '2PFG', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%']

X = final_df[features]
y = final_df['PTS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

reg_model = LinearRegression()
reg_model.fit(X_train_imputed, y_train)
y_pred = reg_model.predict(X_test_imputed)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
print("EVS:", evs)
print("MAE:", mae)

# Algorithme 2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.impute import SimpleImputer

features = ['FG', '2PFA', '3PA', '2PFG', '3P', '3PA', 'FT', 'FTA']

X = final_df[features]
y = final_df['PTS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_imputed, y_train)
y_pred = rf_model.predict(X_test_imputed)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
print("EVS:", evs)
print("MAE:", mae)

#Algorithme 3 KNN

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

features = ['FG', '2PFA', '3PA', '2PFG', '2PFA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%']
X = final_df[features]
y = final_df['PTS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

knn = KNeighborsRegressor(n_neighbors=5) 
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
print("EVS:", evs)
print("MAE:", mae)













 