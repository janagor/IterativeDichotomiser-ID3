from ucimlrepo import fetch_ucirepo


mushroom = fetch_ucirepo(id=73)

# data
X_mush = mushroom.data.features
Y_mush = mushroom.data.targets

# metadata
print(mushroom.metadata)

# variable information
print(mushroom.variables)


breast_cancer = fetch_ucirepo(id=14)

X_bre = breast_cancer.data.features
Y_bre = breast_cancer.data.targets

print(breast_cancer.metadata)

print(breast_cancer.variables)


# usefull
# Y_mush.to_numpy() inpuy ass np.array
# X_mush.to_numpy()
# X_mush.keys() to have all the keys than I can find set of values per a key


