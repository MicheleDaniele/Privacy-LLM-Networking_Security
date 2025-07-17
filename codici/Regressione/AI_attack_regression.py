# ============================
#  ATTACK SULL'ATTRIBUTO GENDER
# ============================

# 1. Import delle librerie specifiche per la classificazione
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 2. Preparazione dati per l'attacco
sensitive_attr = 'Gender'

# Ricostruzione X completo (con Gender incluso) e y
X_full = df.drop(columns=[target_col])
y_full = df[target_col]

# Rimuove i valori mancanti
X_full = pd.get_dummies(X_full, drop_first=True)
X_full = X_full.fillna(X_full.mean(numeric_only=True))
y_full = y_full.fillna(y_full.mean())

# Recupera la colonna Gender originale
sensitive = df[sensitive_attr].fillna(df[sensitive_attr].mode().iloc[0])

# Divisione coerente in train/test
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X_full, y_full, sensitive,
    test_size=0.3,
    random_state=42,
    stratify=sensitive
)

# Standardizzazione coerente con i modelli già addestrati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 3. Funzione di Attribute Inference Attack per classificazione
def attribute_inference_attack_gender(target_model, X_tr, X_te, y_sens_tr, y_sens_te):
    # 3.1. Predizioni del modello target
    pred_tr = target_model.predict(X_tr).flatten()
    pred_te = target_model.predict(X_te).flatten()

    # 3.2. Costruzione delle feature di attacco
    attack_tr = np.column_stack([X_tr, pred_tr])
    attack_te = np.column_stack([X_te, pred_te])

    # 3.3. Codifica dell'attributo sensibile (es. Gender)
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_sens_tr)
    y_te_enc = le.transform(y_sens_te)

    # 3.4. Addestramento attaccante
    attacker = LogisticRegression(max_iter=1000)
    attacker.fit(attack_tr, y_tr_enc)

    # 3.5. Valutazione
    y_tr_pred = attacker.predict(attack_tr)
    y_te_pred = attacker.predict(attack_te)

    acc_tr = accuracy_score(y_tr_enc, y_tr_pred)
    acc_te = accuracy_score(y_te_enc, y_te_pred)
    prec   = precision_score(y_te_enc, y_te_pred)
    rec    = recall_score(y_te_enc, y_te_pred)

    return {
        'acc_train': acc_tr,
        'acc_test':  acc_te,
        'precision': prec,
        'recall':    rec,
        'y_true':    y_sens_te.values,
        'y_pred':    le.inverse_transform(y_te_pred)
    }

# 4. Esecuzione dell’attacco sui due modelli
res_nn    = attribute_inference_attack_gender(model_overfit, X_train_scaled, X_test_scaled, s_train, s_test)
res_ridge = attribute_inference_attack_gender(ridge, X_train_scaled, X_test_scaled, s_train, s_test)

# 5. Stampa dei risultati
print("=== Attribute Inference Attack su Gender ===\n")

# Calcolo accuratezze medie
acc_nn_avg    = (res_nn['acc_train'] + res_nn['acc_test']) / 2
acc_ridge_avg = (res_ridge['acc_train'] + res_ridge['acc_test']) / 2

print(">> Rete Neurale (Overfitting)")
print(f"- Accuracy Media: {acc_nn_avg:.2f}")
# print(f"- Precision (Test): {res_nn['precision']:.2f}")
# print(f"- Recall    (Test): {res_nn['recall']:.2f}\n")

print(">> Ridge Regression (No Overfitting)")
print(f"- Accuracy Media: {acc_ridge_avg:.2f}")
# print(f"- Precision (Test): {res_ridge['precision']:.2f}")
# print(f"- Recall    (Test): {res_ridge['recall']:.2f}\n")

# Confusion matrix per il modello su test set
cm = confusion_matrix(res_nn['y_true'], res_nn['y_pred'], labels=['female','male'])
print("Confusion Matrix (NN - Test):")
print(pd.DataFrame(cm, index=['Vero Female','Vero Male'], columns=['Pred Female','Pred Male']))
