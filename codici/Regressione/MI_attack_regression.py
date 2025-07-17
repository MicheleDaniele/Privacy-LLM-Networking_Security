# Aggiunta delle librerie necessarie per l'attacco
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# =============================================
# Membership Inference Attack (MIA) - FUNZIONI
# =============================================

def compute_mia(model, X_train, X_test, y_train, y_test):
    """
    Esegue un Membership Inference Attack basato sulla loss individuale.
    Restituisce AUC e dati per le curve ROC.
    """
    # Calcolo delle predizioni
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()

    # Calcolo della loss per ogni campione (errore quadratico)
    loss_train = (y_train - y_pred_train) ** 2
    loss_test = (y_test - y_pred_test) ** 2

    # Creazione del dataset per l'attacco
    df_mia = pd.DataFrame({
        'loss': np.concatenate([loss_train, loss_test]),
        'member': np.concatenate([np.ones_like(loss_train), np.zeros_like(loss_test)])
    })

    # Split in train/test per il modello di attacco
    X_attack = df_mia[['loss']]
    y_attack = df_mia['member']
    X_train_attack, X_test_attack, y_train_attack, y_test_attack = train_test_split(
        X_attack, y_attack, test_size=0.3, random_state=42
    )

    # Addestramento del classificatore (Regressore Logistico)
    clf = LogisticRegression()
    clf.fit(X_train_attack, y_train_attack)

    # Predizione delle probabilità
    y_proba = clf.predict_proba(X_test_attack)[:, 1]

    # Calcolo AUC e curva ROC
    auc = roc_auc_score(y_test_attack, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test_attack, y_proba)

    return auc, fpr, tpr, loss_train, loss_test

# =============================================
# ESECUZIONE DELL'ATTACCO SUI DUE MODELLI
# =============================================

# Attacco sul modello in overfitting (rete neurale)
auc_nn, fpr_nn, tpr_nn, loss_train_nn, loss_test_nn = compute_mia(
    model_overfit, X_train_scaled, X_test_scaled, y_train.values, y_test.values
)

# Attacco sul modello Ridge (no overfitting)
auc_ridge, fpr_ridge, tpr_ridge, loss_train_ridge, loss_test_ridge = compute_mia(
    ridge, X_train_scaled, X_test_scaled, y_train.values, y_test.values
)

# =============================================
# VISUALIZZAZIONE RISULTATI
# =============================================

plt.figure(figsize=(15, 6))

# Curva ROC
plt.subplot(1, 2, 1)
plt.plot(fpr_nn, tpr_nn, label=f'Rete Neurale (AUC = {auc_nn:.2f})')
plt.plot(fpr_ridge, tpr_ridge, label=f'Ridge (AUC = {auc_ridge:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Membership Inference Attack')
plt.legend()

# Distribuzione delle loss
plt.subplot(1, 2, 2)
plt.hist(loss_train_nn, bins=50, alpha=0.5, label='Train (NN)')
plt.hist(loss_test_nn, bins=50, alpha=0.5, label='Test (NN)')
plt.hist(loss_train_ridge, bins=50, alpha=0.5, label='Train (Ridge)')
plt.hist(loss_test_ridge, bins=50, alpha=0.5, label='Test (Ridge)')
plt.yscale('log')
plt.xlabel('Loss (MSE individuale)')
plt.ylabel('Frequenza (log scale)')
plt.title('Distribuzione delle Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Risultati numerici
print("\n=== Risultati Membership Inference Attack ===")
print(f"Modello Overfitting (Rete Neurale):")
print(f"- AUC: {auc_nn:.2f}")
print(f"- Gap medio loss (Train-Test): {np.mean(loss_train_nn) - np.mean(loss_test_nn):.2f}")

print(f"\nModello Ridge (No Overfitting):")
print(f"- AUC: {auc_ridge:.2f}")
print(f"- Gap medio loss (Train-Test): {np.mean(loss_train_ridge) - np.mean(loss_test_ridge):.2f}")
