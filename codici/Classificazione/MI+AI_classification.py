import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

def get_probs_pytorch(model, inputs, device):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
    return probs.cpu().numpy()

def membership_inference_attack_pytorch(model, trainloader, testloader, device, n_samples=2000):
    model.eval()
    # Estrai n_samples dal train e dal test
    X_train, y_train = [], []
    for inputs, labels in trainloader:
        X_train.append(inputs)
        y_train.append(labels)
        if len(torch.cat(y_train)) >= n_samples:
            break
    X_train = torch.cat(X_train)[:n_samples]
    y_train = torch.cat(y_train)[:n_samples]

    X_test, y_test = [], []
    for inputs, labels in testloader:
        X_test.append(inputs)
        y_test.append(labels)
        if len(torch.cat(y_test)) >= n_samples:
            break
    X_test = torch.cat(X_test)[:n_samples]
    y_test = torch.cat(y_test)[:n_samples]

    # Calcola la probabilità della classe vera
    probs_train = get_probs_pytorch(model, X_train, device)
    probs_test = get_probs_pytorch(model, X_test, device)
    scores_train = probs_train[np.arange(len(y_train)), y_train.numpy()]
    scores_test = probs_test[np.arange(len(y_test)), y_test.numpy()]

    scores = np.concatenate([scores_train, scores_test])
    labels = np.concatenate([np.ones_like(scores_train), np.zeros_like(scores_test)])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    preds = (scores >= best_thresh).astype(int)

    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print(f"Membership Inference - Accuracy: {acc:.3f}, AUC: {roc_auc:.3f}")
    print("Confusion Matrix:\n", cm)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Membership Inference ROC')
    plt.legend()
    plt.show()
    return acc, roc_auc, cm

def attribute_inference_attack_pytorch(model, dataloader, device, feature_idx=0, n_candidates=5, n_samples=300):
    model.eval()
    # Estrai n_samples dal dataloader
    X, y = [], []
    for inputs, labels in dataloader:
        X.append(inputs)
        y.append(labels)
        if len(torch.cat(y)) >= n_samples:
            break
    X = torch.cat(X)[:n_samples].cpu().numpy()
    y = torch.cat(y)[:n_samples].cpu().numpy()

    correct = 0
    for i in range(n_samples):
        x = X[i].copy()
        true_val = x.flat[feature_idx]
        candidates = np.linspace(0, 1, n_candidates)
        best_score = -np.inf
        best_val = None
        for val in candidates:
            x_cand = x.copy()
            x_cand.flat[feature_idx] = val
            x_cand_tensor = torch.from_numpy(x_cand).unsqueeze(0).to(device)
            probs = get_probs_pytorch(model, x_cand_tensor, device)[0]
            prob = probs[y[i]]
            if prob > best_score:
                best_score = prob
                best_val = val
        if np.isclose(best_val, true_val, atol=1.0/(n_candidates-1)):
            correct += 1
    acc = correct / n_samples
    print(f"Attribute Inference - Accuracy: {acc:.3f} (feature idx {feature_idx})")
    plt.bar(['Correct', 'Incorrect'], [correct, n_samples-correct])
    plt.title(f'Attribute Inference (feature idx {feature_idx})')
    plt.show()
    return acc

# Membership Inference Attack
print("MIA - Overfit")
membership_inference_attack_pytorch(model, trainloader_overfit, testloader, device)

print("MIA - No Overfit")
membership_inference_attack_pytorch(no_overfit_model, trainloader_no_overfit, testloader, device)

# Attribute Inference Attack su una sola feature (ad esempio la prima)
feature_idx = 0  # Puoi cambiare l'indice per attaccare un altro pixel
print("AIA - Overfit")
attribute_inference_attack_pytorch(model, testloader, device, feature_idx=feature_idx, n_candidates=5)

print("AIA - No Overfit")
attribute_inference_attack_pytorch(no_overfit_model, testloader, device, feature_idx=feature_idx, n_candidates=5)
