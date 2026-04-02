import nbformat
import sys

nb_path = r"d:\MTECH S2\DL\FINAL SEM PROJECT\VIVA 2\mlp-for-labels.ipynb"

try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = "".join(cell.source)
            if "X_train, X_test" in source and "train_test_split" in source:
                cell.source = "from sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import confusion_matrix\nn_features = X.shape[1]\nprint(f\"Number of features: {n_features}\")"
            elif "class_weights_arr = compute_class_weight" in source:
                cell.source = "# Class weights will be computed dynamically within each fold in the training loop."
            elif "callbacks = [" in source and "all_results = []" in source:
                new_source = """callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=0)
]

all_results = []
best_val_acc = 0
best_config  = None
best_model   = None

total_combinations = len(architectures) * len(optimizers) * len(activations) * len(dropout_rates) * len(batch_sizes)
count = 0

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for arch_name, layers in architectures.items():
    for opt_name, optimizer in optimizers.items():
        for activation in activations:
            for dropout in dropout_rates:
                for batch_size in batch_sizes:
                    count += 1
                    config = f"{arch_name} | {opt_name} | {activation} | drop={dropout} | bs={batch_size}"
                    print(f"\\n[{count}/{total_combinations}] {config}")
                    
                    fold_accs = []
                    fold_val_accs = []
                    all_y_true = []
                    all_y_pred = []
                    
                    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        scaler = StandardScaler()
                        X_train_sc = scaler.fit_transform(X_train)
                        X_test_sc = scaler.transform(X_test)
                        
                        y_train_cat = to_categorical(y_train, num_classes=3)
                        y_test_cat = to_categorical(y_test, num_classes=3)
                        
                        class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                        class_weights = dict(enumerate(class_weights_arr))
                        
                        if 'adam_1e-3' in opt_name: opt = Adam(learning_rate=1e-3)
                        elif 'adam_1e-4' in opt_name: opt = Adam(learning_rate=1e-4)
                        elif 'rmsprop' in opt_name: opt = RMSprop(learning_rate=1e-3)
                        else: opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
                        
                        model = build_model(layers, activation, dropout, opt)
                        history = model.fit(
                            X_train_sc, y_train_cat,
                            validation_split=0.15,
                            epochs=150,
                            batch_size=batch_size,
                            class_weight=class_weights,
                            callbacks=callbacks,
                            verbose=0
                        )
                        
                        loss, acc = model.evaluate(X_test_sc, y_test_cat, verbose=0)
                        val_acc = max(history.history['val_accuracy'])
                        
                        fold_accs.append(acc)
                        fold_val_accs.append(val_acc)
                        
                        # Generate predictions for this fold
                        y_pred = model.predict(X_test_sc, verbose=0)
                        all_y_true.extend(y_test)
                        all_y_pred.extend(np.argmax(y_pred, axis=1))
                        
                    mean_acc = np.mean(fold_accs)
                    mean_val_acc = np.mean(fold_val_accs)
                    
                    print(f"  Mean Val Acc: {mean_val_acc:.4f} | Mean Test Acc: {mean_acc:.4f}")
                    cm = confusion_matrix(all_y_true, all_y_pred)
                    print(f"  Confusion Matrix:\\n{cm}")
                    
                    all_results.append({
                        'config':     config,
                        'arch':       arch_name,
                        'optimizer':  opt_name,
                        'activation': activation,
                        'dropout':    dropout,
                        'batch_size': batch_size,
                        'n_layers':   len(layers),
                        'val_acc':    mean_val_acc,
                        'test_acc':   mean_acc,
                    })

                    if mean_val_acc > best_val_acc:
                        best_val_acc = mean_val_acc
                        best_config  = config
                        best_model   = model
                        print(f"  ★ New best model configuration!")"""
                cell.source = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print("Notebook updated successfully.")
except Exception as e:
    print(f"Failed to update notebook: {e}")
