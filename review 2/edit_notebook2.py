import nbformat
import sys

nb_path = r"d:\MTECH S2\DL\FINAL SEM PROJECT\VIVA 2\mlp-for-labels.ipynb"

try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = "".join(cell.source)
            if "cm = confusion_matrix(all_y_true, all_y_pred)" in source and "all_results.append({" in source:
                # Let's replace the block with the classification report and plot
                old_block = """                    print(f"  Mean Val Acc: {mean_val_acc:.4f} | Mean Test Acc: {mean_acc:.4f}")
                    cm = confusion_matrix(all_y_true, all_y_pred)
                    print(f"  Confusion Matrix:\\n{cm}")"""
                
                new_block = """                    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
                    import matplotlib.pyplot as plt
                    
                    print(f"  Mean Val Acc: {mean_val_acc:.4f} | Mean Test Acc: {mean_acc:.4f}")
                    cm = confusion_matrix(all_y_true, all_y_pred)
                    
                    class_report = classification_report(all_y_true, all_y_pred, target_names=['Normal', 'Suspect', 'Pathologic'])
                    print(f"\\n  Classification Report:\\n{class_report}")
                    
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Suspect', 'Pathologic'])
                    fig, ax = plt.subplots(figsize=(5,5))
                    disp.plot(cmap='Blues', ax=ax)
                    ax.set_title(f"{arch_name} | {opt_name} | {activation}")
                    plt.show()"""
                
                if old_block in source:
                    cell.source = source.replace(old_block, new_block)
                else:
                    print("Block to replace not found exactly, updating differently...")
                    # Fallback replace
                    source = source.replace('print(f"  Confusion Matrix:\\n{cm}")', 
                            "from sklearn.metrics import classification_report, ConfusionMatrixDisplay\\n                    import matplotlib.pyplot as plt\\n                    class_report = classification_report(all_y_true, all_y_pred, target_names=['Normal', 'Suspect', 'Pathologic'])\\n                    print(f\"\\\\n  Classification Report:\\\\n{class_report}\")\\n                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Suspect', 'Pathologic'])\\n                    fig, ax = plt.subplots(figsize=(5,5))\\n                    disp.plot(cmap='Blues', ax=ax)\\n                    ax.set_title(f\"{arch_name} | {opt_name} | {activation}\")\\n                    plt.show()")
                    cell.source = source

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print("Notebook updated with classification report and confusion matrix plot successfully.")
except Exception as e:
    print(f"Failed to update notebook: {e}")
