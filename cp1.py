import tkinter as tk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

data = None

def load_dataset():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Dataset Loaded Successfully")


# SUPERVISED WINDOW

def supervised_window():
    sup = tk.Toplevel(root)
    sup.title("Supervised Learning")
    sup.geometry("480x500")
    sup.configure(bg="white")

    algo = tk.StringVar(value="Logistic Regression")

    result_label = tk.Label(
        sup, text="", bg="white", fg="green", font=("Arial", 11)
    )
    result_label.pack(pady=10)

    def run_supervised():
        if data is None:
            messagebox.showerror("Error", "Please load dataset")
            return

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # -------- Algorithm Selection --------
        if algo.get() == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        elif algo.get() == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)

        elif algo.get() == "SVM":
            model = SVC(kernel="rbf")

        elif algo.get() == "Decision Tree":
            model = DecisionTreeClassifier()

        elif algo.get() == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)

        elif algo.get() == "Naive Bayes":
            model = GaussianNB()

        else:
            messagebox.showerror("Error", "Select an algorithm")
            return

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        result_label.config(text=f"Accuracy: {acc:.2f}")

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{algo.get()} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # ---------- UI ----------o
    tk.Label(
        sup, text="SUPERVISED LEARNING",
        bg="black", fg="white",
        font=("Arial", 14, "bold")
    ).pack(pady=15)

    tk.Button(
        sup, text="Load Dataset",
        command=load_dataset,
        bg="gray20", fg="white",
        width=25
    ).pack(pady=10)

    tk.OptionMenu(
        sup, algo,
        "Logistic Regression",
        "KNN",
        "SVM",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
        "Gradient Boosting"
    ).pack(pady=10)

    tk.Button(
        sup, text="Run Algorithm",
        command=run_supervised,
        bg="green", fg="black",
        width=25
    ).pack(pady=15)

# UNSUPERVISED WINDOW

def unsupervised_window():
    unsup = tk.Toplevel(root)
    unsup.title("Unsupervised Learning")
    unsup.geometry("480x450")
    unsup.configure(bg="white")

    algo = tk.StringVar(value="K-Means")

    result_label = tk.Label(
        unsup, text="", bg="white", fg="blue", font=("Arial", 11)
    )
    result_label.pack(pady=10)

    def run_unsupervised():
        if data is None:
            messagebox.showerror("Error", "Please load dataset")
            return

        X = data.select_dtypes(include=['int64', 'float64'])

        if X.shape[1] < 2:
            messagebox.showerror("Error", "Dataset must have at least 2 numeric columns")
            return

        #fit Alogrithm
        if algo.get() == "K-Means":
            model = KMeans(n_clusters=3, random_state=42)
            clusters = model.fit_predict(X)

        elif algo.get() == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
            clusters = model.fit_predict(X)

        elif algo.get() == "Hierarchical Clustering":
            model = AgglomerativeClustering(n_clusters=3)
            clusters = model.fit_predict(X)

        elif algo.get() == "PCA (Visualization Only)":
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap="viridis")
            plt.title("PCA Visualization")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.show()

            result_label.config(text="PCA Visualization Completed")
            return

        else:
            messagebox.showerror("Error", "Select an algorithm")
            return

        data["Cluster"] = clusters
        result_label.config(text=f"{algo.get()} Completed")

        #PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters,
            cmap="viridis"
        )
        plt.title(f"{algo.get()} Result (PCA View)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()

   # GUI 
    tk.Label(
        unsup, text="UNSUPERVISED LEARNING",
        bg="black", fg="white",
        font=("Arial", 14, "bold")
    ).pack(pady=15)

    tk.Button(
        unsup, text="Load Dataset",
        command=load_dataset,
        bg="gray20", fg="white",
        width=25
    ).pack(pady=10)

    tk.OptionMenu(
        unsup, algo,
        "K-Means",
        "DBSCAN",
        "Hierarchical Clustering",
        "PCA (Visualization Only)"
    ).pack(pady=10)

    tk.Button(
        unsup, text="Run Algorithm",
        command=run_unsupervised,
        bg="blue", fg="white",
        width=25
    ).pack(pady=15)


# MAIN WINDOW
root = tk.Tk()
root.title("Smart System for Predictive Data Models")
root.geometry("800x600")
root.configure(bg="black")

tk.Label(
    root, text="MACHINE LEARNING SYSTEM",
    bg="black", fg="white",
    font=("Arial", 16, "bold")
).pack(pady=25)

tk.Button(
    root, text="Supervised Learning",
    command=supervised_window,
    bg="green", fg="black",
    width=30
).pack(pady=10)

tk.Button(
    root, text="Unsupervised Learning",
    command=unsupervised_window,
    bg="blue", fg="white",
    width=30
).pack(pady=10)

tk.Label(
    root, text="Created by Naveen",
    bg="black", fg="gray"
).pack(side="bottom", pady=10)

root.mainloop()
