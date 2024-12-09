import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
from ttkbootstrap import Window, Label, Combobox, Entry, Button, messagebox, Treeview

# Load the dataset
file_path = "path_to_your_file/business_class.csv"
business_data = pd.read_csv(file_path)

# Convert the date column to datetime format
business_data['date'] = pd.to_datetime(business_data['date'], format='%d/%m/%Y')

# Drop columns that are not needed or might cause issues
business_data.drop(['date', 'ch_code', 'num_code', 'dep_time', 'arr_time', 'time_taken', 'stop'], axis=1, inplace=True)

# Prepare data for model
def prepare_data(data):
    data['month'] = data['month'].str.strip()
    data['airline'] = data['airline'].str.strip()
    data['from'] = data['from'].str.strip()
    data['to'] = data['to'].str.strip()
    data['price'] = data['price'].str.replace(',', '').astype(float)
    data = pd.get_dummies(data, columns=['month', 'airline', 'from', 'to'])
    return data

business_data = prepare_data(business_data)

X_business = business_data.drop('price', axis=1)
y_business = business_data['price']

# Train models
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, r2

business_model, business_r2 = train_model(X_business, y_business)

# Save the trained models
joblib.dump(business_model, "business_class_model.joblib")

def predict_price():
    try:
        class_type = class_var.get()
        month = month_var.get()
        airline = airline_var.get()
        from_location = from_var.get()
        to_location = to_var.get()
        
        if class_type == 'Business':
            model = business_model
            data = X_business
        
        input_data = pd.DataFrame(columns=data.columns)
        input_data.loc[0] = 0
        
        for col in input_data.columns:
            if col.startswith('month_') and col.endswith(month):
                input_data[col] = 1
            elif col.startswith('airline_') and col.endswith(airline):
                input_data[col] = 1
            elif col.startswith('from_') and col.endswith(from_location):
                input_data[col] = 1
            elif col.startswith('to_') and col.endswith(to_location):
                input_data[col] = 1
        
        price = model.predict(input_data)[0]
        result_label.config(text=f"Predicted Price: ₹{price:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_metrics():
    messagebox.showinfo("Model Metrics", f"Business Class R2: {business_r2:.2f}")

def show_data(class_type):
    data_window = Toplevel(root)
    data_window.title(f"{class_type} Class Dataset")
    
    data = business_data
    
    tree = Treeview(data_window, columns=list(data.columns), show='headings', height=10)
    tree.pack(padx=20, pady=20)

    for col in tree["columns"]:
        tree.heading(col, text=col.replace('_', ' ').title())

    for index, row in data.iterrows():
        tree.insert("", "end", values=list(row))

# Tkinter GUI setup
root = Window(themename="solar")
root.title("Flight Price Prediction")
root.attributes('-fullscreen', True)

title_label = Label(root, text="Flight Price Prediction", font=("Arial", 90), bootstyle="info", foreground="white", background="#001f3f")
title_label.pack(pady=20)

class_label = Label(root, text="Enter Class", bootstyle="info", foreground="white", background="#001f3f")
class_label.pack(pady=10)

class_var = StringVar()
class_entry = Combobox(root, textvariable=class_var, values=["Business", "Economy"], width=30)
class_entry.pack(anchor='center', pady=5)

month_label = Label(root, text="Enter Month", bootstyle="info", foreground="white", background="#001f3f")
month_label.pack(pady=10)

month_var = StringVar()
month_entry = Entry(root, textvariable=month_var, width=30)
month_entry.pack(anchor='center', pady=5)

airline_label = Label(root, text="Enter Airline", bootstyle="info", foreground="white", background="#001f3f")
airline_label.pack(pady=10)

airline_var = StringVar()
airline_entry = Entry(root, textvariable=airline_var, width=30)
airline_entry.pack(anchor='center', pady=5)

from_label = Label(root, text="Enter From Location", bootstyle="info", foreground="white", background="#001f3f")
from_label.pack(pady=10)

from_var = StringVar()
from_entry = Entry(root, textvariable=from_var, width=30)
from_entry.pack(anchor='center', pady=5)

to_label = Label(root, text="Enter To Location", bootstyle="info", foreground="white", background="#001f3f")
to_label.pack(pady=10)

to_var = StringVar()
to_entry = Entry(root, textvariable=to_var, width=30)
to_entry.pack(anchor='center', pady=5)

result_label = Label(root, text="Predicted Price", font=("Arial", 14), bootstyle="info", foreground="white", background="#001f3f")
result_label.pack(pady=10)

predict_button = Button(root, text="Predict Price", width=20, command=predict_price, bootstyle="success")
predict_button.pack(pady=20)

metrics_button = Button(root, text="Show Metrics", width=20, command=show_metrics, bootstyle="primary")
metrics_button.pack(pady=10)

business_data_button = Button(root, text="Show Business Data", width=20, command=lambda: show_data('Business'), bootstyle="warning")
business_data_button.pack(pady=10)

root.bind("<Escape>", lambda e: root.attributes('-fullscreen', False))
root.bind("<F11>", lambda e: root.attributes('-fullscreen', not root.attributes('-fullscreen')))

root.mainloop()
