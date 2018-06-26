from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pandas as pd
import numpy as np

from msilib import Table
from tkinter import *
from tkinter import filedialog
from pandastable import Table
import pandas as pd

#pencere oluşturma
root = Tk()
root.title("Yapay Sinir Ağları Projesi")
root.geometry("1000x500")

dataset=None

frame1 = Frame(root)
frame1.pack()


#label ekleme
label = Label(frame1,text="       Cevher ile Muhammed'in sınıflandırma için yapay sinir ağları programına hoşgeldiniz..")
label.grid()


label3 = Label(frame1,text="                       ")
label3.grid(row= 3,column = 0)

#Dosya seç butonu
def btn1Tıklandı():
    root2 = Tk()
    root2.withdraw()
    global file_path
    file_path = filedialog.askopenfilename(filetypes=(("Csv Files", "*.csv"),("All files", "*.*")))

    global dataset
    dataset = pd.read_csv(file_path)
    print(dataset.shape)

    frame2 = Frame(root)
    frame2.pack(side=BOTTOM)  # asagıda cerceve olusmasını sağladık.
    pt = Table(frame2, dataframe=dataset, showstatusbar=True, showtoolbar=True)
    pt.show()

#Sonuç Göster Butonu
def btn2Tıklandı():
    root3 = Tk()
    root3.title("Model - Başarı")
    root3.geometry("1000x500")


    boy = int(entry2.get())
    global dataset
    dataset=dataset.head(boy)

    hedefNitelikIndis = int(entry1.get())
    X = dataset.iloc[:, 0:hedefNitelikIndis-1]
    y = dataset.iloc[:, hedefNitelikIndis-1:hedefNitelikIndis]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    # Tahmin yap
    frame3 = Frame(root3)
    frame3.pack()

    list = []
    list.append("Sonuçlar")
    list.append(" ")
    list.append("Accuracy Score : " + str(accuracy_score(y_true=y_test,y_pred=predictions)))

    y_test=y_test.reset_index(drop=True)
    y_test = np.asarray(y_test)
    print(type(y_test))

    for i in range(len(predictions)):
        list.append("P:" + str(predictions[i])+" T:" + str(y_test[i]))

    listbox = Listbox(frame3, width=50, height=50)
    for i in list:
        listbox.insert(END, i)
    listbox.pack(fill=BOTH, expand=0)

    root3.mainloop()

#Dosya seç butonuna tıklanma olayı
btn1 = Button(frame1, text="Dataset seç", fg="blue", command=btn1Tıklandı)
btn1.grid(row=2)

label2 = Label(frame1,text="Hedef Kolon Indis Giriniz:")
label2.grid(row=5, column=0, sticky=E, pady=1)
entry1 = Entry(frame1)
entry1.grid(row=5, column=1)



label3 = Label(frame1,text="Veriseti sınır indeksini yazınız:")
label3.grid(row=6, column=0, sticky=E, pady=1)
entry2 = Entry(frame1)
entry2.grid(row=6, column = 1)

#Sonuçları göster butonuna tıklanma olayı
btn2 = Button(frame1,text="Sonuçları Göster",fg="green" ,command=btn2Tıklandı)
btn2.grid(row=6)


root.mainloop()


