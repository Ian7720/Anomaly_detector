from features import load_dataset
from features import quantify_image
from sklearn.ensemble import IsolationForest
import numpy as np
import pickle
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 169, 361, 361))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(409, 169, 361, 361))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 140, 131, 21))
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 20, 141, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.button_clicked)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 60, 141, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.select_image)
        self.pushButton_2.clicked.connect(self.button_clicked_2)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(409, 140, 131, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(250, 20, 131, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(200, 70, 600, 21))
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSelect_Dataset = QtWidgets.QAction(MainWindow)
        self.actionSelect_Dataset.setObjectName("actionSelect_Dataset")
        self.actionSelect_Dataset.triggered.connect(self.update_dataset)
        self.menuFile.addAction(self.actionSelect_Dataset)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Input Image"))
        self.label_2.setText(_translate("MainWindow", "Result Image"))
        self.label_3.setText(_translate("MainWindow", "Input Image"))
        self.pushButton.setText(_translate("MainWindow", "Train"))
        self.pushButton_2.setText(_translate("MainWindow", "Test"))
        self.label_4.setText(_translate("MainWindow", "Result Image"))
        self.label_5.setText(_translate("MainWindow", "DataSet : "))
        self.label_6.setText(_translate("MainWindow", " "))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionSelect_Dataset.setText(_translate("MainWindow", "Select Dataset"))
        
    def update_dataset_label(self, message):
        self.label_5.setText(message)
 
    def update_status_label(self, message):
        self.label_6.setText(message)
        
    def update_dataset(self):
        # Open a file dialog to select the dataset directory
        dataset_path = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, "Select Dataset Directory")
        
        # Check if a directory was selected
        if dataset_path:
            self.dataset_path = dataset_path
            
            # Extract the directory name from the path
            directory_name = os.path.basename(self.dataset_path)
            
            # Update the label with the selected dataset directory name
            self.update_dataset_label(f"Dataset: {directory_name}")
            
            # Load and quantify the image dataset
            print("[INFO] preparing dataset...")
            dataset = load_dataset(self.dataset_path, bins=(3, 3, 3))
            
            # Assuming load_dataset returns a tuple, extract the actual data part
            if isinstance(dataset, tuple):
                self.data = dataset[0]
            else:
                self.data = dataset

            # Check the shape of the data
            print(f"[INFO] dataset shape: {self.data.shape}")

            # Ensure the data is a 2D array
            if self.data.ndim == 1:
                self.data = self.data.reshape(-1, 1)
            
            print("[INFO] dataset updated")
        else:
            print("[INFO] no directory selected")
            self.update_dataset_label("No directory selected")

    def button_clicked(self):
        # Check if the dataset has been updated
        if not hasattr(self, 'data'):
            print("[INFO] dataset not loaded. Please update the dataset first.")
            return
        
        # Set the path for the output model file
        model_path = r"D:\UMS\SEM6\ComputerVisionandApplication\Code\Assignment2\test.model"  # Update with your desired model output path

        # Train the anomaly detection model
        print("[INFO] fitting anomaly detection model...")
        model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        model.fit(self.data)

        # Serialize the anomaly detection model to disk
        with open(model_path, "wb") as f:
            f.write(pickle.dumps(model))

        message = f"[INFO] model saved to {model_path}"
        print(message)
        self.update_status_label(message)
        
    def select_image(self):
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Select Image", "", "Images (*.png *.xpm *.jpg *.bmp *.jpeg)")
        if image_path:
            self.image_path = image_path

    def button_clicked_2(self):
        # Check if the image path has been set
        if not hasattr(self, 'image_path'):
            print("[INFO] no image selected. Please select an image first.")
            return

        # Set the paths for the model and the input image
        model_path = r"D:\UMS\SEM6\ComputerVisionandApplication\Code\Assignment2\test.model"  # Update with your model path
        image_path = self.image_path

        # Load the anomaly detection model
        print("[INFO] loading anomaly detection model...")
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Load the input image, convert it to the HSV color space, and
        # quantify the image in the *same manner* as we did during training
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Resize the QImage to fit within the QLabel's dimensions
        q_pixmap = QtGui.QPixmap.fromImage(q_img)
        scaled_pixmap = q_pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio)

        # Set the scaled QPixmap to the QLabel (display the original image)
        self.label.setPixmap(scaled_pixmap)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = quantify_image(hsv, bins=(3, 3, 3))

        # Use the anomaly detector model and extracted features to determine
        # if the example image is an anomaly or not
        preds = model.predict([features])[0]
        label = "anomaly" if preds == -1 else "normal"
        color = (0, 0, 255) if preds == -1 else (0, 255, 0)

        # Draw the predicted label text on the original image
        cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert the image from BGR to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Resize the QImage to fit within the QLabel's dimensions
        q_pixmap = QtGui.QPixmap.fromImage(q_img)
        scaled_pixmap = q_pixmap.scaled(self.label_2.size(), QtCore.Qt.KeepAspectRatio)

        # Set the scaled QPixmap to the QLabel
        self.label_2.setPixmap(scaled_pixmap)

        # Set the text of self.label_2 to display the prediction
        self.label_4.setText(f"Result Image: {label}")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
