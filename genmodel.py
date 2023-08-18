import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
import xgboost as xgb
import cv2
# โหลดข้อมูลจาก pkl
carvectors_train = pickle.load(open('carvectors_train.pkl', 'rb'))
carvectors_test = pickle.load(open('carvectors_test.pkl', 'rb'))
# แบ่งสัดส่วนข้อมูล
X_train_data = [carvectors_train[0:8100] for  carvectors_train in carvectors_train]
X_test_data = [carvectors_test[0:8100] for carvectors_test in carvectors_test]

Y_train_data = [carvectors_train[-1] for carvectors_train in carvectors_train]
Y_test_data = [carvectors_test[-1] for carvectors_test in carvectors_test]
# Label Encoding แปลงข้อมูลจากข้อความให้เป็นตัวเลข
LE = LabelEncoder()
new_y_train= LE.fit_transform(Y_train_data)
new_y_test= LE.fit_transform(Y_test_data)

# def extract_hog_features(img):
#     s = (128,128)
#     new_img = cv2.resize(img, s, interpolation=cv2.INTER_AREA)
#     win_size =  new_img.shape
#     cell_size = (8, 8)
#     block_size = (16, 16)
#     block_stride = (8, 8)
#     num_bins = 9
#     hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
#     cell_size, num_bins)
#     hog_descriptor = hog.compute(new_img)
#     hog_descriptor_list = hog_descriptor.flatten().tolist()
#     print ('HOG Descriptor:', hog_descriptor)
    
#     return hog_descriptor
#สร้าง model Decision tree
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train_data, new_y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test_data)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(new_y_test, y_pred)
accuracy_percentage = accuracy * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")


print(confusion_matrix(new_y_test,y_pred))
write_path = "model.pkl"
pickle.dump(clf, open(write_path,"wb"))
print("data preparation is done")


# # Create a Decision Tree classifier
# clf = DecisionTreeClassifier(random_state=42)

# # Train the classifier using training data
# clf.fit(X_train_data, new_y_train)

# # Extract HOG features from test images
# X_test_hog_features = []
# for image in X_test_data:
#     hog_features = extract_hog_features(image)  # You need to implement this function
#     X_test_hog_features.append(hog_features)

# # Predict outcomes using HOG features of test data
# y_pred = clf.predict(X_test_hog_features)

# # Calculate and print model accuracy
# accuracy = accuracy_score(new_y_test, y_pred)
# accuracy_percentage = accuracy * 100
# print(f"Accuracy: {accuracy_percentage:.2f}%")

# # Display Confusion Matrix
# print(confusion_matrix(new_y_test, y_pred))

# # # Save the model using joblib
# # write_path = "model.joblib"
# # joblib.dump(clf, write_path)

# print("Data preparation is done.")


# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix, accuracy_score
# import pickle
# import cv2

# # Load data from pkl files
# carvectors_train = pickle.load(open('carvectors_train.pkl', 'rb'))
# carvectors_test = pickle.load(open('carvectors_test.pkl', 'rb'))

# # Separate data into feature portions
# X_train_data = [carvectors_train[0:8100] for carvectors_train in carvectors_train]
# X_test_data = [carvectors_test[0:8100] for carvectors_test in carvectors_test]

# Y_train_data = [carvectors_train[-1] for carvectors_train in carvectors_train]
# Y_test_data = [carvectors_test[-1] for carvectors_test in carvectors_test]

# # Label Encoding to convert text data into numerical data
# LE = LabelEncoder()
# new_y_train = LE.fit_transform(Y_train_data)
# new_y_test = LE.fit_transform(Y_test_data)

# # Function to extract HOG features from an image
# def extract_hog_features(img):
#     new_img = cv2.resize(img, None, fx=128, fy=128, interpolation=cv2.INTER_AREA)

#     # s = (128, 128)
#     # new_img = cv2.resize(img, s, interpolation=cv2.INTER_AREA)
#     win_size = new_img.shape
#     cell_size = (8, 8)
#     block_size = (16, 16)
#     block_stride = (8, 8)
#     num_bins = 9
#     hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
#     hog_descriptor = hog.compute(new_img)
#     hog_descriptor_list = hog_descriptor.flatten().tolist()
#     print('HOG Descriptor:', hog_descriptor)
    
#     return hog_descriptor

# # Create a Decision Tree classifier
# clf = DecisionTreeClassifier(random_state=42)

# # Train the classifier using training data
# clf.fit(X_train_data, new_y_train)

# # Extract HOG features from test images
# X_test_hog_features = []
# for image in X_test_data:
#     hog_features = extract_hog_features(image)  # You need to implement this function
#     X_test_hog_features.append(hog_features)

# # Predict outcomes using HOG features of test data
# y_pred = clf.predict(X_test_hog_features)

# # Calculate and print model accuracy
# accuracy = accuracy_score(new_y_test, y_pred)
# accuracy_percentage = accuracy * 100
# print(f"Accuracy: {accuracy_percentage:.2f}%")

# # Display Confusion Matrix
# print(confusion_matrix(new_y_test, y_pred))

# print("Data preparation is done.")
