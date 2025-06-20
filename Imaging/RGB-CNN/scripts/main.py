# Import important libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import random
import time
import keras
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import  metrics

# Import CNN-related libraries 
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


#import functions
from gradcam_file import GradCAM

class0_name='class0'
class1_name='class1'
##Pixel Size
#Define pixel size for rows and columns

pixel_size=300

def read_and_process_image(list_of_images):
    X = []
    for img in list_of_images:
        image = cv2.imread(img)
        X.append(cv2.resize(image, (pixel_size, pixel_size), interpolation=cv2.INTER_CUBIC))

    # Define the output of each image to a different list
    y = [0 if class0_name in addr else 1 for addr in list_of_images]

    return X, y

#Image Directory: The directory of the folder where images are stored.
data_dir = 'all_images/'

image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith(('.tif', '.tiff', '.jpeg', '.png'))]
random.shuffle(image_files)
X, y = read_and_process_image(image_files)
X=np.array(X)
y=np.array(y)
 
print(X.shape)
 
channels=3

#The following values can be changed

# Batch size (should be a factor of 2.***4,8,16,32,64...***)
batch_size = 16

# Dropout rate
drop_rate = 0.1
 
# Number of epochs
num_epochs = 300


 
# save the weights of only the best model
filepath="med.weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('plots/accuracy_loss'):
    os.makedirs('plots/accuracy_loss')
if not os.path.exists('plots/roc'):
    os.makedirs('plots/roc')
    
#Define number of classes
num_classes=2
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
nval = len(X_val)
ntest = len(X_test)

#Define architecture with number of nodes and layers for Convolutional Layers
model = Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu',input_shape=(pixel_size,pixel_size,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))

#Add extra convolutional layer
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(drop_rate))
model.add(layers.Flatten())
model.add(layers.Dropout(drop_rate))

#Define architecture with number of nodes and layers for Dense Layers
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(64,activation='relu'))

#Add extra fully connected layer
# model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile the model (Different optimizer can be selected)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Different image data augmentation techniques can be selected
train_datagen = ImageDataGenerator(rescale=1./255, #Scale the image between 0 and 1
                                # rotation_range=90,
                                width_shift_range=0.1,
                                height_shift_range=0.1)
                                # zoom_range=0.1)
                                
                                
                                
# Set the validation dataset but not augmenting it (only rescaling it)
val_datagen = ImageDataGenerator(rescale=1./255)

# Set the test dataset but not augmenting it (only rescaling it)
test_datagen = ImageDataGenerator(rescale=1./255)

#Early stopping technique is applied to stop the training, when for '15' consecutive epochs the val loss has difference.
early_stopping = EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4, restore_best_weights = True)
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)

test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)
# Model training
start = time.time()
history = model.fit(train_generator,
                                epochs=num_epochs,
                                validation_data = val_generator,
                                validation_steps = ntest // batch_size,
                                callbacks=[early_stopping])
end = time.time()
print('TIME = ', end-start, "sec.")

#For each fold the trained model is saved to the folder
model.save(f"model_rgb.keras")

# Model evaluation
val_generator.reset() # reset the generator to force it start from the begining
eval_gen = model.evaluate(val_generator, steps = nval // batch_size, workers=1, use_multiprocessing=False)
print("*****************************")
print("Evaluation accuracy and loss")
print(" accuracy =", eval_gen[1] * 100)
print(" loss =", eval_gen[0])
print("*****************************")
test_generator.reset() # reset the generator to force it start from the begining
pred_gen = model.evaluate(test_generator, steps = ntest // batch_size, workers=1, use_multiprocessing=False)
print("Testing accuracy and loss")
print(" accuracy =", pred_gen[1] * 100)
print(" loss =", pred_gen[0])
print("*****************************")

accuracy_val = eval_gen[1] * 100
accuracy_test =pred_gen[1] * 100
loss_val = eval_gen[0]
loss_test = pred_gen[0]

predictions = model.predict(test_generator, steps =None, workers=1, use_multiprocessing=False)

predictions = predictions > 0.5

# Separate Confusion Matrices
y_true = y_test
y_pred = predictions

# # Creating the Confusion Matrix
cm = confusion_matrix(y_test, predictions)
target_names = ['Class0', 'Class1']
cr = classification_report(y_test, predictions, target_names=target_names)

auc_Value = metrics.roc_auc_score(y_test, predictions)*100
cm = confusion_matrix(y_test, predictions)

TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]

sensitivity = (cm[0,0]/(cm[0,0]+cm[0,1]))*100

specificity = (cm[1,1]/(cm[1,0]+cm[1,1]))*100



def precision(TP, FP):
    return TP / (TP + FP)
# calculate precision
precision_score = precision(TP, FP)
percision = precision_score

f1score = f1_score(y_test, predictions)
f1 = f1score

# #Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.gcf().set_size_inches(10, 6)
plt.savefig(f'plots/accuracy_loss/accuracy.png', dpi=300)
plt.close()
plt.show()

# # Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'plots/accuracy_loss/loss.png')
plt.gcf().set_size_inches(10, 6)
plt.savefig(f'plots/accuracy_loss/loss.png', dpi=300)
plt.close()
plt.show()

from sklearn.metrics import auc
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.gcf().set_size_inches(10, 6)
plt.savefig(f'plots/roc/roc.png', dpi=300)
plt.close()
plt.show()



print("\n Val Accuracy")
print(accuracy_val)

print("\n Loss")
print(loss_val)

print("\n Test Accuracy")
print(accuracy_test)

print("\n Loss")
print(loss_test)


print("Sum of Confusion Matrices:")
print(cm)



print("\n AUC")
print(auc_Value)

print("\n Sensitivity")
print(sensitivity)

print("\n Specificity")
print(specificity)

print("\n Precision")
print(precision)

print("\n F1-score")
print(f1)

####Grad-CAM methodology
model = keras.models.load_model("model_rgb.keras")

output_folder = 'gradcam_results'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of common image extensions (Add the preffered extension)
image_extensions = ['.tiff', '.tif', '.TIFF', '.jpeg', '.jpg', '.png']
last_conv_layer_name = 'conv2d_2'  # Replace 'conv2d_2' with your actual last convolutional layer name

# Iterate over all files in the directory that match the extensions

for filename in os.listdir(data_dir):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        file_path = os.path.join(data_dir, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"File not found or unable to read: {file_path}")
            continue

        print("Processing:", file_path)
        if class0_name in filename.lower():
            actual_class = class0_name
        else:
            actual_class = class1_name
        # Preprocess the image
        image_resized = cv2.resize(image, (pixel_size, pixel_size))
        image_resized = image_resized.astype('float32') / 255
        image_resized = np.expand_dims(image_resized, axis=0)

        # Predict the class
        preds = model.predict(image_resized)
        prediction = (preds > 0.5).astype(int)

        if prediction == 0:
            print("The model misclassified this instance as " + class0_name)
        else:
            print("The model predicted this instance as " + class1_name)
            
        if prediction == 0:
            predicted_class = 'class1'
            print("The model misclassified this instance as " + class1_name)
        else:
            predicted_class = class0_name
            print("The model predicted this instance as" + class0_name)

        # Define the GradCAM instance with the manually set layer name
        icam = GradCAM(model, np.argmax(preds[0]), layerName=last_conv_layer_name)
        heatmap = icam.compute_heatmap(image_resized)
        heatmap_resized = cv2.resize(heatmap, (pixel_size, pixel_size))

        # Overlay the heatmap onto the original image
        (heatmap, output) = icam.overlay_heatmap(heatmap_resized, image, alpha=0.5)

        # Convert images to RGB for display
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        
        # Concatenate heatmap and original image side by side
        concatenated = np.concatenate((image, heatmap_rgb), axis=1)

        # Save the concatenated image
        output_filename = os.path.join(output_folder, f"gradcam_actual{actual_class}_pred_{predicted_class}_{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(output_filename, concatenated)

        print(f"Saved concatenated image at: {output_filename}")
