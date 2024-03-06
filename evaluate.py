from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('C:/Users/rjeev/Desktop/project/data set/PCB-Defects-Classification-Using-Deep-Learning-main/Code/pcb_defect.h5')


# Define data generators for evaluation
test_datagen = ImageDataGenerator(data_format="channels_last")
test_generator = test_datagen.flow_from_directory('C:/Users/rjeev/Desktop/project/data set/DeepPCB-master/PCBData',
                                                 target_size=(64, 64), color_mode='grayscale', batch_size=8, class_mode='categorical')

# Evaluate the model
evaluation = model.evaluate(test_generator)
from tensorflow.keras.models import save_model
save_model(model, 'C:\\Users\\rjeev\\Desktop\\project\\data set\\PCB-Defects-Classification-Using-Deep-Learning-main\\Code\\pcb_defect_test.h5')
print('Loss:', evaluation[0])
print('Accuracy:', evaluation[1])
