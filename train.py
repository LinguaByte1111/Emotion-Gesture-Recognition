# Importing necessary libraries for model creation and training
from keras.applications import MobileNet
from keras.models import Sequential,Model 
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# MobileNet is designed to work with images of dim 224,224
img_rows,img_cols = 224,224

# Load MobileNet with pre-trained ImageNet weights (excluding the top layers for fine-tuning)
MobileNet = MobileNet(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True by default

for layer in MobileNet.layers:
    layer.trainable = True

# Let's print our layers
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i),layer.__class__.__name__,layer.trainable)

# Function to add the top (fully connected) layers to the MobileNet base model
def addTopModelMobileNet(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    
    top_model = Dense(1024,activation='relu')(top_model)
    
    top_model = Dense(512,activation='relu')(top_model)
    
    top_model = Dense(num_classes,activation='softmax')(top_model)

    return top_model

# Define number of classes (for emotion classification, 5 classes)
num_classes = 5

# Attach the custom top model to the MobileNet base
FC_Head = addTopModelMobileNet(MobileNet, num_classes)

# Define the complete model with MobileNet base + new top layers
model = Model(inputs = MobileNet.input, outputs = FC_Head)

# Print model summary to check the architecture
print(model.summary())

# Define paths for training and validation datasets
train_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/fer2013/train'
validation_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/fer2013/validation'

train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest'
                                   )

# Validation data is only rescaled without augmentation
validation_datagen = ImageDataGenerator(rescale=1./255)

# Batch size for training
batch_size = 32

# Load training data using flow_from_directory (assumes folder structure with subfolders for each class)
train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size = (img_rows,img_cols), # Resize images to 224x224
                        batch_size = batch_size, # Define the batch size
                        class_mode = 'categorical'  # Categorical labels for multi-class classification
                        )

# Load validation data using flow_from_directory
validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir,
                            target_size=(img_rows,img_cols),
                            batch_size=batch_size,
                            class_mode='categorical') # Categorical labels for multi-class classification

# Import optimizers and callbacks
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

# Checkpoint callback: Save the best model based on validation loss
checkpoint = ModelCheckpoint(
    'emotion_face_mobilNet.h5',  # Save model to this file
    monitor='val_loss',  # Monitor validation loss for improvement
    mode='min',  # Save model with minimum validation loss
    save_best_only=True,  # Save only the best model
    verbose=1  # Print when a new best model is saved
)

# Early stopping callback: Stop training if no improvement in validation loss for 10 epochs
earlystop = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    min_delta=0,  # Minimum change to qualify as an improvement
    patience=10,  # Stop after 10 epochs with no improvement
    verbose=1,  # Print information about early stopping
    restore_best_weights=True  # Restore model to best weights when stopping
)

# Learning rate reduction callback: Reduce learning rate if validation accuracy plateaus
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc',  # Monitor validation accuracy for plateau
    patience=5,  # Patience of 5 epochs before reducing the learning rate
    verbose=1,  # Print when the learning rate is reduced
    factor=0.2,  # Reduce learning rate by a factor of 0.2
    min_lr=0.0001  # Minimum learning rate
)

# List of callbacks to be used during training
callbacks = [earlystop, checkpoint, learning_rate_reduction]

# Compile the model with categorical crossentropy loss (for multi-class classification) and Adam optimizer
model.compile(
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    optimizer=Adam(lr=0.001),  # Adam optimizer with a learning rate of 0.001
    metrics=['accuracy']  # Monitor accuracy during training
)

# Define number of training and validation samples
nb_train_samples = 24176  # Number of training samples
nb_validation_samples = 3006  # Number of validation samples

# Define number of epochs (iterations over the entire dataset)
epochs = 25

# Train the model using fit_generator method
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,  # Number of steps per epoch
    epochs=epochs,  # Total number of epochs to train
    callbacks=callbacks,  # Use the defined callbacks during training
    validation_data=validation_generator,  # Validation data
    validation_steps=nb_validation_samples // batch_size  # Number of validation steps per epoch
)
