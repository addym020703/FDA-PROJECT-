install.packages("imager")
imager::as.cimg(img)


# Load required libraries
library(keras)
library(tensorflow)
library(imager)
library(abind)

# Load and preprocess the images (assuming the images are stored in 'fire' and 'nofire' subfolders)
load_images <- function(directory, target_size = c(224, 224)) {
  images <- list()
  labels <- list()
  
  # List 'fire' and 'nofire' subfolders
  classes <- c("fire", "nofire")  # These are the class names
  
  for (class in classes) {
    class_path <- file.path(directory, class)
    files <- list.files(class_path, full.names = TRUE)
    
    for (file in files) {
      img <- load.image(file) %>%
        resize(target_size[1], target_size[2]) %>%
        as.cimg()
      img <- as.array(img) / 255.0  # Normalize the image
      images <- append(images, list(img))
      labels <- append(labels, class)
    }
  }
  
  # Convert labels to numeric (fire = 1, nofire = 0)
  labels <- as.numeric(factor(labels, levels = c("nofire", "fire"))) - 1
  
  list(
    x = abind::abind(images, along = 1),
    y = labels
  )
}

# Load datasets
train_data <- load_images("/Users/hitteshkumarm/Desktop/COLLEGE/7th sem/FOUNDATIONAL ANALYTICS/PROJECT/Classification/train")  # Path to your train folder
valid_data <- load_images("/Users/hitteshkumarm/Desktop/COLLEGE/7th sem/FOUNDATIONAL ANALYTICS/PROJECT/Classification/valid")  # Path to your validation folder
test_data <- load_images("/Users/hitteshkumarm/Desktop/COLLEGE/7th sem/FOUNDATIONAL ANALYTICS/PROJECT/Classification/test")    # Path to your test folder

x_train <- train_data$x
y_train <- train_data$y
x_valid <- valid_data$x
y_valid <- valid_data$y
x_test <- test_data$x
y_test <- test_data$y

# Load the pre-trained VGG16 model, excluding the top layers (we will add our own)
base_model <- application_vgg16(weights = "imagenet", include_top = FALSE, input_shape = c(224, 224, 3))

# Freeze the layers of the base model to avoid updating during training
freeze_weights(base_model)

# Add custom layers on top of VGG16
model <- keras_model_sequential() %>%
  base_model %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")  # Sigmoid for binary classification

# Compile the model
model %>% compile(
  optimizer = optimizer_adam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Model summary
summary(model)

# Train the model with early stopping
history <- model %>% fit(
  x = x_train, y = y_train,
  validation_data = list(x_valid, y_valid),
  epochs = 20,
  batch_size = 32,
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 4))
)

# Evaluate the model on test data
score <- model %>% evaluate(x_test, y_test)
cat("Test loss:", score$loss, "\n")
cat("Test accuracy:", score$accuracy, "\n")

# Make predictions (if required)
predictions <- model %>% predict(x_test)

