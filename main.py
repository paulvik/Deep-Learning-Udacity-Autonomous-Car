from data_preprocessing import * 
from neural_net import *
from keras import optimizers


directory = "/work/paulv/PycharmProjects/Datasyn_project/image_folder3/IMG"
crop_directory = "/work/paulv/PycharmProjects/Datasyn_project/image_folder3/IMG_CROP"
driving_log_dir = "/work/paulv/PycharmProjects/Datasyn_project/image_folder3/driving_log.csv" 
total_image_list = os.listdir(directory)

cropped_images = crop_images(total_image_list, directory)
data_set = create_data_set(total_image_list, driving_log_dir, crop_directory)
input_images_set, target_set = split_data_set(data_set)

epochs = 8
batch_size = 64
alpha = 0.0001

model = neural_net()
adam = optimizers.Adam(lr=alpha)
model.compile(loss='mse', optimizer=adam)


history = model.fit(np.array(input_images_set), np.array(target_set), validation_split=0.2, epochs=epochs, batch_size=batch_size, shuffle=True)

model.save("/work/paulv/PycharmProjects/Datasyn_project/models/model9.h5")
print("Saved model to disk")


