import pickle
from flask import Flask, render_template, request
from random import random
import  sys
import os
import numpy as np  
import cv2
import tensorflow as tf
import keras
from keras.preprocessing.image import  img_to_array, load_img  
from tensorflow.keras.applications.vgg16 import VGG16
  # from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
model_VGG16 = VGG16(weights='imagenet', include_top=False)
 
def FeatureExtractionVGG16(file,model):

 
  img = load_img(file, target_size=(224, 224)) # chuyển ảnh về size (224,224)
  x = img_to_array(img)        # chuyển ảnh về thành 1 array
  x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
  x = preprocess_input(x)
  features = model.predict(x)
  features = np.array(features).reshape(-1,1)
  return features
def DuDoan1AnhVGG16(model,file): # dự đoán 1 ảnh dựa theo cách trích xuất VGG16
  X = []
  features = FeatureExtractionVGG16(file,model_VGG16)
  X.append(features)
  X = np.array(X)
  dimX1_, dimX2_, dimX3_ =X.shape
  X = np.reshape(np.array(X), (dimX1_, dimX2_*dimX3_))
  y_pred = model.predict(X)
  return y_pred[0]

def FineTuningVgg16(model,imagefile):
    inv_map_classes = {0: 'Bãi xép gành Ông-Phú Yên', 1: 'Bưu điện - TP.Hồ Chí Minh', 2: 'Bảo tàng Mỹ thuật - Tp.Hồ Chí Minh', 3: 'Bảo tàng lịch sử Việt Nam - Tp.Hồ Chí Minh', 4: 'Bảo tàng tranh 3D Artinus - Tp.Hồ Chí Minh', 5: 'Bến nhà rồng - Tp.Hồ Chí Minh', 6: 'Bờ kè Khánh Hội - Ninh Thuận', 7: 'Chùa Bửu Long - Tp.Hồ Chí Minh', 8: 'Chùa Thanh Lương-Phú Yên', 9: 'Chợ Bến Thành - Tp.Hồ Chí Minh', 10: 'Cánh đồng hoa Thì Là - Ninh Thuận', 11: 'Cánh đồng rêu - Ninh Thuận', 12: 'Cầu Ông Cọp-Phú Yên', 13: 'Cầu ánh sao - Hồ Bán Nguyệt - Tp.Hồ Chí Minh', 14: 'Dinh Độc Lập - Tp.Hồ Chí Minh', 15: 'Ghềnh Đá Dĩa-Phú Yên', 16: 'Gành Đèn-Phú Yên', 17: 'Hang Rái - Ninh Thuận', 18: 'Hòn Yến-Phú Yên', 19: 'Hải đăng Đại Lãnh(Mũi Điện)-Phú Yên', 20: 'Hồ Con Rùa - Tp.Hồ Chí Minh', 21: 'Hồ điều hòa Hồ Sơn-Phú Yên', 22: 'Khu du lịch Suối Tiên - Tp. Hồ Chí Minh', 23: 'Khu du lịch Tanyoli - Ninh Thuận', 24: 'Khu du lịch văn hóa và sinh thái sen Charaih - Ninh Thuận', 25: 'Khu phố Nhật Bản- Tp.Hồ Chí Minh', 26: 'Land Mark 81 - Tp.Hồ Chí Minh', 27: 'Long Vân garden-Phú Yên', 28: 'Mũi Dinh - Ninh Thuận', 29: 'Nhà thờ Mằng Lăng-Phú Yên', 30: 'Nhà thờ Tân Định - Tp. Hồ Chí Minh', 31: 'Nhà thờ Đức Bà - Tp.Hồ Chí Minh', 32: 'Núi Nhạn-Phú Yên', 33: 'Núi Đá Bia-Phú Yên', 34: 'Suối Lồ Ồ - Ninh Thuận', 35: 'Thác Chaper - Ninh Thuận', 36: 'Tháp Nghinh Phong -Phú Yên', 37: 'Tháp PoKlong Garai - Ninh Thuận', 38: 'Trùng Sơn Cổ Tự - Thiền viện trúc lâm Viên Ngộ - Ninh Thuận', 39: 'Tu viện Khánh An - Tp. Hồ Chí Minh', 40: 'Tòa nhà Bitexco - Tp.Hồ Chí Minh', 41: 'Vườn nho Ba Mọi - Ninh Thuận', 42: 'Vịnh Vĩnh Hy - Ninh Thuận', 43: 'Vịnh Vũng Rô-Phú Yên', 44: 'Xóm Rớ-Phú Yên', 45: 'Điện gió Đầm Nại - Ninh Thuận', 46: 'Đường Sách - TP. Hồ Chí Minh', 47: 'Đảo Bình Hưng - Ninh Thuận', 48: 'Đập Đồng Cam-Phú Yên', 49: 'Địa đạo Củ Chi - Tp.Hồ Chí Minh', 50: 'Đồi cát Nam Cương - Ninh Thuận', 51: 'Đồng cừu An Hòa - Ninh Thuận'}
    display = cv2.imread(imagefile)
    #         # print(os.path.join(train_dir,folder,image_ids[j]))
    # cv2_imshow(display)
    display = tf.image.resize(display, [224, 224])
    # print(imagefile.split('/')[-2])
    # print(inv_map_classes[np.argmax(model_used.predict(np.array([display])))])
    return inv_map_classes[np.argmax(model.predict(np.array([display])))]
# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

filename = './Model/model_Logistic_Regression_vgg16.sav'
model_Logistic_Regression_vgg16 = pickle.load(open(filename, 'rb'))
vgg_16_saved_model = keras.models.load_model('./Model/vgg_16_saved_model.h5',custom_objects={'tf': tf})


# Hàm xử lý request
@app.route("/", methods=['GET','POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         
            # Lấy file gửi lên
            image_ = request.files['file']
            
                # Lưu file
            dirs_image = os.listdir(app.config['UPLOAD_FOLDER'])
            print(image_.filename)
            print(app.config['UPLOAD_FOLDER'])
            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image_.filename)
            for file_ in dirs_image:
              if file_ == image_.filename:
                res_model_Logistic_Regression_vgg16 = DuDoan1AnhVGG16(model_Logistic_Regression_vgg16,path_to_save)
                res_vgg_16_saved_model = FineTuningVgg16(vgg_16_saved_model,path_to_save)
                print(image_.filename)
                print(res_model_Logistic_Regression_vgg16,res_vgg_16_saved_model)
                # return render_template("index.html", user_image = image.filename , result = res)
                return render_template("index.html", msg_1 ="Features Extraction using VGG16 + Logistic Regression: " +  str(res_model_Logistic_Regression_vgg16) ,msg_2= "\nFine-tuning using VGG16: "+str(res_vgg_16_saved_model),user_image = image_.filename)

        
            
            print("Save = ", path_to_save)
            image_.save(path_to_save)                                        
            res_model_Logistic_Regression_vgg16 = DuDoan1AnhVGG16(model_Logistic_Regression_vgg16,path_to_save)
            res_vgg_16_saved_model = FineTuningVgg16(vgg_16_saved_model,path_to_save)
            return render_template("index.html",  msg_1 ="Features Extraction using VGG16 + Logistic Regression: " +  str(res_model_Logistic_Regression_vgg16) ,msg_2= "\nFine-tuning using VGG16: "+str(res_vgg_16_saved_model),user_image = image_.filename)


    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)
    app.run()
      # app.run(host="0.0.0.0")
    # app.run(debug=True,host='0.0.0.0', port=9007)
  # http_server = WSGIServer(('', 5000), app)
  # http_server.serve_forever()