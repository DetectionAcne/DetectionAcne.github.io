from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("model/ACNEModel.h5")


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about_acne.html")

@app.route("/informasi")
def informasi():
    return render_template("informasi.html")

@app.route("/detection", methods=["GET", "POST"])
def detection():
    if request.method == "GET":
        return render_template("detection.html")
    else:
        print("Test Upload File", request.files)
        file_gambar = request.files['gambar']
        print(file_gambar)
        file_gambar.save("upload/test.jpg")

        path = "upload/test.jpg"
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        classes = model.predict(x)

        hasil_prediksi = None
        for i, class_ in enumerate(classes[0]):
            if class_ == classes[0].max():
                if i == 0:
                    hasil_prediksi = "Jerawat Conglobata"
                    print("Jerawat_Conglobata")

                elif i == 1:
                    hasil_prediksi = "Jerawat Hormonal"
                    print("Jerawat_Hormonal")
                    
                elif i == 2:
                    hasil_prediksi = "Jerawat Pasir"
                    print("Jerawat_Pasir")

                else:
                    hasil_prediksi = "papula"
                    print("Jerawat_Papula")

        return render_template("detection.html", hasil=hasil_prediksi )

if __name__ == "__main__":
    app.run(debug=True)