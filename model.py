from keras.models import model_from_json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Sad", "Surprise",
                     "Neutral"]
emo_detect = []
#C:\Users\Aditya\PycharmProjects\Expression_Students\face_model.json
with open('C:\\Users\\Aditya\\PycharmProjects\\Expression_Students\\face_model.json', "r") as json_file:
         loaded_model_json = json_file.read()
         #print(loaded_model_json)
         loaded_model = model_from_json(loaded_model_json)
         print('----------',loaded_model)
        # load weights into the new model
loaded_model.load_weights('C:\\Users\\Aditya\\PycharmProjects\\Expression_Students\\face_model1.h5')
graph = tf.get_default_graph()
print("Model loaded from disk")
loaded_model.summary()

def predict_emotion(img):
    #print("---------------",img)
    #loaded_model._make_prediction_function()  # added
    global graph  # added
    with graph.as_default():  # added
        #model.predict_proba(new_X)

        preds = loaded_model.predict(img)
    res = np.argmax(preds)
    emo_detect.append(EMOTIONS_LIST[res])

    print(res)
    print(EMOTIONS_LIST[res])
    return EMOTIONS_LIST[res],res

#
# if __name__ == '__main__':
#     pass