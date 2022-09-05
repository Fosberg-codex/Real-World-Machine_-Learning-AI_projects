import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Fosgaxy galaxy classifier"
description = "<p style='text-align: center'><b>As far as human civilization and space travel is concerned, you may want to know types of Galaxies in this universe. We are trying to classifier three types of galaxies here.<b><p>"
article="<p style='text-align: center'> WE have the Spiral galaxy, elliptical galaxy, Peculiar galaxy and Irregular galaxy. Upload your galaxy for correct classification<b></p>"
examples = ['ffff.jpg']
#interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,enable_queue=enable_queue).launch()
