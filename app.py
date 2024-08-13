from transformers import pipeline
import gradio as gr

model = pipeline("summarization")

def predict(prompt):
    summery = model(prompt)[0]['summary_text']
    return summery

with gr.Block() as demo:
    textbox = gr.Textbox(placeholer="Enter text block to summarize", lines=4)
    gr.Interface(fn=predict, inputs=textbox, outputs="text")


demo.launch()