import gradio as gr
from datasets import load_dataset
import os
import rootutils 

root = rootutils.setup_root(__file__, dotenv=True)

ds = load_dataset("RIET-lab/moral-kg-sample-labels", token=os.getenv("HF_TOKEN"))["train"]
df = ds.to_pandas()

def show_table():
    return df

gr.Interface(fn=show_table, inputs=[], outputs=gr.Dataframe()).launch()
