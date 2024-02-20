from fastapi import FastAPI, Query
from typing import List, Optional
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from utils.load_data import get_test_data_loader
from torch.utils.data import DataLoader
from utils.helper import convert_to_timestamp
import torch
import json
import plotly.graph_objs as go
import os
import pandas as pd
import numpy as np
from datetime import datetime
from models.GPT4TS_multi_prompt_residual import GPT4TS_multi
import plotly.express as px

import gradio as gr
# app = FastAPI()
#
# app.mount("/static", StaticFiles(directory="static"), name="static")
device = torch.device('cpu')
models = {96: None, 192: None, 336: None, 720: None}

# @app.on_event("startup")

# async def load_model(checkpoints_path, length):
#     global model
#     # Step 2: Load the model during the startup event
#     model = GPT4TS_multi(device, pred_len=length)
#     model = model.load_state_dict(torch.load(checkpoints_path, map_location=device))


# help(gr.outputs)
# def display_page():
#     with open("static/index.html", "r") as f:
#         html_content = f.read()
#     return html_content


# 这是转换后的forecast函数，与原始函数逻辑相同
async def forecast_for_gradio(datasets: str, lengths: int, index: int = 0):
    # 确保长度和索引参数是整数
    lengths = int(lengths)
    index = int(index)

    filename = f'./dataset/test_{datasets}{lengths}_dataset.pth'
    if not os.path.exists(filename):
        test_dataset = get_test_data_loader(datasets, lengths)
        torch.save(test_dataset, filename)
    else:
        test_dataset = torch.load(filename)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    data = test_loader.dataset[index]
    batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid = [
        torch.tensor(data[i]).unsqueeze(0).float().to(device) for i in range(7)
    ]

    if models[lengths] is None:
        print(f"{lengths} model not exist, creating model")
        best_model_path = f'checkpoints/{lengths}checkpoint.pth'
        models[lengths] = GPT4TS_multi(device, pred_len=lengths)
        models[lengths].load_state_dict(torch.load(best_model_path, map_location=device))

    outputs = models[lengths](batch_x, 0, seq_trend, seq_seasonal, seq_resid)
    outputs = outputs.squeeze().detach().cpu().numpy().tolist()

    timestamps = [convert_to_timestamp(*row.tolist(), flag=datasets) for row in batch_x_mark.squeeze()]
    pred_timestamps = [convert_to_timestamp(*row.tolist(), flag=datasets) for row in batch_y_mark.squeeze()[-lengths:]]

    timestamps.extend(pred_timestamps)

    true_values = batch_x.squeeze().cpu().numpy().tolist()
    true_values_y = batch_y.squeeze()[-lengths:].cpu().numpy().tolist()
    true_values.extend(true_values_y)

    forecast_data = {
        "time": timestamps,
        "values": true_values,
        "prediction": outputs,
        "prediction_time": pred_timestamps,
        "data_max": len(test_loader.dataset)
    }

    # 此处应该是您用于生成图表和数据表格的逻辑
    plot_figure = forecast_plot(forecast_data)
    data_table = data_graph(forecast_data)

    return plot_figure, data_table


def forecast_plot(forecast_data):
    # 分别提取时间、真实值、预测值及其对应的时间
    timestamps = forecast_data["time"]
    true_values = forecast_data["values"]
    prediction = forecast_data["prediction"]
    prediction_time = forecast_data["prediction_time"]

    # 创建真实值的折线图
    trace1 = go.Scatter(x=timestamps, y=true_values, mode='lines', name='Actual')

    # 创建预测值的折线图
    trace2 = go.Scatter(x=prediction_time, y=prediction, mode='lines', name='Prediction')

    # 定义图表布局
    layout = go.Layout(title='Prediction Data', xaxis=dict(title='Time'), yaxis=dict(title='Value'))

    # 生成图表
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # 返回图表对象
    return fig


# 定义数据表格输出函数
def data_graph(forecast_data):
    # 提取时间和真实值数据
    timestamps = forecast_data["time"]
    true_values = forecast_data["values"]

    # 创建数据表格
    df = pd.DataFrame({'Time': timestamps, 'Value': true_values})

    return df


# 定义Gradio界面
iface = gr.Interface(
    fn=forecast_for_gradio,
    inputs=[
        gr.inputs.Dropdown(label="Select Dataset", choices=["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "Traffic"]),
        gr.inputs.Dropdown(label="Select Predict Length", choices=["96", "192", "336", "720"]),
        gr.inputs.Slider(minimum=0, maximum=10000, step=1, label="Index")
    ],
    outputs=[
        gr.Plot(label="Forecast Plot"),
        gr.Dataframe(label="Data Graph")
    ]
)

# 启动Gradio界面
iface.launch(share=True)
