from fastapi import FastAPI, Query
from typing import List, Optional
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from utils.load_data import get_test_data_loader
from torch.utils.data import DataLoader
from utils.helper import convert_to_timestamp
import torch
import json
import os
import numpy as np
from datetime import datetime
from models.GPT4TS_multi_prompt_residual import GPT4TS_multi

# app = FastAPI()
#
# app.mount("/static", StaticFiles(directory="static"), name="static")
# device = torch.device('cpu')
# models = {96: None, 192: None, 336: None, 720: None}

# @app.on_event("startup")

# async def load_model(checkpoints_path, length):
#     global model
#     # Step 2: Load the model during the startup event
#     model = GPT4TS_multi(device, pred_len=length)
#     model = model.load_state_dict(torch.load(checkpoints_path, map_location=device))


import gradio as gr

# 定义Gradio界面
iface = gr.Interface(
    fn=forecast_for_gradio,  # 使用新的预测函数
    inputs=[
        gr.inputs.Textbox(label="Datasets"),  # 对应于datasets参数
        gr.inputs.Number(label="Lengths"),  # 对应于lengths参数
        gr.inputs.Slider(minimum=0, maximum=10000, default=0, label="Index")  # 对应于index参数
    ],
    outputs=gr.outputs.JSON(label="Forecast Results")  # 输出为JSON
)


def display_page():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return html_content


# 这是转换后的forecast函数，与原始函数逻辑相同
def forecast_for_gradio(datasets, lengths, index):
    async def forecast(datasets: str = Query(...), lengths: int = Query(...), index: int = 0,
                       dataloader: Optional[List[int]] = Query(None)):
        print("lengths: ", lengths)
        print("datasets: ", datasets)
        global model
        num_workers = 0
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        filename = './dataset/test_' + datasets + str(lengths) + '_dataset.pth'

        if os.path.exists(filename):
            print("pth exist, loading...")
            test_dataset = torch.load(filename)
        else:
            print(datasets + str(lengths) + " dataset not exist, create new pth")
            test_dataset = get_test_data_loader(datasets, lengths)
            torch.save(test_dataset, filename)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last)
        # print(test_loader)
        # data_list = [test_loader.dataset[i] for i in range(len(test_loader.dataset))]
        # data_list = [
        #                 (arr1.tolist(), arr2.tolist(), arr3.tolist(), arr4.tolist(), arr5.tolist(), arr6.tolist(), arr7.tolist())
        #                 for arr1, arr2, arr3, arr4, arr5, arr6, arr7 in data_list
        #             ]
        # all_timestamps = []
        # all_pred_timestamps = []
        # all_values = []
        # all_predictions = []

        data = test_loader.dataset[index]
        # print(np.array(data[0]).shape)

        batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid = torch.tensor(
            data[0]).unsqueeze(
            dim=0), torch.tensor(data[1]).unsqueeze(dim=0), \
            torch.tensor(data[2]).unsqueeze(dim=0), torch.tensor(data[3]).unsqueeze(dim=0), \
            torch.tensor(data[4]).unsqueeze(dim=0), torch.tensor(data[5]).unsqueeze(dim=0), \
            torch.tensor(data[6]).unsqueeze(dim=0)
        batch_x = batch_x.float().to(device)
        seq_trend = seq_trend.float().to(device)
        seq_seasonal = seq_seasonal.float().to(device)
        seq_resid = seq_resid.float().to(device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # print(batch_y.shape)
        # print(seq_trend.shape)
        # print(seq_seasonal.shape)
        # print(seq_resid.shape)
        # print(batch_y.shape)
        # print(batch_x_mark.shape)
        # print(batch_y_mark.shape)

        if models[lengths] is None:
            # model = GPT4TS_multi(device, pred_len=lengths)
            print(str(lengths) + ' model not exist, creating model')
            best_model_path = 'checkpoints/' + str(lengths) + 'checkpoint.pth'
            # print(best_model_path)
            # model.load_state_dict(torch.load(best_model_path, map_location=device))
            # Step 2: Load the model during the startup event
            models[lengths] = GPT4TS_multi(device, pred_len=lengths)
            models[lengths].load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            print(str(lengths) + ' model exist')
        outputs = models[lengths](batch_x, 0, seq_trend, seq_seasonal, seq_resid)
        print("outputs init: ", outputs.shape)
        outputs = outputs[:, -lengths:, :]
        print("outputs before: ", outputs.shape)
        batch_y_mark = batch_y_mark[:, -lengths:, :]
        outputs = outputs.squeeze()
        print("outputs: ", outputs.shape)

        # print(batch_x_mark)
        # print(batch_x_mark.shape) # [336, 5]

        batch_x = batch_x.squeeze()
        batch_x_mark = batch_x_mark.squeeze()
        # 2020, 10, 15, 12,0
        # 2020, 10, 15, 12,0
        batch_y_mark = batch_y_mark.squeeze()
        print("batch_x_mark: ", batch_x_mark.shape)
        print("batch_y_mark: ", batch_y_mark.shape)
        print('batch_x_mark: ', batch_x_mark)
        timestamps = [convert_to_timestamp(*row, flag=datasets) for row in batch_x_mark]
        print('before timestamps length:', len(timestamps))
        # all_timestamps.append(timestamps)
        pred_timestamps = [convert_to_timestamp(*row, flag=datasets) for row in batch_y_mark]
        timestamps.extend(pred_timestamps)
        print('after timestamps length:', len(timestamps))
        # all_pred_timestamps.append(pred_timestamps)
        true_values = batch_x.cpu().numpy().tolist()
        true_values_y = batch_y[:, -lengths:, :].squeeze().cpu().numpy().tolist()
        true_values.extend(true_values_y)
        # all_values.append(batch_x.cpu().numpy().tolist())
        # all_predictions.append(outputs.detach().cpu().numpy().tolist())
        forecast_data = {
            "time": timestamps,
            "values": true_values,
            # "values": batch_x.cpu().numpy().tolist(),
            "prediction": outputs.detach().cpu().numpy().tolist(),
            "prediction_time": pred_timestamps,
            "data_max": len(test_loader.dataset)
            # "data_list": data_list
        }

        # return JSONResponse(content=json.dumps(forecast_data))
        return forecast_data
        return 1

        for i, data in enumerate(test_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid = data[0], data[1], data[
                2], \
                data[3], data[4], data[5], data[6]
            batch_x = batch_x.float().to(device)
            seq_trend = seq_trend.float().to(device)
            seq_seasonal = seq_seasonal.float().to(device)
            seq_resid = seq_resid.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            print(batch_x.shape)
            print(seq_trend.shape)
            print(seq_seasonal.shape)
            print(seq_resid.shape)
            print(batch_y.shape)
            print(batch_x_mark.shape)
            print(batch_y_mark.shape)
        #     model = GPT4TS_multi(device)
        #     best_model_path = 'checkpoints/336checkpoint.pth'
        #     model.load_state_dict(torch.load(best_model_path, map_location=device))
        #     outputs = model(batch_x, 0, seq_trend, seq_seasonal, seq_resid)
        #     print("outputs init: ", outputs.shape)
        #     outputs = outputs[:, -lengths:, :]
        #     print("outputs before: ", outputs.shape)
        #     batch_y_mark = batch_y_mark[:, -lengths:, :]
        #     outputs = outputs.squeeze()
        #     print("outputs: ", outputs.shape)

        #     # print(batch_x_mark)
        #     # print(batch_x_mark.shape) # [336, 5]

        #     batch_x = batch_x.squeeze()
        #     batch_x_mark = batch_x_mark.squeeze()
        #     batch_y_mark = batch_y_mark.squeeze()
        #     print("batch_x_mark: ", batch_x_mark.shape)
        #     print("batch_y_mark: ", batch_y_mark.shape)
        #     timestamps = [convert_to_timestamp(*row) for row in batch_x_mark]
        #     all_timestamps.append(timestamps)
        #     pred_timestamps = [convert_to_timestamp(*row) for row in batch_y_mark]
        #     all_pred_timestamps.append(pred_timestamps)

        #     all_values.append(batch_x.cpu().numpy().tolist())
        #     all_predictions.append(outputs.detach().cpu().numpy().tolist())
        # forecast_data = {
        #     "time": timestamps,
        #     "values": batch_x.cpu().numpy().tolist(),
        #     "prediction": outputs.detach().cpu().numpy().tolist(),
        #     "prediction_time": pred_timestamps
        # }

        # return forecast_data
        # forecast_data = {
        #     "time": all_timestamps,
        #     "values": all_values,
        #     "prediction": all_predictions,
        #     "prediction_time": all_pred_timestamps
        # }

        # return forecast_data

        model = GPT4TS_multi(args, device)
        best_model_path = 'checkpoints/336checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))

        return {"load": "successful"}


# 启动Gradio界面
iface.launch()
