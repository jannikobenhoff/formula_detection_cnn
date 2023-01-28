import dash
# import dash_core_components as dcc
# import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output

import pandas as pd

from flask import Flask, Response, request
import numpy as np
import cv2
from PIL import ImageGrab, Image
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from dash.dependencies import Output, Input, State
from queue import Queue

from scanning import scan_process, Zahl


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def take_screen(self):
        print("-- taking screen --")
        _, image = self.video.read()
        img = Image.fromarray(image)
        img = img.convert('L')
        img.save("screen.jpg")
        classified_zahlen = scan_process("screen.jpg")
        if len(classified_zahlen) != 0:
            return classified_zahlen[0]
        else:
            return None

    def get_frame(self):
        success, image = self.video.read()
        image = cv2.flip(image, 1)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

@server.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    if request.method == "POST":
        print("POST")
    if request.method == "GET":
        print("GET")

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

sidebar = html.Div(
    [
        html.Span("Filters"),
        html.Hr(),
        html.Span("A simple sidebar layout with filters"),
        dbc.Nav(
            [
                dcc.Dropdown(id='one'),
                html.Br(),
                dcc.Dropdown(id='two'),
                html.Br(),
                dcc.Dropdown(id='three')

            ],
            vertical=True,
            pills=True,
        ),
        dbc.Col([
                #html.Div(dcc.Input(id='input-on-submit', type='text', style={'margin': 15})),
                html.Button('Take screen', id='submit-val', n_clicks=0),
                #html.Div(id='container-button-basic',
                 #       children='Enter a value and press submit'),

            ])
    ],
    # style=SIDEBAR_STYLE,
)


app.layout = html.Div(
    [
    dbc.Row([
        dbc.Col([
            html.H1("Handwritten Mathematical Formulas", style={'margin-top': 25, 'margin-bottom': 25,})
        ], width=12)], justify="center"),
    dbc.Row([
        html.Img(src="/video_feed", style={'height': '40%', 'width': '40%', "margin": 25, "border-radius": "2em"}),
        dbc.Col(sidebar),

    ]),
    dbc.Row([html.Hr()]),
    # dbc.Row(dcc.Graph(figure={
    #                 'data': [
    #                     {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
    #                     {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
    #                 ],
    #                 'layout': {
    #                     'title': 'Dash Data Visualization',
    #                     "height": 300,
    #                     "width": 350,
    #                 },
    #
    #             })),
    # dcc.Graph(
    #             id='imshow',
    #         )

], style={'background-color': '#0B0B0C', "height": "100vh", "width": "100%"})

@app.callback(
    Output(component_id='imshow', component_property='figure'),
    Input(component_id='submit-val', component_property='n_clicks'),
    #State('input-on-submit', 'value')
)
def update_output(n_clicks):
    print("A: ", n_clicks)
    if n_clicks > 0:
        #firstZahl = VideoCamera().take_screen()
        #if firstZahl != None:
        print("--plotting--")
        # img = Image.open("screen.jpg").convert('L')
        # img = np.array(img)
        # fig = px.imshow(img)
        #fig = go.Figure(go.Scatter(x=[1,2,3,4,5], y=[1,2,3,4,5]))
        #fig.update_layout(transition_duration=50)
        # return fig


if __name__ == '__main__':
    OPENCV_AVFOUNDATION_SKIP_AUTH = 1
    app.run_server(debug=True)
#
# import asyncio
# import base64
# import dash, cv2
# import dash_html_components as html
# import threading
#
# from dash.dependencies import Output, Input
# from quart import Quart, websocket
# from dash_extensions import WebSocket
#
#
# class VideoCamera(object):
#     def __init__(self, video_path):
#         self.video = cv2.VideoCapture(video_path)
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         success, image = self.video.read()
#         _, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()
#
#
# # Setup small Quart server for streaming via websocket.
# server = Quart(__name__)
# delay_between_frames = 0.05  # add delay (in seconds) if CPU usage is too high
#
#
# @server.websocket("/stream")
# async def stream():
#     camera = VideoCamera(0)  # zero means webcam
#     while True:
#         if delay_between_frames is not None:
#             await asyncio.sleep(delay_between_frames)  # add delay if CPU usage is too high
#         frame = camera.get_frame()
#         await websocket.send(f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}")
#
#
# # Create small Dash application for UI.
# app = dash.Dash(__name__)
# app.layout = html.Div([
#     html.Img(style={'width': '40%', 'padding': 10}, id="video"),
#     WebSocket(url=f"ws://127.0.0.1:5000/stream", id="ws")
# ])
# # Copy data from websocket to Img element.
# app.clientside_callback("function(m){return m? m.data : '';}", Output(f"video", "src"), Input(f"ws", "message"))
#
# if __name__ == '__main__':
#     threading.Thread(target=app.run_server).start()
#     server.run()