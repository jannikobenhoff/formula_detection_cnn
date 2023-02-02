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
            return classified_zahlen
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

pil_img = Image.open("screen.jpg")

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
    html.Img(id="pic1", style={"height": "10%", "width": "10%"}),
    html.Img(id="pic2", style={"height": "10%", "width": "10%"}),
    html.Img(id="pic3", style={"height": "10%", "width": "10%"}),
    html.Img(id="pic4", style={"height": "10%", "width": "10%"}),
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
    # dbc.Row(
    #     dcc.Graph(
    #             id='imshow',
    #         ))


], style={'background-color': '#0B0B0C', "height": "100vh", "width": "100%"})


@app.callback(
    Output(component_id='pic1', component_property='src'),
    Output(component_id='pic2', component_property='src'),
    Output(component_id='pic3', component_property='src'),
    Output(component_id='pic4', component_property='src'),
    Input(component_id='submit-val', component_property='n_clicks'),
    #State('input-on-submit', 'value')
)
def update_output(n_clicks):
    print("A: ", n_clicks)
    pil_img = Image.open("screen.jpg").convert('L')

    if n_clicks > 0:
        zahlenList = VideoCamera().take_screen()
        pil_img = Image.open("screen.jpg").convert('L')

        if len(zahlenList) == 4 and zahlenList[0] != None:
            print("--plotting--")

            return Image.fromarray(zahlenList[0].imagearray),\
                   Image.fromarray(zahlenList[1].imagearray),\
                   Image.fromarray(zahlenList[2].imagearray),\
                   Image.fromarray(zahlenList[3].imagearray)
        return pil_img, pil_img, pil_img, pil_img
    else:
        return pil_img, pil_img, pil_img, pil_img



if __name__ == '__main__':
    OPENCV_AVFOUNDATION_SKIP_AUTH = 1
    app.run_server(debug=True)
