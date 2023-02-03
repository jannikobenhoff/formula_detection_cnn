import base64
import io

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

from dash.dependencies import Output, Input, State, MATCH, ALL
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
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP,
                                                               dbc.icons.BOOTSTRAP,
                                                               dbc.icons.FONT_AWESOME])


@server.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    if request.method == "POST":
        print("POST")
    if request.method == "GET":
        print("GET")

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


button_style = {"height": "100px", "border-radius": "1.5rem",
                "background-color": "#397146", "border": "none", "margin": 5}

operation_button_style = {"width": "70%","height": "100px", "border-radius": "1.5rem",
                          "background-color": "#8C979A", "border": "none", "margin": 5}

operation_button_style_clicked = {"width": "70%", "height": "100px", "border-radius": "1.5rem",
                                  "background-color": "#DCCF57", "border": "none", "margin": 5}

button_group = html.Div([
    html.Div(
        [
            html.Button(
                id={"type": "operation", "index": 0},
                children=html.Div([html.H5("Calculate", style={'font-size': 15, 'padding-left': 10, "padding-bottom":10}),

                          html.Img(src="/assets/icons/calc.png",
                                   height=50,
                                   style={"filter": "brightness(1) invert(0)"}
                                   ),
                          ]), style=button_style

            ),
            html.Button(id={"type": "operation", "index": 1},

                        children=html.Div([html.H5("Plot", style={'font-size': 15, 'padding-left': 10, "padding-bottom":10}),

                                  html.Img(src="/assets/icons/plot.png",
                                           height=50,
                                           style={"filter": "brightness(1) invert(0)"}
                                           ),
                                  ]), style=button_style
                        ),
            html.Button(id={"type": "operation", "index": 2},

                        children=html.Div([
                            html.H5("Equation", style={'font-size': 15, 'padding-left': 10, "padding-bottom":10}),
                                  html.Img(src="/assets/icons/function.png",
                                           height=50,
                                           style={"filter": "brightness(1) invert(0)"}
                                           ),
                                  ]), style=button_style
                        ),

        ], style={"margin-top": 25,"margin-left": 25, "display": "flex", "flex-direction": "row", "width":"100%",
                  "justify-content": "center", "align-items": "center"}
    ),
    html.Div(children=[
        html.Div(html.Button(id={"type": "input", "index": 0}, n_clicks=0, style=button_style,
                             children=html.Div([
                                 html.Span('Take Photo'),
                                 html.Img(src="/assets/icons/webcam.png",
                                          height=50,
                                          style={"filter": "brightness(1) invert(0)",
                                                 "margin-top": 5}
                                          ),
                             ], style={"flex-direction": "column",
                                       "justify-content": "center", "align-items": "center"}))

                 ),
        html.Div([
            dcc.Upload(id={"type": "input", "index": 1},
                       children=html.Button(
                           children=html.Div([html.Span('Select Files'), html.Img(src="/assets/icons/upload.png",
                                                                                  height=50,
                                                                                  style={
                                                                                      "filter": "brightness(1) invert(0)",
                                                                                      "margin-top": 5}
                                                                                  ),
                                              ],
                                             style={"flex-direction": "column",
                                                    "justify-content": "center", "align-items": "center"}),
                           style=button_style), ),
        ])
    ],
        style={"margin-top": 25, "display": "flex", "flex-direction": "row",
               "justify-content": "center", "align-items": "center","width":"100%"}
    ), ], style={"display": "flex", "flex-direction": "row","width":"100%",
                 "justify-content": "center", "align-items": "center"})

# orange: #C2654E
button_agroup = html.Div([html.Div(
    [dbc.RadioItems(
        id="operation",
        className="btn-group",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary",
        labelCheckedClassName="active",
        options=[
            {"label": "Calculation", "value": 1},
            {"label": "Plot", "value": 2},
            {"label": "Equation", "value": 3},
        ],
        value=1,
        style={"color": "red"}
    ),
        html.Div(id="output"),

    ],
    className="radio-group",
    style={"margin-top": 25, "display": "block"}),
    html.Div(children=[
        html.Div(html.Button(id={"type": "input", "index": 0}, n_clicks=0, style=button_style,
                             children=html.Div([
                                 html.Span('Take Photo'),
                                 html.Img(src="/assets/icons/webcam.png",
                                          height=50,
                                          style={"filter": "brightness(1) invert(0)",
                                                 "margin-top": 5}
                                          ),
                             ], style={"flex-direction": "column",
                                       "justify-content": "center", "align-items": "center"}))

                 ),
        html.Div([
            dcc.Upload(id={"type": "input", "index": 1},
                       children=html.Button(
                           children=html.Div([html.Span('Select Files'), html.Img(src="/assets/icons/upload.png",
                                                                                  height=50,
                                                                                  style={
                                                                                      "filter": "brightness(1) invert(0)",
                                                                                      "margin-top": 5}
                                                                                  ),
                                              ],
                                             style={"flex-direction": "column",
                                                    "justify-content": "center", "align-items": "center"}),
                           style=button_style), ),
        ])
    ],
        style={"margin-top": 25, "display": "flex", "flex-direction": "row",
               "justify-content": "center", "align-items": "center"}
    )], style={"display": "flex", "flex-direction": "row",
               "justify-content": "center", "align-items": "center"})

pil_img = Image.open("screen.jpg")

webcam = dbc.Row([
    html.Img(src="/video_feed", style={'height': '40%', 'width': '40%', "margin": 50, "border-radius": "2em"}),
    dbc.Col(html.Div(id='output-image-upload')),

])

app.layout = html.Div(
    [
        html.Div(style={"height": "10%", 'background-color': "#855BE0", "border-bottom-left-radius": "2em",
                        "border-bottom-right-radius": "2em"},
                 children=dbc.Row([
                     dbc.Col([
                         html.H1("Mathtection",
                                 style={'margin-top': 25, 'margin-bottom': 0, "color": "black"})
                     ], width=12)], justify="center"), ),

        button_group,
        webcam,
        dbc.Row([html.Hr()]),
        html.Div(id={"type": "pics", "index": 0}, n_clicks=0),
        html.Div(id={"type": "pics", "index": 1}, n_clicks=0),
        html.Div(id="pics", n_clicks=0, style={"margin": 20}),
        html.Div(id="pics2", n_clicks=0, style={"margin": 20}),
        html.Div(id="upload-pics", n_clicks=0),
        html.Div(id="upload-pics2", n_clicks=0, style={"margin": 20}),
        html.Div(id="delete"),
        html.Div(id="img0", n_clicks=0),
        # html.Img(id="img0", src=Image.open("screen.jpg").convert('L'), n_clicks=0),
        dcc.Store(id='delete-image')
    ], style={'background-color': '#0B0B0C', "height": "100vh", "width": "100%"})


@app.callback(
    Output("pics", "style"),
    Input("ready", "n_clicks")
)
def ready(click):
    print("READY!")
    img_list = []
    zahlenList = scan_process("screen.jpg")
    for i, zahl in enumerate(zahlenList):
        if i not in del_img:
            img_list.append(zahl)
    print(len(img_list))


def parse_contents(contents, filename):
    data = contents.encode("utf8").split(b";base64,")[1]

    zahlenList = scan_process(io.BytesIO(base64.b64decode(data)), save=True)
    print(len(zahlenList))
    img_list = []
    for i, zahl in enumerate(zahlenList):
        img_list.append(html.Img(id={"type": "img", "index": i}, n_clicks=0, src=Image.fromarray(zahl.imagearray),
                                 style={"margin-right": "10px", "height": "5%",
                                        "width": "5%", "border": "2px white solid"})
                        )
    img_list.append(
        html.Button(id="delete",
                    children=html.Div([html.Span('Deselect'),
                                       html.Img(src="/assets/icons/reload.png",
                                                height=50,
                                                style={"filter": "brightness(1) invert(0)"}
                                                ),
                                       ], style={"flex-direction": "column", "justify-content": "center",
                                                 "align-items": "center"}),
                    style={"height": "100px", "border-radius": "1.5rem",
                           "background-color": "#C2654E", "border": "none", "margin-left": 30}))
    return img_list
    # return html.Div([
    #     html.H5(filename, style={"margin-top": 50}),
    #     html.Img(src=contents, style={"height": "70%", "width": "50%", "border": "2px white solid", "margin-top": 5}),
    # ], style={"margin": 25}), img_list


@app.callback(
    Output({'type': 'pics', "index": 0}, 'children'),
    Input({'type': 'input', 'index': 0}, 'n_clicks'),
    Input({'type': 'input', 'index': 1}, 'contents'),
    State({'type': 'input', 'index': 1}, 'filename'),
)
def update_input(click, content, filename):
    try:
        index = dash.ctx.triggered_id["index"]
    except TypeError:
        return html.Div()
    print(dash.ctx.triggered_id)
    print(click)
    if index == 0:
        '''Webcam'''
        if click != 0 and click != None:
            zahlenList = VideoCamera().take_screen()

            if zahlenList != None and zahlenList[0] != None:
                print("--plotting--")
                img_list = []
                for i, zahl in enumerate(zahlenList):
                    print("img{}".format(i))
                    img_list.append(html.Img(id={"type": "img", "index": i},
                                             n_clicks=0, src=Image.fromarray(zahl.imagearray),
                                             style={"margin-right": "10px", "height": "5%",
                                                    "width": "5%", "border": "2px white solid"})
                                    )
                img_list.append(
                    html.Button(id="delete",
                                children=html.Div([html.Span('Reselect'),
                                                   html.Img(src="/assets/icons/reload.png",
                                                            height=50,
                                                            style={"filter": "brightness(1) invert(0)"}
                                                            ),
                                                   ],
                                                  style={"flex-direction": "column", "justify-content": "center",
                                                         "align-items": "center"}),
                                style={"height": "100px", "border-radius": "1.5rem",
                                       "background-color": "#C2654E", "border": "none", "margin-left": 30}))
                return img_list
    else:
        '''Uploaded Photo'''
        if content is not None:
            return parse_contents(content, filename)

del_img = []

@app.callback(
    Output({'type': 'img', 'index': MATCH}, 'style'),
    Input({'type': 'img', 'index': MATCH}, 'n_clicks'),
    State({'type': 'img', 'index': MATCH}, 'id'))
def select_images(click, index):
    if click != 0 and click != None:
        if index["index"] in del_img:
            del_img.remove(index["index"])
        else:
            del_img.append(index["index"])
        print(del_img)
        if click % 2:
            return {"margin-right": "10px", "height": "5%", "width": "5%", "border": "2px red solid"}
        else:
            return {"margin-right": "10px", "height": "5%", "width": "5%", "border": "2px white solid"}
    else:
        return {"margin-right": "10px", "height": "5%", "width": "5%", "border": "2px white solid"}


@app.callback(
    Output(component_id='pics2', component_property='children'),
    Input("delete", "n_clicks")
)
def delete_images(click):
    if click != 0 and click != None:
        img_list = []
        zahlenList = scan_process("screen.jpg")
        for i, zahl in enumerate(zahlenList):
            if i not in del_img:
                img_list.append(
                    html.Img(id={"type": "img2", "index": i}, n_clicks=0, src=Image.fromarray(zahl.imagearray),
                             style={"margin-right": "10px", "height": "5%",
                                    "width": "5%", "border": "2px white solid"})
                )
        img_list.append(
            html.Button(id="ready",
                        children=html.Div([html.Span('Ready!'),
                                           html.Img(src="/assets/icons/neural.png",
                                                    height=50,
                                                    style={"filter": "brightness(1) invert(0)"}
                                                    ),
                                           ], style={"flex-direction": "column", "justify-content": "center",
                                                     "align-items": "center"}),
                        style={"height": "100px", "border-radius": "1.5rem",
                               "background-color": "#C2654E", "border": "none", "margin-left": 30}))
        return img_list


@app.callback(
    Output({"type": "operation", "index": ALL}, "style"),
    Input({"type": "operation", "index": ALL}, "n_clicks")
)
def operation_select(click):
    try:
        index = dash.ctx.triggered_id["index"]
    except TypeError:
        return operation_button_style, operation_button_style, operation_button_style

    match index:
        case 0:
            return operation_button_style_clicked, operation_button_style, operation_button_style
        case 1:
            return operation_button_style, operation_button_style_clicked, operation_button_style
        case 2:
            return operation_button_style, operation_button_style, operation_button_style_clicked
        case _:
            return operation_button_style, operation_button_style, operation_button_style


if __name__ == '__main__':
    OPENCV_AVFOUNDATION_SKIP_AUTH = 1
    app.run_server(debug=True)

'''

Rechnen, Plotten, Gleichungen 
Webcam / Datei hochladen
stimmt das so? -> neu oder auswählen und entfernen (Webcam weg)

rechnen/plotten/lösen lassen

computation time (animieren lassen?) / performance / auslastung 

'''
