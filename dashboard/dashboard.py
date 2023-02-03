import base64
import io

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html

from flask import Flask, Response, request
import cv2
from PIL import Image

from dash.dependencies import Output, Input, State, MATCH, ALL

from scanning import scan_process
from transfer import predict
from webcam import *
from styles import *

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


button_group = html.Div([
    html.Div(
        [
            html.Button(
                id={"type": "operation", "index": 0},
                children=html.Div(
                    [html.H5("Calculate", style={'font-size': 15, 'padding-left': 10, "padding-bottom": 10}),

                     html.Img(src="/assets/icons/calc.png",
                              height=50,
                              style={"filter": "brightness(1) invert(0)"}
                              ),
                     ]), style=button_style

            ),
            html.Button(id={"type": "operation", "index": 1},

                        children=html.Div(
                            [html.H5("Plot", style={'font-size': 15, 'padding-left': 10, "padding-bottom": 10}),

                             html.Img(src="/assets/icons/plot.png",
                                      height=50,
                                      style={"filter": "brightness(1) invert(0)"}
                                      ),
                             ]), style=button_style
                        ),
            html.Button(id={"type": "operation", "index": 2},

                        children=html.Div([
                            html.H5("Equation", style={'font-size': 15, 'padding-left': 10, "padding-bottom": 10}),
                            html.Img(src="/assets/icons/function.png",
                                     height=50,
                                     style={"filter": "brightness(1) invert(0)"}
                                     ),
                        ]), style=button_style
                        ),

        ], style={"margin-top": 25, "margin-left": 25, "margin-right": 25, "display": "flex", "flex-direction": "row",
                  "width": "100%",
                  "justify-content": "center", "align-items": "center"}
    ),
], style={"display": "flex", "flex-direction": "row", "width": "100%",
          "justify-content": "center", "align-items": "center"})

webcam_group = dbc.Row(style={"margin-top": 25},
                 children=[
                     dbc.Col(width=7, id="screen"),
                     dbc.Col(width=1, children=html.Div(id='output-image-upload')),
                     dbc.Col(width=4, children=html.Div([
                         html.Button(id={"type": "input", "index": 0}, n_clicks=0, style=button_style,
                                     children=html.Div([
                                         html.H5('Take Photo'),
                                         html.Img(src="/assets/icons/webcam.png",
                                                  height=50,
                                                  style={"filter": "brightness(1) invert(0)",
                                                         "margin-top": 5}
                                                  ),
                                     ])),

                         html.Button(
                             children=dcc.Upload(id={"type": "input", "index": 1}, children=[html.Div(
                                 [html.H5('Select Files'), html.Img(src="/assets/icons/upload.png",
                                                                    height=50,
                                                                    style={
                                                                        "filter": "brightness(1) invert(0)",
                                                                        "margin-top": 5}
                                                                    ),
                                  ],
                             )]),
                             style=button_style)
                     ], style={"height": "100%", "justify-content": "center", "display": "flex",
                               "align-items": "center", "flex-direction": "column", }))
                 ])

webcam = html.Div([html.H5("Webcam", style={"color": " white", "margin-bottom": 5}),
                             html.Img(src="/video_feed",
                                      style={'height': '90%', 'width': '90%', "margin-left": 25,
                                             "margin-right": 25, "border-radius": "2em"})])

app.layout = html.Div(
    [
        html.Div(style={"height": "10%", 'background-color': black, "border-bottom-left-radius": "2em",
                        "border-bottom-right-radius": "2em"},
                 children=dbc.Row([
                     dbc.Col([
                         html.H1("MATHTECTION",
                                 style={'margin-top': 25, 'margin-bottom': 0, "color": "white"}),
                         html.Span("Machine Learning Algorithm", style={"color": "white"})
                     ], width=12)], justify="left"), ),

        button_group,
        webcam_group,
        #html.Div(id="screen"),
        dbc.Row([html.Hr(style={"margin-top":25})]),
        html.Div(id={"type": "pics", "index": 0}, n_clicks=0,
                 style={"justify-content": "center", "display": "flex",
                               "align-items": "center", "flex-direction": "row",}),
        html.Div(id={"type": "pics", "index": 1}, n_clicks=0),
        html.Div(id="pics", n_clicks=0),
        html.Div(id="pics2", n_clicks=0, style={"justify-content": "center", "display": "flex",
                               "align-items": "center", "flex-direction": "row","margin-top":25}),
        html.Div(id="upload-pics", n_clicks=0),
        html.Div(id="upload-pics2", n_clicks=0, style={"margin": 20}),
        html.Div(id="delete"),
        html.Div(id="ready"),
        html.Div(id="img0", n_clicks=0),
        # html.Img(id="img0", src=Image.open("screen.jpg").convert('L'), n_clicks=0),
        dcc.Store(id='delete-image')
    ], style={'background-color': black, "height": "100vh", "width": "100%", "overflow": "scroll"})


@app.callback(
    Output("pics", "style"),
    Input("ready", "n_clicks")
)
def ready(click):
    if click != None and click != 0:
        img_list = []
        zahlenList = scan_process("screen.jpg")
        for i, zahl in enumerate(zahlenList):
            if i not in del_img:
                img_list.append(zahl)
        predict(img_list)


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
                    children=html.Div([html.H5('Reselect'),
                                       html.Img(src="/assets/icons/reload.png",
                                                height=50,
                                                style={"filter": "brightness(1) invert(0)"}
                                                ),
                                       ], style={"flex-direction": "column", "justify-content": "center",
                                                 "align-items": "center"}),
                    style=select_button_style#{"height": "100px", "border-radius": "1.5rem",
                           #"background-color": "#C2654E", "border": "none", "margin-left": 30}
                    ))
    return img_list


@app.callback(
    Output({'type': 'pics', "index": 0}, 'children'),
    Output('screen', 'children'),
    Input({'type': 'input', 'index': 0}, 'n_clicks'),
    Input({'type': 'input', 'index': 1}, 'contents'),
    State({'type': 'input', 'index': 1}, 'filename'),
)
def update_input(click, content, filename):
    try:
        index = dash.ctx.triggered_id["index"]
    except TypeError:
        return html.Div(), \
               webcam
    print(dash.ctx.triggered_id)
    print(click)
    if index == 0:
        '''Webcam'''
        if click != 0 and click != None:
            img_list = []
            print((click-1) % 2)
            if (click-1) % 2:
                return img_list, webcam
            else:
                zahlenList = VideoCamera().take_screen()

                if zahlenList != None and zahlenList[0] != None:
                    print("--plotting--")
                    for i, zahl in enumerate(zahlenList):
                        print("img{}".format(i))
                        img_list.append(html.Img(id={"type": "img", "index": i},
                                                 n_clicks=0, src=Image.fromarray(zahl.imagearray),
                                                 style={"margin-right": "10px", "height": "5%",
                                                        "width": "5%", "border": "2px white solid"})
                                        )
                    img_list.append(
                        html.Button(id="delete",
                                    children=html.Div([html.H5('Reselect'),
                                                       html.Img(src="/assets/icons/reload.png",
                                                                height=50,
                                                                style={"filter": "brightness(1) invert(0)"}
                                                                ),
                                                       ],
                                                      style={"flex-direction": "column", "justify-content": "center",
                                                             "align-items": "center"}),
                                     style= select_button_style #{"height": "100px", "border-radius": "1.5rem",
                                    #        "background-color": "#C2654E", "border": "none", "margin-left": 30}
                                    ))
                    return img_list, html.Div([html.H5("Webcam photo", style={"color": "white","margin-bottom":5}),
                                               html.Img(src=Image.open("screen.jpg"), style={'height': '90%', 'width': '90%', "border": "2px white solid", "margin-top": 5})])
        else:
            return html.Div(), webcam
    else:
        '''Uploaded Photo'''
        if content is not None:
            return parse_contents(content, filename),html.Div([
                html.H5(filename, style={"color":"white", "margin-bottom":5}),
                html.Img(src=Image.open("screen.jpg"),
                         style={"height": "90%", "width": "90%", "border": "2px white solid", "margin-top": 5})])


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
            return {"margin-right": "10px", "height": "5%", "width": "5%", "border": "4px red solid"}
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
                        children=html.Div([html.H5('   Ready!   '),
                                           html.Img(src="/assets/icons/neural.png",
                                                    height=50,
                                                    style={"filter": "brightness(1) invert(0)"}
                                                    ),
                                           ], style={"flex-direction": "column", "justify-content": "center",
                                                     "align-items": "center"}),
                        style=ready_button_style #{"height": "100px", "border-radius": "1.5rem",
                               #"background-color": "green", "border": "none", "margin-left": 30}
        ))
        return img_list


@app.callback(
    Output({"type": "operation", "index": ALL}, "style"),
    Input({"type": "operation", "index": ALL}, "n_clicks")
)
def operation_select(click):
    try:
        index = dash.ctx.triggered_id["index"]
    except TypeError:
        return operation_button_style_clicked, operation_button_style, operation_button_style

    match index:
        case 0:
            return operation_button_style_clicked, operation_button_style, operation_button_style
        case 1:
            return operation_button_style, operation_button_style_clicked, operation_button_style
        case 2:
            return operation_button_style, operation_button_style, operation_button_style_clicked
        case _:
            return operation_button_style_clicked, operation_button_style, operation_button_style


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
