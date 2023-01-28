from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    html.Button(id="a", n_clicks=0)
    ], style={"width":"100%"})


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('a', 'n_clicks'))
def update_figure(selected_year):
    print("aa")
    filtered_df = df[df.year == selected_year]
    print(filtered_df.columns.tolist())
    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     )

    fig.update_layout(transition_duration=500)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)