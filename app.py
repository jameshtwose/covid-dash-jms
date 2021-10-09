import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# import dash_core_components as dcc
# import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
df = df.assign(date=lambda x: pd.to_datetime(x["date"]))

tmp_df = df.drop(["iso_code", "continent", "location", "date"], axis=1)

features_list = tmp_df.count()[tmp_df.count() > tmp_df.shape[0] * 0.4].index.sort_values().tolist()

country_list = df["location"].unique().tolist()

app.layout = html.Div([
    html.H1(id='H1', children='COVID Numbers', style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),
    html.Div([dcc.Dropdown(
            id='country_choice',
            options=[{'label': i.title().replace("_", " "), 'value': i} for i in country_list],
            value="Netherlands"
        )], style={'width': '49.8%', 'display': 'inline-block'}),
    html.Div([dcc.Dropdown(
        id='feature_choice',
        options=[{'label': i.title().replace("_", " "), 'value': i} for i in features_list],
        value="new_cases"
    )], style={'width': '49.8%', 'float': 'right', 'display': 'inline-block'}),

    # html.Div(id='display-value')
    dcc.Graph(id='line_plot'),
    html.Div(html.A(children="Created by James Twose",
                    href="https://services.jms.rocks",
                    style={'color': "#743de0"}),
                    style = {'textAlign': 'center',
                             'color': "#743de0",
                             'marginTop': 40,
                             'marginBottom': 40})
]
)


@app.callback(Output(component_id='line_plot', component_property='figure'),
              [Input(component_id='country_choice', component_property='value'),
              Input(component_id='feature_choice', component_property='value')]
              )
def graph_update(country_choice, feature_choice):
    # print(dropdown_value)
    # fig = go.Figure([go.Scatter(x=df.loc[df["location"] == '{}'.format(country_choice), 'date'],
    #                             y=df.loc[df["location"] == '{}'.format(country_choice), feature_choice],
    #
    #                             # line=dict(color='firebrick', width=1)
    #                             )
    #                  ])

    fig = px.line(
        data_frame=df.loc[df["location"] == '{}'.format(country_choice), :],
        x='date',
        y=feature_choice,
        markers=True
    )
    fig.update_traces(line_color='#743de0')

    fig.update_layout(title='COVID results == {}'.format(country_choice),
                      xaxis_title='Date',
                      yaxis_title='{}'.format(feature_choice)
                      )
    return fig


# @app.callback(dash.dependencies.Output('display-value', 'children'),
#                 [dash.dependencies.Input('dropdown', 'value')])
# def display_value(value):
#     return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=False)
