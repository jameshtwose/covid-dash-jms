import os
from jmspack.utils import apply_scaling
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

from utils import summary_window_FUN
from sklearn import decomposition

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

window_size_list = [{"label": "week", "value": 7},
                    {"label": "fortnight", "value": 14},
                    {"label": "month", "value": 28}
                    ]

multi_feature_list = ["new_cases", "new_deaths", "new_tests"]

method_list = [{"label": "raw", "value": "raw"},
                    {"label": "smoothed", "value": "smoothed"},
                    # {"label": "per million", "value": "per_million"}
                    ]

app.layout = html.Div([
    html.H1(id='H1', children='COVID Numbers', style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),
    html.Div([dcc.Dropdown(
            id='country_choice',
            options=[{'label': i.title().replace("_", " "), 'value': i} for i in country_list],
            value="Netherlands"
        )], style={'width': '20%', 'display': 'inline-block', "padding": "5px"}),

    html.Div([dcc.Dropdown(
            id='method_choice',
            options=method_list,
            value="raw"
        )], style={'width': '20%', 'display': 'inline-block', "padding": "5px"}),

    # html.Div(id='display-value')
    dcc.Graph(id='line_plot_multi'),
    html.Div([dcc.Dropdown(
        id='window_choice',
        options=window_size_list,
        value=7,
    )], style={'width': '20%', 'display': 'inline-block', "padding": "5px"}), 
    dcc.Graph(id='line_plot_NLTSA'),
    
    html.Div([dcc.Dropdown(
        id='feature_choice',
        options=[{'label': i.title().replace("_", " "), 'value': i} for i in features_list],
        value="new_cases"
    )], style={'width': '20%', 'display': 'inline-block', "padding": "5px"}),

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


@app.callback(Output(component_id='line_plot_NLTSA', component_property='figure'),
              [Input(component_id='country_choice', component_property='value'),
              Input(component_id='window_choice', component_property='value'),
              Input(component_id='method_choice', component_property='value'),
              ]
              )
def graph_update_NLTSA(country_choice, window_choice, method_choice):
    # if method_choice == "raw":
    #     tmp_multi_feature_list=multi_feature_list
    # else:
    #     tmp_multi_feature_list=[f"{x}_{method_choice}" for x in multi_feature_list]
    
    tmp_multi_feature_list=multi_feature_list    
    
    tmp_df = df.loc[df["location"] == '{}'.format(country_choice), :]
    tmp_df[tmp_multi_feature_list] = tmp_df[tmp_multi_feature_list].pipe(apply_scaling)
    plot_df = (tmp_df.loc[:, ["date"] + tmp_multi_feature_list]
               .melt(id_vars="date")
               )
    
    decomps_list = [
        decomposition.DictionaryLearning,
                    decomposition.FactorAnalysis,
                    decomposition.FastICA,
                    # decomposition.IncrementalPCA,
                    # decomposition.KernelPCA,
                    decomposition.NMF,
                    decomposition.PCA
                    ]
    
    plot_df=pd.concat([summary_window_FUN(tmp_df.loc[:, multi_feature_list].interpolate(method="linear").dropna().dropna(axis=1), 
                                          window_size=window_choice, user_func=window_function,
                                          kwargs={"random_state": 42}) for window_function in decomps_list],
                      axis=1).reset_index().melt(id_vars="index")
    
    fig = px.line(
        data_frame=plot_df,
        x='index',
        y="value",
        color="variable",
        markers=True
    )
    # fig.update_traces(line_color='#743de0')

    fig.update_layout(title=f'''COVID results == {country_choice}, window size == {window_choice} days, method == raw 
                      <br><sup>Currently raw is the only supported method for the windowed NLTSA functions. Also the time series are interpolated linearly so no nans are present.</sup>''',
                      xaxis_title='Date',
                      yaxis_title="Scaled Value"
                      )
    return fig


@app.callback(Output(component_id='line_plot_multi', component_property='figure'),
              [Input(component_id='country_choice', component_property='value'),
              Input(component_id='method_choice', component_property='value')
              ]
              )
def graph_update_multi(country_choice, method_choice):
    if method_choice == "raw":
        tmp_multi_feature_list=multi_feature_list
    else:
        tmp_multi_feature_list=[f"{x}_{method_choice}" for x in multi_feature_list]
            
    tmp_df = df.loc[df["location"] == '{}'.format(country_choice), :]
    tmp_df[tmp_multi_feature_list] = tmp_df[tmp_multi_feature_list].pipe(apply_scaling)
    plot_df = (tmp_df.loc[:, ["date"] + tmp_multi_feature_list]
               .melt(id_vars="date")
               )
    fig = px.line(
        data_frame=plot_df,
        x='date',
        y="value",
        color="variable",
        markers=True
    )
    # fig.update_traces(line_color='#743de0')

    fig.update_layout(title=f'COVID results == {country_choice}, method == {method_choice}',
                      xaxis_title='Date',
                      yaxis_title="Scaled Value"
                      )
    return fig


@app.callback(Output(component_id='line_plot', component_property='figure'),
              [Input(component_id='country_choice', component_property='value'),
              Input(component_id='feature_choice', component_property='value')]
              )
def graph_update(country_choice, feature_choice):
    
    fig = px.line(
        data_frame=df.loc[df["location"] == '{}'.format(country_choice), :],
        x='date',
        y=feature_choice,
        markers=True
    )
    fig.update_traces(line_color='#743de0')

    fig.update_layout(title=f'COVID results == {country_choice}, feature choice == {feature_choice}',
                      xaxis_title='Date',
                      yaxis_title='{}'.format(feature_choice)
                      )
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
