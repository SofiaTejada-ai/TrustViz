# trustviz/ve/plotly_adapter.py
import json

def roc_figure_json(fpr, tpr, title="ROC Curve",
                    x_label="False Positive Rate",
                    y_label="True Positive Rate"):
    fig = {
        "data": [
            {"type":"scatter","mode":"lines","name":"ROC","x":list(fpr),"y":list(tpr)},
            {"type":"scatter","mode":"lines","name":"Chance","x":[0,1],"y":[0,1],"line":{"dash":"dash"}},
        ],
        "layout": {
            "title":{"text": title},
            "xaxis":{"title":{"text": x_label}, "range":[0,1]},
            "yaxis":{"title":{"text": y_label}, "range":[0,1]},
            "margin":{"l":60,"r":10,"t":50,"b":60},
            "legend":{"orientation":"h","x":0.02,"y":-0.2},
        },
    }
    return json.dumps(fig)

def bar_figure_json(x, y, title, x_label, y_label):
    fig = {
        "data":[{"type":"bar","x":list(x),"y":list(y),"name":"values"}],
        "layout":{
            "title":{"text": title},
            "xaxis":{"title":{"text": x_label}},
            "yaxis":{"title":{"text": y_label}},
            "margin":{"l":60,"r":10,"t":50,"b":60},
            "legend":{"orientation":"h","x":0.02,"y":-0.2},
        },
    }
    return json.dumps(fig)

def pie_figure_json(labels, values, title):
    fig = {
        "data":[{"type":"pie","labels":list(labels),"values":list(values),"name":"parts"}],
        "layout":{"title":{"text": title}},
    }
    return json.dumps(fig)

def line_figure_json(x, y, title, x_label, y_label):
    fig = {
        "data":[{"type":"scatter","mode":"lines+markers","x":list(x),"y":list(y),"name":"series"}],
        "layout":{
            "title":{"text": title},
            "xaxis":{"title":{"text": x_label}},
            "yaxis":{"title":{"text": y_label}},
            "margin":{"l":60,"r":10,"t":50,"b":60},
        },
    }
    return json.dumps(fig)
