# trustviz/ve/plotly_adapter.py
import plotly.graph_objects as go

def roc_figure_json(fpr, tpr, title, x_label, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance",
                             line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    return fig.to_json()
