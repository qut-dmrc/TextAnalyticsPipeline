import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[0.5, 0.5],
    y=[0.7, 0.9],
    mode='lines+text',
    text=['Pauline', 'is', 'coming', 'to', 'WA'],
    textposition='bottom center',
    line=dict(color='black', width=2)
))

fig.add_trace(go.Scatter(
    x=[0.2, 0.8],
    y=[0.6, 0.6],
    mode='lines+text',
    text=['', '', '', '', ''],
    textposition='bottom center',
    line=dict(color='black', width=2)
))

fig.add_trace(go.Scatter(
    x=[0.2, 0.8],
    y=[0.6, 0.6],
    mode='lines',
    line=dict(color='black', width=2)
))

fig.update_layout(
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    margin=dict(l=0, r=0, b=0, t=0)
)

fig.show()
