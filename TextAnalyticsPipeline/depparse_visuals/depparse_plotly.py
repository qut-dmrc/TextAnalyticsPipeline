import networkx as nx
import plotly.graph_objects as go
import pandas as pd

# Set directory info
fpathm = 'C:/Users/vodden/PycharmProjects/TextAnalyticsPipeline/TextAnalyticsPipeline/depparse_visuals/test_files/'
post_file = 'facebook_posts.csv'
depparse_file = 'spacy_depparse.csv'

# Load post data
df_posts = pd.read_csv(fpathm + post_file)
df_posts = df_posts[['platformId', 'message']]

# Load depparse data
df_depparse = pd.read_csv(fpathm + depparse_file)

# Merge df_posts and df_depparse on platformId = identifier
df = df_posts.merge(df_depparse, how='left', left_on='platformId', right_on='identifier')

# Isolate a specific post, this one is about Pauline Hanson
df = df[df['platformId'] == '100063926926757_381321477342080']

# Isolate first sentence
df = df[df['sentence_num'] == 1]

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to G based on your DataFrame
for _, row in df.iterrows():
    G.add_edge(row['head_text'], row['word_text'], relation=row['relation'])

# Order nodes by word_num
ordered_nodes = sorted(G.nodes(), key=lambda node: df[df['word_text'] == node]['word_num'].values[0])

# Set specific positions for each node
pos = {node: (i, 0) for i, node in enumerate(ordered_nodes)}

edge_x = []
edge_y = []
edge_labels = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]

    # Specify control points for a more square path
    cx1 = x0
    cy1 = y0 + 2.0  # Adjust this value to control the height of the curve
    cx2 = x1
    cy2 = y1 + 2.0  # Adjust this value to control the height of the curve

    edge_x.extend([x0, cx1, cx2, x1, None])
    edge_y.extend([y0, cy1, cy2, y1, None])
    edge_labels.append(edge[2]['relation'])  # Collect the relation values for each edge

# Calculate midpoints for labels, handling None values
label_x = [(x0 + x1) / 2 if x0 is not None and x1 is not None else None for x0, x1 in zip(edge_x[::4], edge_x[2::4])]
label_y = [(y0 + y1) / 2 if y0 is not None and y1 is not None else None for y0, y1 in zip(edge_y[::4], edge_y[2::4])]

label_trace = go.Scatter(
    x=label_x, y=label_y,
    mode='text',
    text=edge_labels,
    textposition='middle center',
    hoverinfo='text',
    showlegend=False,
)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    hoverinfo='none',
    line=dict(width=1, color='#888'),
)

node_x = []
node_y = []
node_labels = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_labels.append(f"{node}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_labels,
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=100,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
    )
)

fig = go.Figure(data=[edge_trace, label_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

fig.update_layout(
    font=dict(
        family="Arial",
        size=18
    )
)

fig.show()
