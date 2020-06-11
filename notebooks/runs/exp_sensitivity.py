import plotly.express as px
import pandas as pd
import numpy as np

pdf = pd.read_csv("exp_hypers_sensitivity.csv", index_col=0)
pdf['mean_squared_error'] = np.log10(pdf.mean_squared_error)

labels = {
    "seed": "Model Seed",
    "g_lr": "G Learning Rate",
    "d_lr": "D Learning Rate",
    "mean_squared_error": "Log MSE",
}

dims = ['seed', 'g_lr', 'd_lr', 'mean_squared_error']

fig = px.parallel_coordinates(
    pdf, 
    dimensions=dims,
    color=pdf['seed'].astype('category').cat.codes,
    labels=labels,
    color_continuous_scale=px.colors.sequential.Rainbow,
)

fig.update_layout(
    font=dict(
        size=30,
        color='black',
    ),
    coloraxis_colorbar=dict(title="Seed",)
    
#     margin=dict(l=80, r=80, t=100, b=90)
#     autosize=True,
#     margin=dict(l=20, r=20, t=20, b=20),
#     height=800,
#     width=1200,
)

# fig.update_layout()

fig.update_yaxes(showticklabels=False)
fig.show()