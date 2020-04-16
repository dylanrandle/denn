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

fig = px.parallel_coordinates(pdf, 
                              dimensions=['seed', 'g_lr', 'd_lr', 'mean_squared_error'],
                              color=pdf['seed'].astype('category').cat.codes,
                              labels=labels,
                              color_continuous_scale=px.colors.sequential.Rainbow,)

fig.update_layout(
    font=dict(
        size=15,
        color='black'
    )
)

fig.update_layout(coloraxis_colorbar=dict(title="Seed",))

fig.show()