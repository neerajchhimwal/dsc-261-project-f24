import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="MINE: Mutual Information Visualization",
    page_icon="📊",
    layout="wide"
)

st.title("📊 MINE: Mutual Information Neural Estimator")
st.markdown("""
    Experiment 1: Estimating Mutual Information for Multivariate Gaussian
""")

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Parameters")
    dimension = st.slider(
        'Select Dimension',
        min_value=2,
        max_value=28,
        value=2,
        step=2,
        help="Adjust the dimension of the multivariate Gaussian distribution"
    )

# sample dfs
rho_values = np.linspace(-0.99, 0.99, 15)
# dataframes = {}

# for dim in range(2, 21):
#     true_mi = -0.5 * np.log(1 - rho_values**2) * (dim/10)
#     mi_estimates = true_mi + np.random.normal(0, 0.05, len(rho_values))
    
#     df = pd.DataFrame({
#         'rho': rho_values,
#         'estimated_mi': mi_estimates,
#         'true_mi': true_mi
#     })
#     dataframes[dim] = df
dataframes = pd.read_csv('./results/df_mine_corr_gaussians_exp_2_to_30.csv')

with col1:
    def plot_mi_curves(dim):
        df = dataframes[dataframes['dim']==dim]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df['rho'], y=df['estimated_mi'], mode='lines', name='MINE', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['rho'], y=df['true_mi'], mode='lines', name='True MI', line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            xaxis_title='Correlation coefficient (ρ)',
            yaxis_title='Mutual Information',
            title=f'MINE: MI Estimation for Multivariate Gaussian (d={dim})',
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified'
        )
        
        return fig

    st.plotly_chart(plot_mi_curves(dimension), use_container_width=True)