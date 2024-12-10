import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

experiment1_data = pd.read_csv('./results/df_mine_corr_gaussians_exp_2_to_30.csv')
experiment2_data = pd.read_csv('./results/Results_exp_2_correct_format.csv')

# page config
st.set_page_config(
    page_title="MINE: Mutual Information & CI Testing",
    page_icon="📊",
    layout="wide"
)

st.title("📊 MINE: Mutual Information & Conditional Independence Testing")

# experiments
# experiment = st.sidebar.selectbox(
#     "Select Experiment",
#     ["Experiment 1: MI for Multivariate Gaussian", "Experiment 2: High-dimensional CI Testing"]
# )

st.sidebar.markdown("## Select Experiments to Display")
show_exp1 = st.sidebar.checkbox("Experiment 1: MI for Multivariate Gaussian", value=True)
show_exp2 = st.sidebar.checkbox("Experiment 2: High-dimensional CI Testing", value=True)


# exp 1: MI for Multivariate Gaussian
# if experiment == "Experiment 1: MI for Multivariate Gaussian":
if show_exp1:
    st.markdown("""
        ## Experiment 1: Estimating Mutual Information for Multivariate Gaussian
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

    with col1:
        def plot_mi_curves(dim):
            df = experiment1_data[experiment1_data['dim'] == dim]

            fig = px.line(
                df,
                x='rho',
                y=['estimated_mi', 'true_mi'],
                labels={'value': 'Mutual Information', 'variable': 'Type'},
                title=f"MINE: MI Estimation for Multivariate Gaussian (d={dim})"
            )
            return fig

        st.plotly_chart(plot_mi_curves(dimension), use_container_width=True)

# exp 2: CI Testing
def plot_experiment2_facet(data):
    import plotly.express as px
    
    fig = px.line(
        data,
        x='Num_Features',
        y='CMI_Estimate',
        color='Corr_Strength',
        facet_col='Num_Samples',
        markers=True,
        labels={
            'Num_Features': 'Number of Features',
            'CMI_Estimate': 'CMI Estimate',
            'Corr_Strength': 'Correlation Strength',
            'Num_Samples': 'Number of Samples'
        },
        title='CMI Estimates Across Number of Features and Correlation Strengths'
    )

    fig.update_layout(
        legend_title="Correlation Strength",
        margin=dict(t=50, b=50, l=50, r=50),
        height=600,
        width=1000
    )
    return fig

# if experiment == "Experiment 2: High-dimensional CI Testing":
if show_exp2:
    st.markdown("""
        ## Experiment 2: High-dimensional Conditional Independence Testing Using CMI
    """)
    st.plotly_chart(plot_experiment2_facet(experiment2_data), use_container_width=True)
