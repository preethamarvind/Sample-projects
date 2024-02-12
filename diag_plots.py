#Importing basic libraries
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot


#Creating a helper function to bin the data
def BinData(data, var, n_bins):
    #getting the labels of the bins
    temp_perc = data[var].quantile(np.arange(0, 1+(1/n_bins), 1/n_bins)).to_numpy()
    labels = (temp_perc[1:] + temp_perc[:-1])/2
    labels = labels.round(2)
    
    #cut data into 20 bins
    data[var + '_bin'] = pd.cut(data[var], bins = temp_perc, labels = labels, include_lowest = True)
    return data

#Creating a dummy dataset
df2 = pd.DataFrame(np.random.randn(1000, 4), columns=['z' + str(i+1) for i in range(4)])
df2['k'] = 3*df2.z1 + np.random.randn(1000)
df2['k_pred'] = 2.5*df2.z1 + 0.5*df2.z2

#### AvsE function ###### 
#Purpose: This function is designed to create a model-agnostic simple Actual versus Predicted plot to help us visualise
#         what the model is doing. It also includes an exposure chart to give us an idea of the credibility behind each point. (Currently,
#         the binning only works as equal sized bins, so less interesting.)
#Inputs:  indata - The dataset to be used
#         prim_var_sel - The primary variable to be binned and plotted
#         target_var_sel - The target variable to be plotted
#         pred_var_sel - The predicted variable to be plotted
#         n_bins - The number of bins to be used for the primary variable
#Outputs: A plotly plot of the AvsE chart
#Example: AvsEPlot(indata = df2, prim_var_sel = 'z1', target_var_sel = 'k', pred_var_sel = 'k_pred', n_bins = 20)

def AvsEPlot(indata, prim_var_sel, target_var_sel, pred_var_sel, n_bins = 20):
    
    #Binning by the primary variable
    BinData(indata, prim_var_sel, n_bins)

    #Summarising by the primary variable
    df_sum = indata.groupby(prim_var_sel + '_bin', observed = True).agg(N = (prim_var_sel, 'count'),
                                            target_var = (target_var_sel, 'mean'),
                                            pred_var = (pred_var_sel, 'mean')).reset_index()

    #Melting the data to a long format
    df_sum = df_sum.melt(id_vars = prim_var_sel + '_bin', value_vars = ['N', 'target_var', 'pred_var'])


    #Create a line plot using plotly
    figures = [
            px.line(data_frame = df_sum.loc[df_sum['variable'] != 'N'], x = prim_var_sel + '_bin', y = 'value', color = 'variable', markers = True),
            px.bar(data_frame=df_sum.loc[df_sum['variable'] == 'N'], x = prim_var_sel + '_bin', y = 'value')
        ]

    fig = make_subplots(rows=len(figures), cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('AvsE Chart', 'Exposure Chart')) 

    for i, figure in enumerate(figures):
        for trace in range(len(figure["data"])):
            fig.append_trace(figure["data"][trace], row=i+1, col=1)

    fig.update_layout(
        title_text="AvsE plot for " + prim_var_sel
    )

    # Update xaxis properties if needed
    fig.update_xaxes(title_text=prim_var_sel, row=2, col=1)

    # Update yaxis properties for each subplot if needed
    fig.update_yaxes(title_text="Actual/Pred", row=1, col=1)
    fig.update_yaxes(title_text="Exposure", row=2, col=1)
    fig.update_legends(title_text="Exposure", row=2, col=1)
    fig.update_legends()

    #Plotting the final figure
    plot(fig)

    #Removing created columns\
    indata.drop(columns=prim_var_sel+ "_bin" , inplace = True)

#Testing out the function
AvsEPlot(indata = df2, prim_var_sel = 'z1', target_var_sel = 'k', pred_var_sel = 'k_pred', n_bins = 20)