# crosstab_eda = pd.crosstab(
#     index=train_df['channelGrouping'], normalize=True,
#     columns=train_df[train_df['device.browser'].isin(train_df['device.browser'].value_counts()[:5].index.values)]['device.browser']
# )

# crosstab_eda.plot(
#     kind="bar",
#     figsize=(14,6),
#     stacked=True
# )

# plt.title("Channel Grouping % for which Browser", fontsize=12)
# plt.xlabel("The Channel Grouping Name", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.xticks(rotation=0)

# plt.show()

# crosstab_eda = pd.crosstab(
#     index=train_df[train_df['device.operatingSystem'].isin(train_df['device.operatingSystem'].value_counts()[:6].index.values)]['device.operatingSystem'],
#     columns=train_df[train_df['device.browser'].isin(train_df['device.browser'].value_counts()[:5].index.values)]['device.browser'])

# crosstab_eda.plot(
#     kind="bar",
#     figsize=(14,6),
#     stacked=True
# )

# crosstab_eda = pd.crosstab(
#     index=train_df['device.deviceCategory'],
#     columns=train_df[train_df['device.operatingSystem'].isin(train_df['device.operatingSystem'].value_counts()[:6].index.values)]['device.operatingSystem']
# )

# crosstab_eda.plot(
#     kind="bar",
#     figsize=(14,6),
#     stacked=True
# )
# plt.title("Most frequent OS's by Device Categorys of users", fontsize=12)
# plt.xlabel("Device Name", fontsize=12)
# plt.ylabel("Count Device x OS", fontsize=12)
# plt.xticks(rotation=0)

# plt.show()

# def PieChart(df_colum, title, limit=15):
#     count_trace = train_df[df_colum].value_counts()[:limit].to_frame()
# .reset_index()
#     rev_trace = train_df.groupby(df_colum)["totals.transactionRevenue"]
# .sum().nlargest(10).to_frame().reset_index()

#     trace1 = go.Pie(
#         labels=count_trace['index'],
#         values=count_trace[df_colum],
#         name= "% Acesses",
#         hole= .5,
#         hoverinfo="label+percent+name",
#         showlegend=True,
#         domain= {'x': [0, .48]},
#         marker=dict(colors=color)
#     )

#     trace2 = go.Pie(
#         labels=rev_trace[df_colum],
#         values=rev_trace['totals.transactionRevenue'],
#         name="% Revenue",
#         hole= .5,
#         hoverinfo="label+percent+name",
#         showlegend=False,
#         domain= {'x': [.52, 1]}
#     )

#     layout = dict(
#         title= title,
#         height=450,
#         font=dict(size=15),
#         annotations = [
#             dict(x=.25, y=.5, text='Visits', showarrow=False,
# font=dict(size=20)),
#             dict(x=.80, y=.5, text='Revenue', showarrow=False,
#  font=dict(size=20))
#         ]
#     )

#     fig = dict(data=[trace1, trace2], layout=layout)
#     iplot(fig)

# plt.title("Most frequent OS's by Browsers of users", fontsize=12)
# plt.xlabel("Operational System Name", fontsize=12)
# plt.ylabel("Count OS", fontsize=12)
# plt.xticks(rotation=0)

# plt.show()

# https://www.kaggle.com/kabure/exploring-the-consumer-patterns-ml-pipeline
