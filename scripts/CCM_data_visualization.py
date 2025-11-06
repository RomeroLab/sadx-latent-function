import argparse
import json
import os
from datetime import datetime
from os.path import join
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from utils import load_yaml_config, log_func
import plotly.graph_objects as go
plt.rcParams['font.family'] = 'serif'  # Change 'serif' to your desired font family
plt.rcParams['font.serif'] = ['Times New Roman']  # Example: specify a font
plt.rcParams['font.size'] = 18  # Adjust the font size as needed


def visualize_ccm(config,out_dir):
    df =pd.read_excel(config['raw_df_fn'])
    df['variant_label'] = df['Name'].apply(lambda x:x.split('_')[0])
    products=['NSL-OH1/Area','NSL-OH2/Area','NSL-N3/Area','NSL/Area']
    assert np.all(df[products].sum(axis=1)==df['Total'])
    for product in products:
        id=product.split("/")[0]
        assert np.all(df[f"%{id}"]-((df[product]/df["Total"])*100)<1e-6)

    # fig = px.scatter_ternary(
    #     df, a="%NSL-N3", b="%NSL-OH2", c="%NSL-OH1",
    #     symbol="variant_label",
    #     title="Composition of 3 samples per datapoint"
    # )
    # fig.show()

    # fig = px.parallel_coordinates(
    #     df,
    #     # color="datapoint",  # each datapoint group has 3 samples (lines)
    #     dimensions=["%NSL-OH1", "%NSL-OH2", "%NSL-N3", "%NSL"],
    #     labels={
    #         "%NSL-OH1": "OH1",
    #         "%NSL-OH2": "OH2",
    #         "%NSL-N3": "N3",
    #         "%NSL": "NSL"
    #     },
    #     title="Parallel coordinates of variant compositions"
    # )
    # fig.show()


    # measure_cols = ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL']
    #
    # # add inferred replicate number
    # df_rep = df.copy()
    # df_rep['replicate_id'] = df_rep.groupby('variant_label').cumcount() + 1
    #
    # # combine variant and replicate for x labels
    # df_rep['variant_replicate'] = (
    #         df_rep['variant_label'].astype(str) + '_r' + df_rep['replicate_id'].astype(str)
    # )
    #
    # # reshape to long form
    # df_long = df_rep.melt(
    #     id_vars=['variant_label', 'variant_replicate'],
    #     value_vars=measure_cols,
    #     var_name='measurement',
    #     value_name='percent'
    # )
    #
    # # order x-axis as variant groups with r1–r3 sequence
    # x_order = (
    #     df_long.groupby(['variant_label', 'variant_replicate'])
    #     .size()
    #     .reset_index()['variant_replicate']
    #     .tolist()
    # )
    #
    # # interactive stacked bar
    # fig = px.bar(
    #     df_long,
    #     x='variant_replicate',
    #     y='percent',
    #     color='measurement',
    #     barmode='stack',
    #     category_orders={'measurement': measure_cols, 'variant_replicate': x_order},
    #     hover_data=['variant_label', 'measurement', 'percent'],
    #     title='NSL Composition per Sample (All Variants on One Axis)',
    # )
    #
    # fig.update_yaxes(range=[0, 100], title='Percent')
    # fig.update_xaxes(title='Variant Replicates', tickangle=-45)
    #
    # # subtle spacing between variant groups
    # fig.update_layout(
    #     legend_title_text='Measurement',
    #     bargap=0.25,
    #     xaxis=dict(showgrid=False),
    #     plot_bgcolor='white'
    # )
    #
    # fig.show()


    # ---- Setup ----
    # measure_cols = ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL']
    #
    # df_rep = df.copy()
    # # infer replicate index within each variant (since there isn't a column for it)
    # df_rep['replicate_id'] = df_rep.groupby('variant_label').cumcount() + 1
    # df_rep['variant_replicate'] = df_rep['variant_label'].astype(str) + '_r' + df_rep['replicate_id'].astype(str)
    #
    # # long form for stacked bars
    # df_long = df_rep.melt(
    #     id_vars=['variant_label', 'replicate_id', 'variant_replicate'],
    #     value_vars=measure_cols,
    #     var_name='measurement',
    #     value_name='percent'
    # )
    #
    # # helper: list all x labels per variant in r1→rN order
    # def cat_array_for_variant_order(variant_order: list[str]) -> list[str]:
    #     cat = []
    #     for v in variant_order:
    #         reps = (df_rep.loc[df_rep['variant_label'] == v, 'replicate_id']
    #                 .sort_values().tolist())
    #         cat.extend([f'{v}_r{r}' for r in reps])
    #     return cat
    #
    # # helper: sort variants by mean of one or more columns (descending)
    # def variant_order_by(cols):
    #     if isinstance(cols, str):
    #         cols = [cols]
    #     scores = df_rep.groupby('variant_label')[cols].mean().sum(axis=1)
    #     return scores.sort_values(ascending=False).index.tolist()
    #
    # # original order (as they appear)
    # original_variant_order = df_rep['variant_label'].drop_duplicates().tolist()
    # original_cat_array = cat_array_for_variant_order(original_variant_order)
    #
    # # ---- Base figure (all data present) ----
    # fig = px.bar(
    #     df_long,
    #     x='variant_replicate',
    #     y='percent',
    #     color='measurement',
    #     barmode='stack',
    #     category_orders={
    #         'measurement': measure_cols,
    #         'variant_replicate': original_cat_array
    #     },
    #     title='NSL Composition per Sample (use dropdown to sort variants)'
    # )
    #
    # fig.update_yaxes(range=[0, 100], title='Percent')
    # fig.update_xaxes(
    #     title='Variant (Replicate)',
    #     tickangle=-45,
    #     categoryorder='array',
    #     categoryarray=original_cat_array
    # )
    # fig.update_layout(legend_title_text='Measurement', bargap=0.25, plot_bgcolor='white')
    #
    # # ---- Dropdown: only relayout x-axis order; never touch data/traces ----
    # buttons = []
    #
    # # Reset
    # buttons.append(dict(
    #     label='Original order',
    #     method='relayout',
    #     args=[{
    #         'xaxis.categoryorder': 'array',
    #         'xaxis.categoryarray': original_cat_array,
    #         'title.text': 'NSL Composition per Sample (use dropdown to sort variants)'
    #     }]
    # ))
    #
    # # Single-metric sorts
    # for col in measure_cols:
    #     vorder = variant_order_by(col)
    #     cat_array = cat_array_for_variant_order(vorder)
    #     buttons.append(dict(
    #         label=f'Sort by {col} (mean)',
    #         method='relayout',
    #         args=[{
    #             'xaxis.categoryorder': 'array',
    #             'xaxis.categoryarray': cat_array,
    #             'title.text': f'NSL Composition per Sample — sorted by mean {col}'
    #         }]
    #     ))
    #
    # # Combo example: %NSL-OH1 + %NSL-OH2
    # combo_cols = ['%NSL-OH1', '%NSL-OH2']
    # vorder = variant_order_by(combo_cols)
    # cat_array = cat_array_for_variant_order(vorder)
    # buttons.append(dict(
    #     label='Sort by %NSL-OH1 + %NSL-OH2 (mean)',
    #     method='relayout',
    #     args=[{
    #         'xaxis.categoryorder': 'array',
    #         'xaxis.categoryarray': cat_array,
    #         'title.text': 'NSL Composition per Sample — sorted by mean (%NSL-OH1 + %NSL-OH2)'
    #     }]
    # ))
    #
    # fig.update_layout(
    #     updatemenus=[dict(
    #         buttons=buttons,
    #         direction='down',
    #         showactive=True,
    #         x=1.02, y=1.15, xanchor='right', yanchor='top'
    #     )]
    # )
    #
    # fig.show()
    # import pandas as pd
    # import plotly.express as px
    #

    # measure_cols = ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL']
    #
    # df_rep = df.copy()
    # df_rep['replicate_id'] = df_rep.groupby('variant_label').cumcount() + 1
    # df_rep['variant_replicate'] = df_rep['variant_label'] + '_r' + df_rep['replicate_id'].astype(str)
    #
    # df_long = df_rep.melt(
    #     id_vars=['variant_label', 'replicate_id', 'variant_replicate'],
    #     value_vars=measure_cols,
    #     var_name='measurement',
    #     value_name='percent'
    # )
    #
    # def variant_order_by(cols):
    #     if isinstance(cols, str):
    #         cols = [cols]
    #     scores = df_rep.groupby('variant_label')[cols].mean().sum(axis=1)
    #     return scores.sort_values(ascending=False).index.tolist()
    #
    # def cat_array_for_variant_order(vorder):
    #     cat = []
    #     for v in vorder:
    #         reps = df_rep.loc[df_rep['variant_label'] == v, 'replicate_id'].sort_values()
    #         cat += [f'{v}_r{r}' for r in reps]
    #     return cat
    #
    # def add_mean_lines(fig, cols, color="black"):
    #     means = df_rep.groupby('variant_label')[cols].mean().sum(axis=1)
    #     for variant, avg in means.items():
    #         reps = df_rep[df_rep['variant_label'] == variant]
    #         x_vals = reps['variant_replicate'].tolist()
    #         if not x_vals:
    #             continue
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=[x_vals[0], x_vals[-1]],
    #                 y=[avg, avg],
    #                 mode='lines',
    #                 line=dict(color=color, width=2, dash='dot'),
    #                 name=f"{variant} mean ({' + '.join(cols)})",
    #                 hoverinfo='skip'
    #             )
    #         )
    #
    # # ---- base plot ----
    # original_order = df_rep['variant_label'].drop_duplicates().tolist()
    # cat_array = cat_array_for_variant_order(original_order)
    #
    # fig = px.bar(
    #     df_long,
    #     x='variant_replicate',
    #     y='percent',
    #     color='measurement',
    #     barmode='stack',
    #     category_orders={'measurement': measure_cols, 'variant_replicate': cat_array},
    #     title='NSL Composition per Sample (use dropdown to sort)',
    # )
    #
    # fig.update_yaxes(range=[0, 100], title='Percent')
    # fig.update_xaxes(title='Variant (Replicates)', tickangle=-45,
    #                  categoryorder='array', categoryarray=cat_array)
    # fig.update_layout(legend_title_text='Measurement', bargap=0.25, plot_bgcolor='white')
    #
    # # ---- dropdown to reorder + add mean lines ----
    # buttons = []
    #
    # # Original
    # buttons.append(dict(
    #     label='Original order',
    #     method='relayout',
    #     args=[{
    #         'xaxis.categoryorder': 'array',
    #         'xaxis.categoryarray': cat_array,
    #         'title.text': 'NSL Composition per Sample (use dropdown to sort)'
    #     }]
    # ))
    #
    # # Individual metrics + combo
    # for col in measure_cols + ['%NSL-OH1+%NSL-OH2']:
    #     cols = [col] if '+' not in col else ['%NSL-OH1', '%NSL-OH2']
    #     vorder = variant_order_by(cols)
    #     cat_arr = cat_array_for_variant_order(vorder)
    #     fig.add_trace(go.Scatter())  # placeholder for consistency
    #     buttons.append(dict(
    #         label=f"Sort by {col}",
    #         method='update',
    #         args=[
    #             {},
    #             {
    #                 'xaxis.categoryorder': 'array',
    #                 'xaxis.categoryarray': cat_arr,
    #                 'title.text': f'Sorted by mean of {col}'
    #             }
    #         ]
    #     ))
    #
    # fig.update_layout(
    #     updatemenus=[dict(
    #         buttons=buttons,
    #         direction='down',
    #         showactive=True,
    #         x=1.02, y=1.15, xanchor='right', yanchor='top'
    #     )]
    # )
    #
    # # Add mean lines for the default order (none for "original")
    # add_mean_lines(fig, ['%NSL-N3'], color="black")
    #
    # fig.show()
    # import pandas as pd
    # import plotly.graph_objects as go

    # # --- inputs ---
    # measure_cols = ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL']
    #
    # # df: columns = ['variant_label', '%NSL-OH1','%NSL-OH2','%NSL-N3','%NSL']
    # df_rep = df.copy()
    # df_rep['replicate_id'] = df_rep.groupby('variant_label').cumcount() + 1
    # df_rep['variant_replicate'] = df_rep['variant_label'].astype(str) + '_r' + df_rep['replicate_id'].astype(str)
    #
    # # long -> wide for quick trace building
    # pivot = (
    #     df_rep.melt(id_vars=['variant_label', 'variant_replicate'],
    #                 value_vars=measure_cols,
    #                 var_name='measurement', value_name='percent')
    #     .pivot_table(index=['variant_label', 'variant_replicate'],
    #                  columns='measurement', values='percent', aggfunc='first')
    #     .reset_index()
    # )
    #
    # # helpers
    # def variant_order_by(cols):
    #     if isinstance(cols, str): cols = [cols]
    #     scores = df_rep.groupby('variant_label')[cols].mean().sum(axis=1)
    #     return scores.sort_values(ascending=False).index.tolist()
    #
    # def cat_array_for_variant_order(vorder):
    #     # r1..rN order within each variant
    #     cat = []
    #     for v in vorder:
    #         reps = (df_rep.loc[df_rep['variant_label'] == v, 'replicate_id'].sort_values())
    #         cat.extend([f'{v}_r{r}' for r in reps])
    #     return cat
    #
    # def make_shapes_for_metric(cols, vorder):
    #     # draw horizontal mean line(s) across each variant’s replicate span
    #     shapes = []
    #     if isinstance(cols, str): cols = [cols]
    #     for v in vorder:
    #         reps = sorted(df_rep.loc[df_rep['variant_label'] == v, 'replicate_id'].tolist())
    #         if not reps: continue
    #         x0 = f'{v}_r{reps[0]}'
    #         x1 = f'{v}_r{reps[-1]}'
    #         # One line per metric (pairs draw two lines)
    #         for i, c in enumerate(cols):
    #             avg_val = df_rep.loc[df_rep['variant_label'] == v, c].mean()
    #             shapes.append(dict(
    #                 type='line',
    #                 xref='x', yref='y',
    #                 x0=x0, x1=x1, y0=avg_val, y1=avg_val,
    #                 line=dict(width=2, dash='dot' if i == 0 else 'dash', color='black'),
    #                 layer='above'
    #             ))
    #     return shapes
    #
    # def build_bar_traces(measure_order):
    #     # traces in the given order => bottom-to-top stacking order
    #     traces = []
    #     xvals = pivot['variant_replicate']
    #     for m in measure_order:
    #         traces.append(go.Bar(
    #             x=xvals,
    #             y=pivot[m],
    #             name=m,
    #             offsetgroup='g',  # keep all sets stacking aligned
    #             hovertemplate='%{x}<br>' + m + ': %{y:.2f}%<extra></extra>'
    #         ))
    #     return traces
    #
    # # modes: measurement orders
    # modes = {
    #     'Original order': measure_cols,  # default
    #     'Sort by %NSL-N3': ['%NSL-N3', '%NSL-OH1', '%NSL-OH2', '%NSL'],
    #     'Sort by %NSL-OH1': ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL'],
    #     'Sort by %NSL-OH2': ['%NSL-OH2', '%NSL-OH1', '%NSL-N3', '%NSL'],
    #     'Sort by %NSL': ['%NSL', '%NSL-OH1', '%NSL-OH2', '%NSL-N3'],
    #     'Sort by %NSL-OH1 + %NSL-OH2': ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL'],  # pair on bottom
    # }
    #
    # # prebuild all trace sets
    # fig = go.Figure()
    # trace_sets = {}  # mode -> list of trace indices
    # start = 0
    # for mode, order in modes.items():
    #     traces = build_bar_traces(order)
    #     idxs = list(range(start, start + len(traces)))
    #     start += len(traces)
    #     trace_sets[mode] = idxs
    #     for i, t in zip(idxs, traces):
    #         # only original visible initially
    #         t.visible = (mode == 'Original order')
    #         fig.add_trace(t)
    #
    # # initial layout
    # original_vorder = df_rep['variant_label'].drop_duplicates().tolist()
    # original_cat = cat_array_for_variant_order(original_vorder)
    #
    # fig.update_layout(
    #     barmode='stack',
    #     bargap=0.25,
    #     plot_bgcolor='white',
    #     legend_title_text='Measurement',
    #     title='NSL Composition per Sample (use dropdown to sort)'
    # )
    # fig.update_yaxes(range=[0, 100], title='Percent')
    # fig.update_xaxes(title='Variant (Replicate)',
    #                  tickangle=-45,
    #                  categoryorder='array',
    #                  categoryarray=original_cat)
    #
    # # buttons: toggle visibility of trace sets + relayout x-order + shapes (mean line[s])
    # buttons = []
    #
    # # Original (no mean lines)
    # vis = [False] * len(fig.data)
    # for i in trace_sets['Original order']: vis[i] = True
    # buttons.append(dict(
    #     label='Original order',
    #     method='update',
    #     args=[
    #         {'visible': vis},
    #         {
    #             'xaxis.categoryorder': 'array',
    #             'xaxis.categoryarray': original_cat,
    #             'title.text': 'NSL Composition per Sample (use dropdown to sort)',
    #             'shapes': []  # no mean lines
    #         }
    #     ]
    # ))
    #
    # # Single-metric sorts
    # for label in ['%NSL-N3', '%NSL-OH1', '%NSL-OH2', '%NSL']:
    #     mode = f'Sort by {label}'
    #     vorder = variant_order_by(label)
    #     cat = cat_array_for_variant_order(vorder)
    #     shapes = make_shapes_for_metric(label, vorder)
    #     vis = [False] * len(fig.data)
    #     for i in trace_sets[mode]: vis[i] = True  # show only this set to put chosen metric at bottom
    #     buttons.append(dict(
    #         label=mode,
    #         method='update',
    #         args=[
    #             {'visible': vis},
    #             {
    #                 'xaxis.categoryorder': 'array',
    #                 'xaxis.categoryarray': cat,
    #                 'title.text': f'NSL Composition per Sample — sorted by mean {label}',
    #                 'shapes': shapes  # mean line for that metric
    #             }
    #         ]
    #     ))
    #
    # # Pair sort: %NSL-OH1 + %NSL-OH2 (both lines)
    # pair_label = 'Sort by %NSL-OH1 + %NSL-OH2'
    # vorder = variant_order_by(['%NSL-OH1', '%NSL-OH2'])
    # cat = cat_array_for_variant_order(vorder)
    # shapes = make_shapes_for_metric(['%NSL-OH1', '%NSL-OH2'], vorder)
    # vis = [False] * len(fig.data)
    # for i in trace_sets[pair_label]: vis[i] = True
    # buttons.append(dict(
    #     label=pair_label,
    #     method='update',
    #     args=[
    #         {'visible': vis},
    #         {
    #             'xaxis.categoryorder': 'array',
    #             'xaxis.categoryarray': cat,
    #             'title.text': 'NSL Composition per Sample — sorted by mean (%NSL-OH1 + %NSL-OH2)',
    #             'shapes': shapes
    #         }
    #     ]
    # ))
    #
    # fig.update_layout(
    #     updatemenus=[dict(
    #         buttons=buttons,
    #         direction='down',
    #         showactive=True,
    #         x=1.02, y=1.15, xanchor='right', yanchor='top'
    #     )]
    # )
    #
    # fig.show()
    # import pandas as pd
    # import plotly.graph_objects as go

    if config.get('stacked',False):
        # ---- Inputs ----
        measure_cols = ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL']

        # Consistent colors across all modes
        color_map = {
            '%NSL-OH1': '#1f77b4',  # blue
            '%NSL-OH2': '#ff7f0e',  # orange
            '%NSL-N3': '#2ca02c',  # green
            '%NSL': '#d62728',  # red
        }

        # Example structure: df must have these columns
        # df = pd.DataFrame({
        #     "variant_label": ["WT", "WT", "WT", "Mut1", "Mut1", "Mut1"],
        #     "%NSL-OH1": [10, 12, 11, 5, 6, 7],
        #     "%NSL-OH2": [30, 28, 32, 20, 18, 19],
        #     "%NSL-N3":  [40, 42, 39, 55, 56, 54],
        #     "%NSL":     [20, 18, 18, 20, 20, 20]
        # })

        df_rep = df.copy()
        df_rep['replicate_id'] = df_rep.groupby('variant_label').cumcount() + 1
        df_rep['variant_replicate'] = df_rep['variant_label'] + '_r' + df_rep['replicate_id'].astype(str)

        pivot = (
            df_rep.melt(id_vars=['variant_label', 'variant_replicate'],
                        value_vars=measure_cols,
                        var_name='measurement', value_name='percent')
            .pivot_table(index=['variant_label', 'variant_replicate'],
                         columns='measurement', values='percent', aggfunc='first')
            .reset_index()
        )

        # ---- Helpers ----
        def variant_order_by(cols):
            """Return variant order sorted by mean of selected columns (sum if multiple)."""
            if isinstance(cols, str):
                cols = [cols]
            tmp = df_rep.copy()
            tmp[cols] = tmp[cols].apply(pd.to_numeric, errors='coerce')
            means = tmp.groupby('variant_label')[cols].mean()  # mean per variant
            scores = means.sum(axis=1)  # sum across columns if multiple
            return scores.sort_values(ascending=False).index.tolist()

        def cat_array_for_variant_order(vorder):
            cat = []
            for v in vorder:
                reps = df_rep.loc[df_rep['variant_label'] == v, 'replicate_id'].sort_values()
                cat.extend([f'{v}_r{r}' for r in reps])
            return cat

        def mean_line_shapes(cols, vorder, color='black'):
            """Draw one line per variant = mean of (sum of cols) across replicates."""
            if isinstance(cols, str):
                cols = [cols]
            shapes = []
            for v in vorder:
                reps = sorted(df_rep.loc[df_rep['variant_label'] == v, 'replicate_id'].tolist())
                if not reps:
                    continue
                x0, x1 = f'{v}_r{reps[0]}', f'{v}_r{reps[-1]}'
                avg_val = df_rep.loc[df_rep['variant_label'] == v, cols].sum(axis=1).mean()
                shapes.append(dict(
                    type='line', xref='x', yref='y',
                    x0=x0, x1=x1, y0=avg_val, y1=avg_val,
                    line=dict(width=2, dash='dot', color=color),
                    layer='above'
                ))
            return shapes

        def build_bar_traces(order):
            """Build stacked bar traces in the given measurement order."""
            traces = []
            for m in order:
                traces.append(go.Bar(
                    x=pivot['variant_replicate'],
                    y=pivot[m],
                    name=m,
                    marker_color=color_map[m],
                    hovertemplate='%{x}<br>' + m + ': %{y:.2f}%<extra></extra>'
                ))
            return traces

        # ---- Setup plot ----
        modes = {
            'Original order': measure_cols,
            'Sort by %NSL-N3': ['%NSL-N3', '%NSL-OH1', '%NSL-OH2', '%NSL'],
            'Sort by %NSL-OH1': ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL'],
            'Sort by %NSL-OH2': ['%NSL-OH2', '%NSL-OH1', '%NSL-N3', '%NSL'],
            'Sort by %NSL': ['%NSL', '%NSL-OH1', '%NSL-OH2', '%NSL-N3'],
            'Sort by %NSL-OH1 + %NSL-OH2': ['%NSL-OH1', '%NSL-OH2', '%NSL-N3', '%NSL']
        }

        fig = go.Figure()
        trace_sets = {}
        start = 0
        for mode, order in modes.items():
            traces = build_bar_traces(order)
            idxs = list(range(start, start + len(traces)))
            start += len(traces)
            trace_sets[mode] = idxs
            for t in traces:
                t.visible = (mode == 'Original order')
                fig.add_trace(t)

        original_vorder = df_rep['variant_label'].drop_duplicates().tolist()
        original_cat = cat_array_for_variant_order(original_vorder)

        fig.update_layout(
            barmode='stack',
            bargap=0.25,
            plot_bgcolor='white',
            legend_title_text='Measurement',
            title='NSL Composition per Sample (use dropdown to sort)'
        )
        fig.update_yaxes(range=[0, 100], title='Percent')
        fig.update_xaxes(title='Variant (Replicate)', tickangle=-45,
                         categoryorder='array', categoryarray=original_cat)

        # ---- Dropdown buttons ----
        buttons = []

        # Original
        vis = [False] * len(fig.data)
        for i in trace_sets['Original order']:
            vis[i] = True
        buttons.append(dict(
            label='Original order',
            method='update',
            args=[
                {'visible': vis},
                {
                    'xaxis.categoryorder': 'array',
                    'xaxis.categoryarray': original_cat,
                    'title.text': 'NSL Composition per Sample (use dropdown to sort)',
                    'shapes': []
                }
            ]
        ))

        # Single metrics
        for label in ['%NSL-N3', '%NSL-OH1', '%NSL-OH2', '%NSL']:
            mode = f'Sort by {label}'
            vorder = variant_order_by(label)
            cat = cat_array_for_variant_order(vorder)
            shapes = mean_line_shapes(label, vorder)
            vis = [False] * len(fig.data)
            for i in trace_sets[mode]:
                vis[i] = True
            buttons.append(dict(
                label=mode,
                method='update',
                args=[
                    {'visible': vis},
                    {
                        'xaxis.categoryorder': 'array',
                        'xaxis.categoryarray': cat,
                        'title.text': f'NSL Composition per Sample — sorted by mean {label}',
                        'shapes': shapes
                    }
                ]
            ))

        # Pair
        pair_label = 'Sort by %NSL-OH1 + %NSL-OH2'
        vorder = variant_order_by(['%NSL-OH1', '%NSL-OH2'])
        cat = cat_array_for_variant_order(vorder)
        shapes = mean_line_shapes(['%NSL-OH1', '%NSL-OH2'], vorder)
        vis = [False] * len(fig.data)
        for i in trace_sets[pair_label]:
            vis[i] = True
        buttons.append(dict(
            label=pair_label,
            method='update',
            args=[
                {'visible': vis},
                {
                    'xaxis.categoryorder': 'array',
                    'xaxis.categoryarray': cat,
                    'title.text': 'NSL Composition per Sample — sorted by mean (%NSL-OH1 + %NSL-OH2)',
                    'shapes': shapes
                }
            ]
        ))

        fig.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction='down',
                showactive=True,
                x=1.02, y=1.15,
                xanchor='right', yanchor='top'
            )]
        )

        fig.write_html(join(out_dir,"ccm_visualization_stack.html"), include_plotlyjs='cdn')
    else:


        # ==== INPUT ====
        # df must have: variant_label, %NSL-OH1, %NSL-OH2, %NSL-N3, %NSL

        COLOR = {
            '%NSL-N3': '#2ca02c',  # green
            '%NSL-OH1+OH2': '#ff7f0e',  # orange
            '%NSL': '#d62728',  # red
        }
        VARIANT_GAP = 1.0  # small gap between variant clusters
        CONDENSED = ['%NSL-N3', '%NSL-OH1+OH2', '%NSL']  # grouped bars per replicate

        # ==== PREP ====
        df_rep = df.copy()
        df_rep['variant_label'] = df_rep['variant_label'].astype(str)
        df_rep['replicate_id'] = df_rep.groupby('variant_label').cumcount() + 1
        df_rep['%NSL-OH1+OH2'] = df_rep['%NSL-OH1'] + df_rep['%NSL-OH2']

        # Chemoselectivity ratios (avoid div-by-zero -> NA)
        df_rep['ratio_A'] = df_rep['%NSL-N3'] / df_rep['%NSL'].replace(0, pd.NA)
        df_rep['ratio_B'] = df_rep['%NSL-N3'] / (df_rep['%NSL-OH1'] + df_rep['%NSL-OH2']).replace(0, pd.NA)

        # ==== HELPERS ====
        def order_variants_by(cols):
            """Sort variants by mean of selected column(s) (sum if multiple)."""
            if isinstance(cols, str):
                cols = [cols]
            means = df_rep.groupby('variant_label')[cols].mean()
            scores = means.sum(axis=1)
            return scores.sort_values(ascending=False).index.tolist()

        def dataframe_in_variant_order(vorder):
            """Sort rows by variant order, then replicate_id."""
            tmp = df_rep.copy()
            tmp['variant_label'] = pd.Categorical(tmp['variant_label'], categories=vorder, ordered=True)
            return tmp.sort_values(['variant_label', 'replicate_id']).reset_index(drop=True)

        def x_positions_for(df_ordered, gap=VARIANT_GAP, replicate_spacing=1):
            """Add a bit more space between replicates within each variant cluster."""
            x, pos = [], 0.0
            for _, g in df_ordered.groupby('variant_label', sort=False):
                n = len(g)
                # replicate_spacing controls distance between bars within the same variant
                x.extend([pos + i * (1 + replicate_spacing) for i in range(n)])
                pos += n * (1 + replicate_spacing) + gap
            return x

        def tick_text_for(df_ordered):
            return (df_ordered['variant_label'].astype(str)
                    + '_r' +
                    df_ordered['replicate_id'].astype(str)).tolist()

        def mean_line_shapes_for(df_ordered, x_vals, cols, color='black'):
            """One bold line per variant: mean of sum(cols), spanning the cluster."""
            if isinstance(cols, str):
                cols = [cols]
            shapes = []
            start = 0
            for _, g in df_ordered.groupby('variant_label', sort=False):
                n = len(g)
                if n == 0:
                    continue
                x0 = x_vals[start] - 0.45
                x1 = x_vals[start + n - 1] + 0.45
                avg_val = g[cols].sum(axis=1).mean()
                shapes.append(dict(
                    type='line', xref='x', yref='y',
                    x0=x0, x1=x1, y0=avg_val, y1=avg_val,
                    line=dict(width=4, color=color),
                    layer='above'
                ))
                start += n
            return shapes

        def build_bar_traces(df_ordered, x_vals):
            """Three side-by-side bars per replicate (N3, OH1+OH2, NSL), always in legend."""
            traces = []
            for m in CONDENSED:
                traces.append(go.Bar(
                    x=x_vals,
                    y=df_ordered[m],
                    name=m,
                    legendgroup=m,  # same group name across modes
                    showlegend=True,  # <-- always show so you can toggle in any mode
                    marker=dict(color=COLOR[m], opacity=0.7),
                    hovertemplate='%{x}<br>' + m + ': %{y:.2f}%<extra></extra>'
                ))
            return traces

        def build_chem_trace(df_ordered, x_vals, which):
            """Chemoselectivity scatter on right axis (per mode), NOT in legend."""
            if which == 'A':
                y = df_ordered['ratio_A']
                name = 'Chemoselectivity: %NSL-N3 / %NSL'
                marker = dict(color='black', size=10, symbol='circle-open', line=dict(width=2))
            else:
                y = df_ordered['ratio_B']
                name = 'Chemoselectivity: %NSL-N3 / (%NSL-OH1+%NSL-OH2)'
                marker = dict(color='purple', size=10, symbol='diamond-open', line=dict(width=2))
            return go.Scatter(
                x=x_vals, y=y,
                mode='markers',
                name=name,
                legendgroup=f'chem_{which}',
                showlegend=False,  # <-- keep dots out of legend
                yaxis='y2',
                marker=marker,
                hovertemplate='%{x}<br>' + name + ': %{y:.2f}<extra></extra>'
            )

        # ==== MODES & DATA ====
        modes = {
            'Original order': dict(sort_cols=None, mean_cols=None),
            'Sort by %NSL-N3': dict(sort_cols=['%NSL-N3'], mean_cols=['%NSL-N3']),
            'Sort by %NSL-OH1 + %NSL-OH2': dict(sort_cols=['%NSL-OH1+OH2'], mean_cols=['%NSL-OH1+OH2']),
            'Sort by %NSL': dict(sort_cols=['%NSL'], mean_cols=['%NSL']),
        }

        fig = go.Figure()
        trace_index = {}  # mode -> {'bars':[i,i,i], 'chemA':i, 'chemB':i}
        layout_state = {}  # mode -> {'tickvals':..., 'ticktext':..., 'shapes':...}

        # Build everything for each mode (bars + chem A + chem B); only Original is visible initially
        for mode, cfg in modes.items():
            vorder = df_rep['variant_label'].drop_duplicates().tolist() if cfg[
                                                                               'sort_cols'] is None else order_variants_by(
                cfg['sort_cols'])
            df_ord = dataframe_in_variant_order(vorder)
            x_vals = x_positions_for(df_ord)
            ticks = tick_text_for(df_ord)
            shapes = [] if cfg['mean_cols'] is None else mean_line_shapes_for(df_ord, x_vals, cfg['mean_cols'])

            # bars
            bars = build_bar_traces(df_ord, x_vals)
            bar_idxs = []
            for tr in bars:
                tr.visible = (mode == 'Original order')
                fig.add_trace(tr)
                bar_idxs.append(len(fig.data) - 1)

            # chem dots (hidden by default)
            chemA = build_chem_trace(df_ord, x_vals, which='A')
            chemB = build_chem_trace(df_ord, x_vals, which='B')
            chemA.visible = False
            chemB.visible = False
            fig.add_trace(chemA)
            fig.add_trace(chemB)
            chemA_idx = len(fig.data) - 2
            chemB_idx = len(fig.data) - 1

            trace_index[mode] = {'bars': bar_idxs, 'chemA': chemA_idx, 'chemB': chemB_idx}
            layout_state[mode] = {'tickvals': x_vals, 'ticktext': ticks, 'shapes': shapes}

        # Right y-axis range
        max_ratio = float(pd.concat([df_rep['ratio_A'], df_rep['ratio_B']]).max())

        # ==== LAYOUT ====
        fig.update_layout(
            barmode='group',
            bargap=0.25,
            plot_bgcolor='white',
            font=dict(family='Times New Roman', size=14),
            title='NSL Composition per Sample (Grouped Bars with Variant Spacing)',
            legend_title_text='Measurement (click to toggle)',
            yaxis=dict(title='Percent', range=[0, 100]),
            yaxis2=dict(
                title='Chemoselectivity Ratio',
                overlaying='y', side='right', showgrid=False,
                range=[0, max(1.0, max_ratio * 1.1)]
            ),
        )

        # Initial ticks (original)
        orig = layout_state['Original order']
        fig.update_xaxes(
            title='Variant (Replicate)',
            tickmode='array', tickvals=orig['tickvals'], ticktext=orig['ticktext'],
            tickangle=-45, showgrid=False
        )

        # ==== SINGLE COMBINED DROPDOWN (mode × chem state) ====
        buttons = []
        for mode in modes.keys():
            for chem_state, label_suffix in [
                ('none', '(No chem)'),
                ('A', '(Ratio A: N3/NSL)'),
                ('B', '(Ratio B: N3/(OH1+OH2))')
            ]:
                vis = [False] * len(fig.data)

                # bars for this mode ON
                for i in trace_index[mode]['bars']:
                    vis[i] = True

                # chem for this mode ON/OFF
                if chem_state == 'A':
                    vis[trace_index[mode]['chemA']] = True
                elif chem_state == 'B':
                    vis[trace_index[mode]['chemB']] = True

                st = layout_state[mode]
                title_suffix = '' if mode == 'Original order' else f' — {mode}'

                buttons.append(dict(
                    label=f"{mode} {label_suffix}",
                    method='update',
                    args=[
                        {'visible': vis},
                        {
                            'xaxis.tickvals': st['tickvals'],
                            'xaxis.ticktext': st['ticktext'],
                            'shapes': st['shapes'],
                            'title.text': f'NSL Composition per Sample{title_suffix}'
                        }
                    ]
                ))

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction='down',
                    showactive=True,
                    x=1.02, y=1.15,
                    xanchor='right', yanchor='top',
                    bgcolor='white', bordercolor='gray'
                )
            ]
        )

        # ==== SHOW or SAVE ====
        # fig.show()
        fig.write_html(join(out_dir,"ccm_visualization_bar.html"), include_plotlyjs='cdn')
        # fig.write_image("ccm_visualization.png", scale=3, width=1600, height=900)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    # Mandatory positional argument (not optional with --flag)
    parser.add_argument("config_file",
                        help="Path to the YAML file",
                        type=str)

    args = parser.parse_args()

    # Load and validate YAML config
    config = load_yaml_config(args.config_file)


    if config.get('now', True):
        now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        out_dir = join(config['out_dir'], f'{config["run_name"]}_{now}')
    else:
        now = ''
        out_dir = join(config['out_dir'], f'{config["run_name"]}')


    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, f'{config["run_name"]}.json'), "w") as f:
        json.dump(config, f, indent=4)

    mylogger = log_func(out_dir)
    mylogger.info(f'running {config["run_name"]} analysis')
    mylogger.info(f'out_dir {out_dir}')


    visualize_ccm(config,out_dir)




