import json

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import copy


def hex_to_rgba(hex, alpha=0.4):
    hex = hex.lstrip('#')
    lv = len(hex)
    return 'rgba(' + ', '.join(str(int(hex[i:i + lv // 3], 16)) for i in range(0, lv, lv // 3)) + f', {alpha})'


def generate_radar_plots(model_scores):
    num_models = len(model_scores)
    columns_per_row = 3
    specs = [[
        {
            'type': 'polar',
            'colspan': columns_per_row,
        },
        None,
        None,
    ]]
    for _ in range((num_models // columns_per_row)):
        specs.append([{'type': 'polar'} for _ in range(columns_per_row)])

    fig = make_subplots(
        rows=len(specs),
        cols=columns_per_row,
        specs=specs,
        subplot_titles=["All Models"] + [model for model in model_scores.keys()],
    )

    colors = px.colors.qualitative.Plotly
    row = 1
    for i, (model, model_score) in enumerate(model_scores.items()):
        category_scores = model_score['category_scores']
        categories = list(category_scores.keys())
        scores = [category_scores[category]['score'] for category in categories]
        color = colors[i % len(colors)]
        rgba_color = hex_to_rgba(color)

        trace = go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            mode='none',
            name=model,
            fillcolor=rgba_color
        )
        fig.add_trace(trace, row=1, col=1)

        model_plot_trace = copy.deepcopy(trace)
        model_plot_trace.showlegend = False

        col = 1 + (i % columns_per_row)
        if col == 1:
            row += 1

        fig.add_trace(model_plot_trace, row=row, col=col)

    fig.update_polars(
        radialaxis=dict(
            range=[0, 100],
            angle=-90,
            tickangle=-90,
        ),
    )

    fig.update_layout(
        legend=dict(
            orientation='h',
            xanchor='center',
            x=0.5, y=1.1
        ),
        width=500 * columns_per_row,
        height=500 * (num_models // columns_per_row),
        autosize=True,
        template='plotly_dark'
    )

    fig.show()
    fig.write_image("reports/aggregate_summary_radar.svg")


def generate_bar_charts(model_scores):
    models = []
    passed = []
    failed = []

    for model, model_score in model_scores.items():
        total_score = model_score['total_score']['score']
        models.append(model)
        passed.append(total_score)
        failed.append(100 - total_score)

    # Sort in descending order
    df = pd.DataFrame({'Model': models, 'Passed': passed, 'Failed': failed})
    df = df.sort_values(by='Passed', ascending=False)

    colors = px.colors.qualitative.Plotly[1:3]
    pass_color = hex_to_rgba(colors[1], 0.7)
    fail_color = hex_to_rgba(colors[0], 0.7)
    fig = go.Figure([
        go.Bar(
            y=df['Model'],
            x=df['Passed'],
            name='Passed',
            text=[f'{x:.1f}' for x in df['Passed']],
            textposition='inside',
            textfont=dict(color='white'),
            orientation='h',
            marker={'color': pass_color}
        ),
        go.Bar(
            y=df['Model'],
            x=df['Failed'],
            name='Failed',
            text=[f'{x:.1f}' for x in df['Failed']],
            textposition='inside',
            textfont=dict(color='white'),
            orientation='h',
            marker={'color': fail_color}
        )
    ])

    fig.update_layout(
        barmode='stack',
        xaxis=dict(showticklabels=False),
        yaxis=dict(
            autorange='reversed',
            ticksuffix='   '
        ),
        legend=dict(
            orientation='h',
            xanchor='center',
            x=0.5, y=1.1
        ),
        height=500,
        width=1500,
        bargap=0.1,
        autosize=True,
        template='plotly_dark'
    )

    # Show the figure
    fig.show()
    fig.write_image("reports/aggregate_summary_bar.svg")


def plot_results(csv_path: str):
    df = pd.read_csv(csv_path, quotechar='"', delimiter=",")

    # Collect model scores
    model_scores = {}
    model_columns = df.columns[4:]
    for index, row in df.iterrows():
        for model in model_columns:
            pass_rate = row.get(model, "0/0")
            passed, runs = map(int, pass_rate.split('/')) if '/' in pass_rate else (0, 0)

            if model not in model_scores:
                model_scores[model] = {
                    'total_score': {
                        'passed': 0,
                        'runs': 0
                    },
                    'category_scores': {}
                }

            model_scores[model]['total_score']['passed'] += passed
            model_scores[model]['total_score']['runs'] += runs

            categories = row['categories'].split(' ')
            for category in categories:
                if category not in model_scores[model]['category_scores']:
                    model_scores[model]['category_scores'][category] = {
                        'passed': 0,
                        'runs': 0
                    }

                model_scores[model]['category_scores'][category]['passed'] += passed
                model_scores[model]['category_scores'][category]['runs'] += runs

    # Normalize scores
    for model, model_score in model_scores.items():
        total_passed = model_score['total_score']['passed']
        total_runs = model_score['total_score']['runs']
        model_score['total_score']['score'] = 100 * total_passed / float(total_runs) if total_runs > 0 else 0

        for category, category_score in model_score['category_scores'].items():
            category_passed = category_score['passed']
            category_runs = category_score['runs']
            category_score['score'] = 100 * category_passed / float(category_runs) if category_runs > 0 else 0
            category_score['total_score_contribution'] = 100 * category_passed / float(total_runs) if total_runs > 0 else 0
            model_score['category_scores'][category] = category_score

        model_scores[model] = model_score

    print(f"{json.dumps(model_scores, indent=4)}")
    generate_radar_plots(model_scores)
    generate_bar_charts(model_scores)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "reports/aggregate_summary.csv"
    plot_results(csv_path)


if __name__ == "__main__":
    main()
