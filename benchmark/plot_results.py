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

    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.0, 100.0],
                gridcolor='white',
                linecolor='white'
            ),
            angularaxis=dict(
                linecolor='white',
                gridcolor='white'
            )
        ),
        autosize=True,
        template='plotly_dark',
        width=600 * columns_per_row,
        height=600 * (num_models // columns_per_row),
        title='Function Calling Benchmark Results by Category'
    )
    fig.update_layout(layout)

    # Create a trace for each model
    colors = px.colors.qualitative.Plotly
    row = 1
    for i, (model, scores) in enumerate(model_scores.items()):
        categories = list(scores.keys())
        results = [scores[category]['normalized'] for category in categories]
        color = colors[i % len(colors)]
        rgba_color = hex_to_rgba(color)

        trace = go.Scatterpolar(
            r=results,
            theta=categories,
            fill='toself',
            name=model,
            mode='lines+markers',
            line=dict(color=color),
            fillcolor=rgba_color
        )
        fig.add_trace(trace, row=1, col=1)

        model_plot_trace = copy.deepcopy(trace)
        model_plot_trace.showlegend = False

        col = 1 + (i % columns_per_row)
        if col == 1:
            row += 1

        fig.add_trace(model_plot_trace, row=row, col=col)

    fig.update_polars(radialaxis=dict(range=[0.0, 100.0]), angularaxis=dict(type='category'))
    fig.show()
    fig.write_image("reports/aggregate_summary_radar.svg", scale=2.0)


def plot_results(csv_path: str):
    df = pd.read_csv(csv_path, quotechar='"', delimiter=",")

    # Collect model scores
    model_scores = {}
    model_columns = df.columns[4:]
    for index, row in df.iterrows():
        test_categories = row['categories'].split(' ')
        for category in test_categories:
            for model in model_columns:
                pass_rate = row.get(model, "0/0")
                passed, runs = map(int, pass_rate.split('/')) if '/' in pass_rate else (0, 0)

                if model not in model_scores:
                    model_scores[model] = {}
                if category not in model_scores[model]:
                    model_scores[model][category] = {'passed': 0.0, 'runs': 0.0}

                model_scores[model][category]['passed'] += passed
                model_scores[model][category]['runs'] += runs

    # Normalize scores
    for model, scores in model_scores.items():
        for category, score in scores.items():
            total_runs = score['runs']
            normalized_score = 0.0
            if total_runs > 0:
                normalized_score = 100.0 * (score['passed'] / total_runs)  # Normalize to [0, 10]

            model_scores[model][category]['normalized'] = normalized_score

    generate_radar_plots(model_scores)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "reports/aggregate_summary.csv"
    plot_results(csv_path)


if __name__ == "__main__":
    main()
