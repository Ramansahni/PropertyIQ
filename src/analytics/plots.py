def apply_plotly_dark_theme(fig):
    """
    Applies unified dark mode styling to Plotly figures.
    """
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#f8fafc"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

CHART_THEME_COLORS = ['#8b5cf6', '#3b82f6', '#06b6d4', '#10b981', '#f43f5e']
