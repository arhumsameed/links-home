import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    DATA_PATH,
    FEATURE_COLS,
    ROAD_LABELS,
    TRAFFIC_LABELS,
    evaluate,
    feature_importance,
    load_data,
    train_model,
)

# --- 1. PAGE CONFIGURATION & DARK THEME STYLING ---
st.set_page_config(
    page_title="Links-Home | Cold Chain Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    header[data-testid="stHeader"] {
        background-color: #0b0f19;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f293d;
    }
    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1f293d;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 6px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- 2. DATA & MODEL (logic lives in pipeline.py so CI can test it) ---
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Add it to the repository root.")
    st.stop()


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(features):
    return train_model(get_data(), list(features))


df = get_data()
model, X_test, y_test = get_model(tuple(FEATURE_COLS))

plotly_dark_layout = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#cbd5e1", family="sans-serif"),
    xaxis=dict(gridcolor="#1f293d", zerolinecolor="#1f293d"),
    yaxis=dict(gridcolor="#1f293d", zerolinecolor="#1f293d"),
)

colors_map = {"On_Time": "#10b981", "At_Risk": "#f59e0b", "Delayed": "#ef4444"}

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("LINKS-HOME")
st.sidebar.caption("Cold Chain Intelligence & Logistics Analytics")

app_mode = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "AI Dispatch Risk Guard",
        "Fleet & Route Analytics",
        "Model Performance",
    ],
)

# --- 4. DASHBOARD VIEWS ---

if app_mode == "Executive Overview":
    st.title("Executive Operations Overview")
    st.caption("Historical shipment data: delivery status and thermal compliance")

    total_shipments = len(df)
    on_time_count = len(df[df["Delivery_Status"] == "On_Time"])
    at_risk_count = len(df[df["Delivery_Status"] == "At_Risk"])
    delayed_count = len(df[df["Delivery_Status"] == "Delayed"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SHIPMENTS IN DATASET", f"{total_shipments:,}")
    col2.metric("ON-TIME DELIVERIES", f"{on_time_count:,}")
    col3.metric("AT-RISK SHIPMENTS", f"{at_risk_count:,}")
    col4.metric("DELAYED / EXCURSIONS", f"{delayed_count:,}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Delivery Status Breakdown")
        status_counts = df["Delivery_Status"].value_counts().reset_index()
        status_counts.columns = ["Delivery_Status", "Count"]

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=status_counts["Delivery_Status"],
                    values=status_counts["Count"],
                    hole=0.55,
                    marker=dict(
                        colors=[
                            colors_map.get(s, "#3b82f6")
                            for s in status_counts["Delivery_Status"]
                        ]
                    ),
                )
            ]
        )
        fig_donut.update_layout(**plotly_dark_layout, height=380)
        st.plotly_chart(fig_donut)

    with col_b:
        st.subheader("Cargo Temperature Spread by Status")
        fig_hist = go.Figure()
        for status, color in colors_map.items():
            subset = df[df["Delivery_Status"] == status]
            if not subset.empty:
                fig_hist.add_trace(
                    go.Histogram(
                        x=subset["Temperature_C"],
                        name=status,
                        marker_color=color,
                        opacity=0.75,
                    )
                )
        fig_hist.update_layout(
            **plotly_dark_layout,
            height=380,
            barmode="overlay",
            xaxis_title="Cargo Temperature (Celsius)",
            yaxis_title="Shipment Volume",
        )
        st.plotly_chart(fig_hist)

elif app_mode == "AI Dispatch Risk Guard":
    st.title("AI Dispatch Risk Guard")
    st.caption(
        "Risk estimate from a Random Forest trained on historical shipments. "
        "See 'Model Performance' for accuracy and limitations."
    )

    with st.form("dispatch_form"):
        st.subheader("Input Trip & Cargo Parameters")
        col1, col2 = st.columns(2)

        with col1:
            temp_c = st.slider(
                "Current Cargo Temperature (Celsius)", 2.0, 12.0, 4.5, 0.1
            )
            temp_var = st.slider(
                "Temperature Variation Rate (Celsius/hr)", 0.1, 4.0, 0.5, 0.1
            )
            vehicle_age = st.slider("Vehicle Age (Years)", 1, 15, 4)
            cooling_load = st.slider(
                "Cooling Capacity Utilization (%)", 40.0, 100.0, 75.0, 1.0
            )

        with col2:
            distance = st.number_input(
                "Route Distance (km)", min_value=5, max_value=500, value=120
            )
            traffic_level = st.selectbox(
                "Expected Traffic Intensity",
                list(TRAFFIC_LABELS.keys()),
                format_func=lambda x: TRAFFIC_LABELS[x],
            )
            road_condition = st.selectbox(
                "Road Quality Tier",
                list(ROAD_LABELS.keys()),
                format_func=lambda x: ROAD_LABELS[x],
            )
            expected_delay = st.slider(
                "Estimated Transit Delay (Hours)", 0.0, 5.0, 0.5, 0.25
            )

        submitted = st.form_submit_button("RUN RISK EVALUATION")

    if submitted:
        input_data = pd.DataFrame(
            [[
                temp_c,
                temp_var,
                vehicle_age,
                cooling_load,
                distance,
                traffic_level,
                road_condition,
                expected_delay,
            ]],
            columns=FEATURE_COLS,
        )

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        prob_dict = dict(zip(model.classes_, probabilities))

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("Decision Output")
            if prediction == "On_Time":
                st.success(
                    f"DISPATCH APPROVED\n\nModel probability:"
                    f" {prob_dict.get('On_Time', 0):.1%}\n\nCold chain parameters"
                    " and transit window look secure."
                )
            elif prediction == "At_Risk":
                st.warning(
                    f"DISPATCH AT RISK\n\nModel probability:"
                    f" {prob_dict.get('At_Risk', 0):.1%}\n\nSecondary thermal"
                    " protection or rerouting recommended before departure."
                )
            else:
                st.error(
                    f"DISPATCH BLOCKED\n\nModel probability:"
                    f" {prob_dict.get('Delayed', 0):.1%}\n\nHigh failure"
                    " probability. Review before releasing the vehicle."
                )

        with res_col2:
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame(
                list(prob_dict.items()), columns=["Status", "Probability"]
            )
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=prob_df["Status"],
                        y=prob_df["Probability"],
                        marker_color=[
                            colors_map.get(s, "#3b82f6") for s in prob_df["Status"]
                        ],
                        text=[f"{p:.1%}" for p in prob_df["Probability"]],
                        textposition="auto",
                    )
                ]
            )
            fig_bar.update_layout(
                **plotly_dark_layout, height=260, yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_bar)

elif app_mode == "Fleet & Route Analytics":
    st.title("Fleet Health & Route Vulnerability Analysis")
    st.caption("Failure rates and degradation patterns in historical shipments")

    selected_road = st.selectbox(
        "FILTER ANALYTICS BY ROAD CONDITION TIER",
        list(ROAD_LABELS.keys()),
        format_func=lambda x: f"Tier {x} - {ROAD_LABELS[x]}",
    )

    filtered_df = df[df["Road_Condition"] == selected_road]

    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        st.subheader("Vehicle Age vs. Thermal Failure Rate")
        age_analysis = (
            filtered_df.groupby("Vehicle_Age_Years")["Delivery_Status"]
            .apply(lambda s: (s != "On_Time").mean() * 100)
            .reset_index(name="Failure_Rate_Pct")
        )

        fig_age = go.Figure(
            data=[
                go.Scatter(
                    x=age_analysis["Vehicle_Age_Years"],
                    y=age_analysis["Failure_Rate_Pct"],
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=3),
                    marker=dict(size=8, color="#60a5fa"),
                )
            ]
        )
        fig_age.update_layout(
            **plotly_dark_layout,
            height=380,
            xaxis_title="Vehicle Age (Years)",
            yaxis_title="Failure / Risk Rate (%)",
        )
        st.plotly_chart(fig_age)

    with r1_col2:
        st.subheader("Temperature Variance by Traffic Level")
        fig_box = go.Figure()
        for status, color in colors_map.items():
            subset = filtered_df[filtered_df["Delivery_Status"] == status]
            if not subset.empty:
                fig_box.add_trace(
                    go.Box(
                        x=subset["Traffic_Level"],
                        y=subset["Temperature_Variation"],
                        name=status,
                        marker_color=color,
                    )
                )
        fig_box.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="#cbd5e1", family="sans-serif"),
            height=380,
            boxmode="group",
            yaxis_title="Temp Variation Rate (C/hr)",
            yaxis=dict(gridcolor="#1f293d", zerolinecolor="#1f293d"),
        )
        fig_box.update_xaxes(
            gridcolor="#1f293d",
            zerolinecolor="#1f293d",
            tickmode="array",
            tickvals=list(TRAFFIC_LABELS.keys()),
            ticktext=list(TRAFFIC_LABELS.values()),
            title="Traffic Intensity",
        )
        st.plotly_chart(fig_box)

    st.subheader("Cooling Load Utilization vs. Cargo Temperature Excursion")
    sample_scatter = filtered_df.sample(
        n=min(1000, len(filtered_df)), random_state=42
    )
    fig_scatter = go.Figure()
    for status, color in colors_map.items():
        subset = sample_scatter[sample_scatter["Delivery_Status"] == status]
        if not subset.empty:
            fig_scatter.add_trace(
                go.Scatter(
                    x=subset["Cooling_Load_Percent"],
                    y=subset["Temperature_C"],
                    mode="markers",
                    name=status,
                    marker=dict(
                        size=subset["Transit_Delay_hr"] * 5 + 6,
                        color=color,
                        opacity=0.7,
                    ),
                )
            )
    fig_scatter.update_layout(
        **plotly_dark_layout,
        height=450,
        xaxis_title="Cooling System Capacity Utilization (%)",
        yaxis_title="Recorded Cargo Temperature (Celsius)",
    )
    st.plotly_chart(fig_scatter)

elif app_mode == "Model Performance":
    st.title("Model Performance")
    st.caption("Evaluated on a 20% held-out test set (stratified, fixed seed)")

    scores = evaluate(model, X_test, y_test)

    m1, m2, m3 = st.columns(3)
    m1.metric("ACCURACY", f"{scores['accuracy']:.1%}")
    m2.metric("MACRO F1", f"{scores['macro_f1']:.3f}")
    m3.metric("TEST SHIPMENTS", f"{len(y_test):,}")

    st.markdown("---")
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Confusion Matrix")
        labels = scores["labels"]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=scores["confusion"],
                x=labels,
                y=labels,
                text=scores["confusion"],
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            **plotly_dark_layout,
            height=380,
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        st.plotly_chart(fig_cm)

    with c_right:
        st.subheader("Feature Importance")
        imp = feature_importance(model, FEATURE_COLS)
        fig_imp = go.Figure(
            data=[
                go.Bar(
                    x=imp["Importance"],
                    y=imp["Feature"],
                    orientation="h",
                    marker_color="#3b82f6",
                )
            ]
        )
        fig_imp.update_layout(**plotly_dark_layout, height=380)
        st.plotly_chart(fig_imp)

    st.markdown("---")
    st.subheader("Leakage check: is the model relying on transit delay?")
    st.write(
        "Transit delay is only fully known during or after a trip, so a pre-departure"
        " model should not depend heavily on it. This compares accuracy with and"
        " without that feature."
    )

    features_no_delay = tuple(c for c in FEATURE_COLS if c != "Transit_Delay_hr")
    model_nd, X_test_nd, y_test_nd = get_model(features_no_delay)
    scores_nd = evaluate(model_nd, X_test_nd, y_test_nd)
    drop = scores["accuracy"] - scores_nd["accuracy"]

    k1, k2, k3 = st.columns(3)
    k1.metric("ACCURACY, ALL FEATURES", f"{scores['accuracy']:.1%}")
    k2.metric("ACCURACY, NO TRANSIT DELAY", f"{scores_nd['accuracy']:.1%}")
    k3.metric("DROP", f"{drop:.1%}")

    if drop > 0.10:
        st.warning(
            "Accuracy falls sharply without transit delay, so the model leans on"
            " information that may not be available before departure. Treat the"
            " risk guard as indicative, not predictive."
        )
    else:
        st.info(
            "Accuracy holds up without transit delay, so the model is not relying"
            " on it heavily."
        )
