import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

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

# Feature list based on your dataset columns
feature_cols = [
    "Temperature_C",
    "Temperature_Variation",
    "Vehicle_Age_Years",
    "Cooling_Load_Percent",
    "Route_Distance_km",
    "Traffic_Level",
    "Road_Condition",
    "Transit_Delay_hr",
]


# --- 2. DIRECT CSV DATA LOADING & ROBUST ENCODING ---
@st.cache_data
def load_and_process_data():
  df = pd.read_csv(r"C:\Users\arhum\OneDrive\Documents\Datasets\cold chain logistics\iot_cold_chain_logistics.csv")

  for col in feature_cols:
    if col in df.columns:
      if df[col].dtype == object or not pd.api.types.is_numeric_dtype(df[col]):
        cleaned = df[col].astype(str).str.strip().str.lower()
        if col == "Traffic_Level":
          traffic_map = {
              "low": 1,
              "medium": 2,
              "high": 3,
              "severe": 4,
              "none": 1,
          }
          df[col] = cleaned.map(traffic_map).fillna(2).astype(int)
        elif col == "Road_Condition":
          road_map = {"good": 1, "moderate": 2, "poor": 3}
          df[col] = cleaned.map(road_map).fillna(2).astype(int)
        else:
          df[col], _ = pd.factorize(df[col])
      else:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

  return df


df = load_and_process_data()


@st.cache_resource
def train_model(data):
  X = data[feature_cols]
  y = data["Delivery_Status"]
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  clf = RandomForestClassifier(n_estimators=100, random_state=42)
  clf.fit(X_train, y_train)
  return clf


model = train_model(df)

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
st.sidebar.caption("Cold Chain Intelligence & Logistics Operations System")

app_mode = st.sidebar.radio(
    "NAVIGATION",
    ["Executive Overview", "AI Dispatch Risk Guard", "Fleet & Route Analytics"],
)

# --- 4. DASHBOARD VIEWS ---

if app_mode == "Executive Overview":
  st.title("Executive Operations Command Center")
  st.caption("Real-time monitoring of fleet status and thermal compliance")

  total_shipments = len(df)
  on_time_count = len(df[df["Delivery_Status"] == "On_Time"])
  at_risk_count = len(df[df["Delivery_Status"] == "At_Risk"])
  delayed_count = len(df[df["Delivery_Status"] == "Delayed"])

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("TOTAL ACTIVE SHIPMENTS", f"{total_shipments:,}")
  col2.metric("ON-TIME DELIVERIES", f"{on_time_count:,}")
  col3.metric("AT-RISK DISPATCHES", f"{at_risk_count:,}")
  col4.metric("EXCURSIONS / DELAYED", f"{delayed_count:,}")

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
    st.plotly_chart(fig_donut, use_container_width=True)

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
    st.plotly_chart(fig_hist, use_container_width=True)

elif app_mode == "AI Dispatch Risk Guard":
  st.title("AI Dispatch Risk Guard")
  st.caption("Predictive risk evaluation before departure")

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
          [1, 2, 3],
          format_func=lambda x: {1: "Low", 2: "Medium", 3: "Severe"}[x],
      )
      road_condition = st.selectbox(
          "Road Quality Tier",
          [1, 2, 3],
          format_func=lambda x: {
              1: "Good (Highway)",
              2: "Moderate",
              3: "Poor (Rural/Damaged)",
          }[x],
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
        columns=feature_cols,
    )

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, probabilities))

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
      st.subheader("Decision Output")
      if prediction == "On_Time":
        st.success(
            f"DISPATCH APPROVED\n\nConfidence Level:"
            f" {prob_dict.get('On_Time', 0):.1%}\n\nCold chain parameters and"
            " transit window secured."
        )
      elif prediction == "At_Risk":
        st.warning(
            f"DISPATCH AT RISK\n\nConfidence Level:"
            f" {prob_dict.get('At_Risk', 0):.1%}\n\nSecondary thermal protection"
            " or rerouting required before departure."
        )
      else:
        st.error(
            f"DISPATCH BLOCKED\n\nConfidence Level:"
            f" {prob_dict.get('Delayed', 0):.1%}\n\nCritical failure"
            " probability. Do not release vehicle."
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
      st.plotly_chart(fig_bar, use_container_width=True)

elif app_mode == "Fleet & Route Analytics":
  st.title("Fleet Health & Route Vulnerability Analysis")
  st.caption("Interactive analysis of degradation vectors and failure rates")

  selected_road = st.selectbox(
      "FILTER ANALYTICS BY ROAD CONDITION TIER",
      [1, 2, 3],
      format_func=lambda x: {
          1: "Tier 1 - Good (Highways)",
          2: "Tier 2 - Moderate",
          3: "Tier 3 - Poor (Damaged / Rural)",
      }[x],
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
    st.plotly_chart(fig_age, use_container_width=True)

  with r1_col2:
    st.subheader("Temperature Variance by Traffic Level")
    fig_box = go.Figure()
    for status, color in colors_map.items():
      subset = filtered_df[filtered_df["Delivery_Status"] == status]
      if not subset.empty and "Temperature_Variation" in subset.columns:
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
        tickvals=[1, 2, 3],
        ticktext=["Low", "Medium", "Severe"],
        title="Traffic Intensity",
    )
    st.plotly_chart(fig_box, use_container_width=True)

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
                  size=(
                      subset["Transit_Delay_hr"] * 5 + 6
                      if "Transit_Delay_hr" in subset.columns
                      else 8
                  ),
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
  st.plotly_chart(fig_scatter, use_container_width=True)