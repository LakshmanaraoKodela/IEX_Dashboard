import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="MCP Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- Load & Preprocess Data Function ----------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_preprocess_data():
    # ---------- Load & Preprocess DAM ----------
    dam = pd.read_excel("DAM_output.xlsx", engine="openpyxl")
    dam['Date'] = pd.to_datetime(dam['Date'])
    dam['Market'] = 'DAM'
    dam.rename(columns={'MCP (Rs/MWh) *': 'MCP'}, inplace=True)
    dam['MCP'] = pd.to_numeric(dam['MCP'], errors='coerce') / 1000
    dam.rename(columns={'MCP': 'MCP (Rs/kWh)'}, inplace=True)

    # ---------- Load & Preprocess GDAM ----------
    gdam = pd.read_excel("GDAM_output.xlsx", engine="openpyxl")
    gdam['Date'] = pd.to_datetime(gdam['Date'])
    gdam['Market'] = 'GDAM'
    gdam.rename(columns={'MCP (Rs/MWh)': 'MCP'}, inplace=True)
    gdam['MCP'] = pd.to_numeric(gdam['MCP'], errors='coerce') / 1000
    gdam.rename(columns={'MCP': 'MCP (Rs/kWh)'}, inplace=True)

    # ---------- Load & Preprocess RTM ----------
    rtm = pd.read_excel("Filtered_MCV_MCP_Data.xlsx", engine="openpyxl")
    rtm['Date'] = pd.to_datetime(rtm['Date'], dayfirst=True)
    rtm['Market'] = 'RTM'
    rtm.rename(columns={'MCP (Rs/MWh)': 'MCP'}, inplace=True)
    rtm['MCP'] = pd.to_numeric(rtm['MCP'], errors='coerce') / 1000
    rtm.rename(columns={'MCP': 'MCP (Rs/kWh)'}, inplace=True)

    # ---------- Combine All ----------
    combined = pd.concat([
        dam[['Date', 'Hour', 'Time Block', 'MCP (Rs/kWh)', 'Market']],
        gdam[['Date', 'Hour', 'Time Block', 'MCP (Rs/kWh)', 'Market']],
        rtm[['Date', 'Hour', 'Time Block', 'MCP (Rs/kWh)', 'Market']]
    ], ignore_index=True)

    # Remove NaN values and sort by date
    combined = combined.dropna(subset=['MCP (Rs/kWh)', 'Date'])
    combined = combined.sort_values(['Date', 'Hour', 'Time Block']).reset_index(drop=True)

    # Create DateTime column for proper time handling
    combined['DateTime'] = pd.to_datetime(combined['Date'].astype(str) + ' ' + combined['Time Block'].str[:5])

    # Pre-calculate commonly used aggregations to speed up filtering
    combined['Year'] = combined['Date'].dt.year
    combined['Month'] = combined['Date'].dt.month
    combined['Day'] = combined['Date'].dt.day

    # Optimize data types for better performance
    combined = optimize_data_types(combined)

    return combined


# ---------- Market Colors ----------
MARKET_COLORS = {
    'DAM': '#1f77b4',  # Blue
    'GDAM': '#2ca02c',  # Green
    'RTM': '#d62728'  # Red
}


# ---------- Helper Functions ----------

@st.cache_data
def format_number(value):
    """Format numbers to 2 decimal places"""
    if pd.isna(value):
        return 0.00
    return round(float(value), 2)


def vectorized_format_numbers(series):
    """Vectorized number formatting for better performance"""
    return np.round(series.fillna(0), 2)
    """Calculate date ranges based on the maximum date in the dataset"""
    max_date = pd.to_datetime(max_date)

    ranges = {
        '7D': max_date - timedelta(days=7),
        '1M': max_date - timedelta(days=30),
        '3M': max_date - timedelta(days=90),
        '6M': max_date - timedelta(days=180)
    }

    return ranges

@st.cache_data
def aggregate_data(df_hash, agg_level):
    """Aggregate data based on the specified level - cached for performance"""
    # Reconstruct df from hash (this is handled by streamlit caching)
    df = df_hash

    if agg_level == '15min':
        # Return raw 15-minute data
        result = df.copy()
        result = result.sort_values(['Market', 'DateTime'])
        return result

    elif agg_level == 'hourly':
        result = df.groupby(['Date', 'Hour', 'Market']).agg({
            'MCP (Rs/kWh)': 'mean'
        }).reset_index()

        # Fix hour == 24
        result['Hour'] = result['Hour'].astype(int)
        mask_24 = result['Hour'] == 24
        result.loc[mask_24, 'Hour'] = 0
        result.loc[mask_24, 'Date'] = result.loc[mask_24, 'Date'] + pd.Timedelta(days=1)

        # Create DateTime for hourly data
        result['DateTime'] = pd.to_datetime(result['Date'].astype(str) + ' ' +
                                            result['Hour'].astype(str).str.zfill(2) + ':00:00')

        # Remove any rows with NaN values
        result = result.dropna(subset=['MCP (Rs/kWh)'])

    elif agg_level == 'daily':
        # Group by Date and Market
        result = df.groupby(['Date', 'Market']).agg({
            'MCP (Rs/kWh)': 'mean'
        }).reset_index()
        result['DateTime'] = result['Date']

        # Remove any rows with NaN values
        result = result.dropna(subset=['MCP (Rs/kWh)'])

    elif agg_level == 'weekly':
        # Group by week and Market
        df['Week'] = df['Date'].dt.to_period('W').dt.start_time
        result = df.groupby(['Week', 'Market']).agg({
            'MCP (Rs/kWh)': 'mean'
        }).reset_index()
        result['DateTime'] = result['Week']
        result['Date'] = result['Week']

        # Remove any rows with NaN values
        result = result.dropna(subset=['MCP (Rs/kWh)'])

    elif agg_level == 'monthly':
        # Group by month and Market
        df['Month'] = df['Date'].dt.to_period('M').dt.start_time
        result = df.groupby(['Month', 'Market']).agg({
            'MCP (Rs/kWh)': 'mean'
        }).reset_index()
        result['DateTime'] = result['Month']
        result['Date'] = result['Month']

        # Remove any rows with NaN values
        result = result.dropna(subset=['MCP (Rs/kWh)'])

    elif agg_level == 'quarterly':
        # Group by quarter and Market
        df['Quarter'] = df['Date'].dt.to_period('Q').dt.start_time
        result = df.groupby(['Quarter', 'Market']).agg({
            'MCP (Rs/kWh)': 'mean'
        }).reset_index()
        result['DateTime'] = result['Quarter']
        result['Date'] = result['Quarter']

        # Remove any rows with NaN values
        result = result.dropna(subset=['MCP (Rs/kWh)'])

    elif agg_level == 'yearly':
        # Group by year and Market
        df['Year'] = df['Date'].dt.to_period('Y').dt.start_time
        result = df.groupby(['Year', 'Market']).agg({
            'MCP (Rs/kWh)': 'mean'
        }).reset_index()
        result['DateTime'] = result['Year']
        result['Date'] = result['Year']

        # Remove any rows with NaN values
        result = result.dropna(subset=['MCP (Rs/kWh)'])

    return result.sort_values(['Market', 'DateTime'])


def get_hover_template(market, agg_level):
    """Get appropriate hover template based on aggregation level"""
    templates = {
        '15min': f'<b>{market}</b><br>Date: %{{x|%d-%m-%Y}}<br>Time: %{{x|%H:%M}}<br>MCP: ₹%{{y:.2f}}/kWh<extra></extra>',
        'hourly': f'<b>{market}</b><br>Date: %{{x|%d-%m-%Y}}<br>Hour: %{{x|%H:00}}<br>Hourly Avg: ₹%{{y:.2f}}/kWh<extra></extra>',
        'daily': f'<b>{market}</b><br>Date: %{{x|%d-%m-%Y}}<br>Daily Avg: ₹%{{y:.2f}}/kWh<extra></extra>',
        'weekly': f'<b>{market}</b><br>Week: %{{x|%d-%m-%Y}}<br>Weekly Avg: ₹%{{y:.2f}}/kWh<extra></extra>',
        'monthly': f'<b>{market}</b><br>Month: %{{x|%b %Y}}<br>Monthly Avg: ₹%{{y:.2f}}/kWh<extra></extra>',
        'quarterly': f'<b>{market}</b><br>Quarter: %{{x|Q%q %Y}}<br>Quarterly Avg: ₹%{{y:.2f}}/kWh<extra></extra>',
        'yearly': f'<b>{market}</b><br>Year: %{{x|%Y}}<br>Yearly Avg: ₹%{{y:.2f}}/kWh<extra></extra>'
    }
    return templates.get(agg_level, templates['daily'])


def get_chart_title(agg_level):
    """Get chart title based on aggregation level"""
    titles = {
        '15min': 'Market Clearing Price Trend - 15-Minute View',
        'hourly': 'Market Clearing Price Trend - Hourly View',
        'daily': 'Market Clearing Price Trend - Daily View',
        'weekly': 'Market Clearing Price Trend - Weekly View',
        'monthly': 'Market Clearing Price Trend - Monthly View',
        'quarterly': 'Market Clearing Price Trend - Quarterly View',
        'yearly': 'Market Clearing Price Trend - Yearly View'
    }
    return titles.get(agg_level, 'Market Clearing Price Trend')


def get_x_axis_title(agg_level):
    """Get x-axis title based on aggregation level"""
    titles = {
        '15min': 'Date & Time (15-min intervals)',
        'hourly': 'Date & Time (Hourly)',
        'daily': 'Date (Daily)',
        'weekly': 'Date (Weekly)',
        'monthly': 'Date (Monthly)',
        'quarterly': 'Date (Quarterly)',
        'yearly': 'Date (Yearly)'
    }
    return titles.get(agg_level, 'Date')


def optimize_data_types(df):
    """Optimize data types for better performance without losing data points"""
    # Convert to more efficient data types
    if 'MCP (Rs/kWh)' in df.columns:
        df['MCP (Rs/kWh)'] = df['MCP (Rs/kWh)'].astype('float32')  # Reduce from float64

    # Optimize categorical data
    if 'Market' in df.columns:
        df['Market'] = df['Market'].astype('category')

    if 'Time Block' in df.columns:
        df['Time Block'] = df['Time Block'].astype('category')

    return df


@st.cache_data
def create_chart(df_hash, agg_level, selected_timeblocks_str, start_date_str, end_date_str):
    """Create the plotly chart - cached for performance, showing ALL data points"""
    df = df_hash

    # Use efficient filtering
    df_filtered = filter_data_efficiently(df_hash, start_date_str, end_date_str, selected_timeblocks_str, agg_level)

    # Early return if no data
    if df_filtered.empty:
        return go.Figure().add_annotation(
            text="No data available for selected date range",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Aggregate data based on selected level
    df_agg = aggregate_data(df_filtered, agg_level)

    # Use WebGL for ALL datasets with >1000 points for better performance
    use_webgl = len(df_agg) > 1000

    fig = go.Figure()

    # Create traces for each market - showing ALL points
    for market in ['DAM', 'GDAM', 'RTM']:  # Fixed order
        if market in df_agg['Market'].unique():
            sub = df_agg[df_agg['Market'] == market].copy()
            sub = sub.sort_values('DateTime')

            # Use vectorized formatting for better performance
            sub['MCP_formatted'] = vectorized_format_numbers(sub['MCP (Rs/kWh)'])
            avg_mcp = format_number(sub['MCP (Rs/kWh)'].mean())

            data_points = len(sub)

            # Optimize visual settings based on data density but show ALL points
            if data_points > 5000:
                marker_size = 1
                line_width = 1
                mode = 'lines'  # Lines only for very dense data
            elif data_points > 2000:
                marker_size = 2
                line_width = 1
                mode = 'lines+markers'
            elif data_points > 1000:
                marker_size = 2
                line_width = 1.5
                mode = 'lines+markers'
            else:
                # Set marker size and line width based on aggregation level
                if agg_level == '15min':
                    marker_size = 3
                    line_width = 2
                elif agg_level == 'hourly':
                    marker_size = 4
                    line_width = 2
                else:
                    marker_size = 6
                    line_width = 3
                mode = 'lines+markers'

            # Choose scatter type based on performance needs
            scatter_type = go.Scattergl if use_webgl else go.Scatter

            # Configure marker settings
            marker_config = dict(size=marker_size) if 'markers' in mode else None

            fig.add_trace(scatter_type(
                x=sub['DateTime'],
                y=sub['MCP_formatted'],
                mode=mode,
                name=f"{market} (Avg: ₹{avg_mcp})",
                line=dict(color=MARKET_COLORS[market], width=line_width),
                marker=marker_config,
                hovertemplate=get_hover_template(market, agg_level),
                connectgaps=False,  # Don't connect gaps in data
                # Optimize hover for large datasets
                hoverinfo='x+y+name' if data_points > 10000 else None
            ))

    # Get the maximum date from the data for range selector
    max_date_in_data = df['Date'].max()
    min_date_in_data = df['Date'].min()
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Update layout with performance optimizations
    fig.update_layout(
        title={
            'text': get_chart_title(agg_level),
            'x': 0.5,
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title=get_x_axis_title(agg_level),
        yaxis_title="MCP (₹/kWh)",
        template="plotly_white",
        hovermode="x unified" if len(df_agg) < 5000 else "closest",
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        height=650,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.5)',
            # Optimize tick spacing for large datasets
            nticks=20 if len(df_agg) > 10000 else None
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.5)',
            tickformat='.2f'
        ),
        margin=dict(l=80, r=40, t=80, b=60),
        # Optimize rendering performance
        uirevision='constant'  # Preserve UI state across updates
    )

    # Add range selector buttons for time-based aggregations
    if agg_level in ['15min', 'hourly', 'daily']:
        # Calculate how many days of data we have
        total_days = (max_date_in_data - min_date_in_data).days

        # Create range selector buttons based on available data
        range_buttons = []

        # Add buttons only if we have enough data for that range
        if total_days >= 7:
            range_buttons.append(dict(count=7, label="7D", step="day", stepmode="backward"))

        if total_days >= 30:
            range_buttons.append(dict(count=30, label="1M", step="day", stepmode="backward"))

        if total_days >= 90:
            range_buttons.append(dict(count=90, label="3M", step="day", stepmode="backward"))

        if total_days >= 180:
            range_buttons.append(dict(count=180, label="6M", step="day", stepmode="backward"))

        # Always add "All" button
        range_buttons.append(dict(step="all", label="All"))

        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=range_buttons,
                    bgcolor="rgba(255,255,255,0.8)",
                    activecolor="rgba(44,62,80,0.8)"
                ),
                rangeslider=dict(visible=False),
                type="date",
                # Set the initial range to show filtered data
                range=[start_date.strftime('%Y-%m-%d'),
                       end_date.strftime('%Y-%m-%d')]
            )
        )

    return fig


@st.cache_data
def filter_data_efficiently(df_hash, start_date_str, end_date_str, selected_timeblocks_str, agg_level):
    """Efficiently filter data with optimized operations"""
    df = df_hash
    selected_timeblocks = selected_timeblocks_str.split(',') if selected_timeblocks_str else []
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Use vectorized operations for filtering
    date_mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df[date_mask].copy()

    # Filter by time blocks if not 'all' (only applies to 15min and hourly)
    if selected_timeblocks and 'all' not in selected_timeblocks and agg_level in ['15min', 'hourly']:
        timeblock_mask = df_filtered['Time Block'].isin(selected_timeblocks)
        df_filtered = df_filtered[timeblock_mask]

    return df_filtered


# ---------- Main App ----------
def main():
    # Load data
    try:
        combined = load_and_preprocess_data()
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        st.info("Please make sure the following files are in the same directory:")
        st.write("- DAM_output.xlsx")
        st.write("- GDAM_output.xlsx")
        st.write("- Filtered_MCV_MCP_Data.xlsx")
        return
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return

    # Title
    st.markdown("""
    <h1 style='text-align: center; color: #2c3e50; font-family: Arial, sans-serif; margin-bottom: 30px;'>
    📈 Market Clearing Price (MCP) Dashboard
    </h1>
    """, unsafe_allow_html=True)

    # Display data info with performance metrics
    # min_date = combined['Date'].min().strftime('%d-%m-%Y')
    # max_date = combined['Date'].max().strftime('%d-%m-%Y')

    # total_records = len(combined)
    #
    # # Create info columns
    # info_col1, info_col2, info_col3 = st.columns(3)
    # with info_col1:
    #     st.metric("📅 Date Range", f"{min_date} to {max_date}")
    # with info_col2:
    #     st.metric("📊 Total Records", f"{total_records:,}")
    # with info_col3:
    #     markets = combined['Market'].nunique()
    #     st.metric("🏪 Markets", f"{markets}")

    # Create three columns for controls
    col1, col2, col3 = st.columns([2, 1.5, 2])

    with col1:
        st.markdown("**🗓️ Select Date Range:**")
        date_range = st.date_input(
            "",
            value=(combined['Date'].min().date(), combined['Date'].max().date()),
            min_value=combined['Date'].min().date(),
            max_value=combined['Date'].max().date(),
            key="date_range"
        )

        # Handle single date selection
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

    with col2:
        st.markdown("**📊 Time Aggregation:**")
        agg_level = st.selectbox(
            "",
            options=['15min', 'hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
            format_func=lambda x: {
                '15min': '15-Min Time Blocks',
                'hourly': 'Hourly Average',
                'daily': 'Daily Average',
                'weekly': 'Weekly Average',
                'monthly': 'Monthly Average',
                'quarterly': 'Quarterly Average',
                'yearly': 'Yearly Average'
            }[x],
            index=2,  # Default to 'daily'
            key="agg_level"
        )

    with col3:
        st.markdown("**🕓 Time Block Filter:**")
        time_blocks = sorted(combined['Time Block'].dropna().unique())
        selected_timeblocks = st.multiselect(
            "",
            options=['all'] + time_blocks,
            default=['all'],
            format_func=lambda x: 'All Time Blocks' if x == 'all' else f'Block {x}',
            key="timeblocks"
        )

    # Performance optimization: Show loading spinner for large datasets
    if agg_level in ['15min', 'hourly']:
        filtered_size = len(combined[
                                (combined['Date'] >= pd.to_datetime(start_date)) &
                                (combined['Date'] <= pd.to_datetime(end_date))
                                ])

        # if filtered_size > 5000:
        #     st.info(f"⚡ Processing {filtered_size:,} data points. Chart optimized for performance.")

    # Create and display chart
    try:
        # Prepare parameters for caching
        selected_timeblocks_str = ','.join(selected_timeblocks) if selected_timeblocks else ''
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Create chart with caching
        with st.spinner('Updating chart...'):
            fig = create_chart(
                combined,
                agg_level,
                selected_timeblocks_str,
                start_date_str,
                end_date_str
            )

        # Display the chart with optimized config
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'mcp_dashboard_{agg_level}_{start_date_str}',
                'height': 650,
                'width': 1200,
                'scale': 1
            }
        })

        # # Performance tips
        # if agg_level in ['15min', 'hourly']:
        #     st.markdown("""
        #     <div style='background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; margin: 20px 0;'>
        #     <h4 style='color: #1976D2; margin: 0 0 10px 0;'>⚡ Performance Tips:</h4>
        #     <ul style='margin: 0; color: #424242;'>
        #         <li>For large datasets (>10K points), data is intelligently downsampled for smooth interaction</li>
        #         <li>Use date range filters to focus on specific periods for better performance</li>
        #         <li>Daily/Weekly/Monthly views provide faster interaction for trend analysis</li>
        #         <li>Charts are cached to speed up repeated interactions</li>
        #     </ul>
        #     </div>
        #     """, unsafe_allow_html=True)

        # Tips section
        st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
        <p style='color: #7f8c8d; font-style: italic;'>
        💡 Tips: Use mouse wheel to zoom, drag to pan, double-click to reset zoom. Range selector buttons show data relative to your dataset's maximum date.
        </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error creating chart: {e}")
        st.error("Please check your data and try again.")


if __name__ == "__main__":
    main()

