# app.py
import base64
import io
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import firebase_admin
from firebase_admin import credentials, auth, firestore
import numpy as np
import pandas as pd
import pyrebase
import plotly.express as px
import json
import xml.etree.ElementTree as ET

# Initialize Flask app
flask_app = Flask(__name__)
flask_app.secret_key = "your_secret_key"
bcrypt = Bcrypt(flask_app)

# Firebase Admin SDK (for backend operations including Firestore)
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred)
db = firestore.client()  # This initializes Firestore

# Pyrebase (for client-side authentication)
firebase_config = {
    "apiKey": "AIzaSyApRzFpyWco9P1KCbH_WbIx603tBgWQyZ8",
    "authDomain": "smart-dash-544637.firebaseapp.com",
    "databaseURL": "https://smart-dash-544637-default-rtdb.firebaseio.com",  # Required by Pyrebase
    "projectId": "smart-dash-544637",
    "storageBucket": "smart-dash-544637.firebasestorage.app",
    "messagingSenderId": "59772427660",
    "appId": "1:59772427660:web:e4747b18ff740547430aa9"
}
firebase = pyrebase.initialize_app(firebase_config)
auth_client = firebase.auth()

# Dashboard class
class Dashboard:
    def __init__(self):
        self.app = Dash(
            __name__,
            server=flask_app,
            url_base_pathname="/dashboard/",
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        # Store the current dataframe and filename
        self.df = None
        self.filename = None  # Store the original filename
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = dbc.Container([
            html.H1("Smart Data Analysis Dashboard", className="text-center my-4"),
            
            # Upload Section
            dbc.Card([
                dbc.CardBody([
                    html.H4("Upload Your Data", className="text-center my-4"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select CSV File')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center'
                        },
                        multiple=False
                    )
                ])
            ], className="mb-4"),
            
            # Tabs for organizing content
            dbc.Tabs([
                dbc.Tab([
                    dbc.Spinner(html.Div(id='output-data-summary'))
                ], label="Dataset Summary"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Visualization Controls", className="mt-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.Label("Select Visualization Type:"),
                                    dcc.Dropdown(
                                        id='viz-type-dropdown',
                                        options=[
                                            {'label': 'Scatter Plot', 'value': 'scatter'},
                                            {'label': 'Line Plot', 'value': 'line'},
                                            {'label': 'Bar Chart', 'value': 'bar'},
                                            {'label': 'Box Plot', 'value': 'box'},
                                            {'label': 'Histogram', 'value': 'histogram'},
                                            {'label': 'Heatmap', 'value': 'heatmap'},
                                            {'label': 'Pie Chart', 'value': 'pie'}
                                        ],
                                        value='scatter'
                                    ),
                                    html.Label("X-Axis:", className="mt-3"),
                                    dcc.Dropdown(id='x-axis-dropdown'),
                                    html.Label("Y-Axis:", className="mt-3"),
                                    dcc.Dropdown(id='y-axis-dropdown'),
                                    html.Label("Color By (Optional):", className="mt-3"),
                                    dcc.Dropdown(id='color-dropdown', clearable=True)
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Spinner(html.Div(id='visualization-output'))
                        ], width=9)
                    ])
                ], label="Interactive Visualizations"),
                
                dbc.Tab([
                    dbc.Spinner(html.Div(id='output-suggested-viz'))
                ], label="Suggested Visualizations"),

                # History Tab
                dbc.Tab([
                    dbc.Spinner(html.Div(id='history-content'))
                ], label="History")
            ], className="mb-4"),
            
            # Error display
            html.Div(id='error-display')
        ])

    def parse_upload(self, contents, filename):
        """Parse uploaded file contents"""
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename.lower():
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename.lower():
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
            return df
        except Exception as e:
            print(f"Error parsing file: {str(e)}")
            return None
    
    def detect_data_types(self, df):
        """Detect and categorize column types with improved support for sales data"""
        data_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'sales': [],  # New type for sales-related columns
            'product': []  # New type for product-related columns
        }
        
        # First pass - identify basic types
        for column in df.columns:
            # Check for sales-related columns by name
            sales_keywords = ['sales', 'revenue', 'amount', 'price', 'total', 'volume', 'quantity', 'units']
            product_keywords = ['product', 'brand', 'item', 'category', 'type', 'beverage']
            
            col_lower = column.lower()
            
            # Check if column name contains date-related keywords
            date_keywords = ['date', 'time', 'year', 'month', 'day']
            if any(keyword in col_lower for keyword in date_keywords):
                try:
                    pd.to_datetime(df[column])
                    data_types['datetime'].append(column)
                    continue
                except:
                    pass
            
            # Check for sales/revenue columns
            if any(keyword in col_lower for keyword in sales_keywords):
                if pd.api.types.is_numeric_dtype(df[column]):
                    data_types['sales'].append(column)
                    data_types['numeric'].append(column)  # Also add to numeric for compatibility
                    continue
            
            # Check for product/category columns
            if any(keyword in col_lower for keyword in product_keywords):
                data_types['product'].append(column)
                if df[column].nunique() < len(df) * 0.5:  # If relatively few unique values
                    data_types['categorical'].append(column)
                continue
            
            # Check for numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                data_types['numeric'].append(column)
                continue
            
            # Check for categorical columns
            unique_ratio = df[column].nunique() / len(df)
            if unique_ratio < 0.2:  # If less than 20% unique values
                data_types['categorical'].append(column)
            else:
                # Try converting to numeric
                try:
                    pd.to_numeric(df[column])
                    data_types['numeric'].append(column)
                except:
                    if df[column].dtype == 'object':
                        if df[column].nunique() < 50:
                            data_types['categorical'].append(column)
                        else:
                            data_types['text'].append(column)
        
        return data_types
    
    def generate_dataset_summary(self, df):
        """Generate comprehensive dataset summary"""
        data_types = self.detect_data_types(df)
        
        summary = [
            html.H4("Dataset Overview"),
            dbc.Card(dbc.CardBody([
                html.P([
                    f"This dataset contains {df.shape[0]:,} records with {df.shape[1]} features. ",
                    "Here's what I found in your data:"
                ]),
                
                # Data composition
                html.H5("Data Composition", className="mt-3"),
                html.Ul([
                    html.Li(f"Numeric columns: {', '.join(data_types['numeric'])}") 
                        if data_types['numeric'] else None,
                    html.Li(f"Categorical columns: {', '.join(data_types['categorical'])}")
                        if data_types['categorical'] else None,
                    html.Li(f"DateTime columns: {', '.join(data_types['datetime'])}")
                        if data_types['datetime'] else None,
                    html.Li(f"Text columns: {', '.join(data_types['text'])}")
                        if data_types['text'] else None,
                ]),
                
                # Data quality
                html.H5("Data Quality", className="mt-3"),
                self.generate_quality_summary(df),
                
                # Statistical summary
                html.H5("Statistical Summary", className="mt-3"),
                self.generate_statistical_summary(df, data_types)
            ]))
        ]
        
        return summary

    def generate_quality_summary(self, df):
        """Generate data quality information"""
        missing_data = df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        
        quality_items = [
            html.P(f"Completeness: {(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.1f}% of all data points are present"),
        ]
        
        if not missing_cols.empty:
            quality_items.append(html.P("Columns with missing values:"))
            quality_items.append(html.Ul([
                html.Li(f"{col}: {count} missing values ({(count/df.shape[0])*100:.1f}%)")
                for col, count in missing_cols.items()
            ]))
        
        return html.Div(quality_items)

    def generate_statistical_summary(self, df, data_types):
        """Generate statistical summary for different data types"""
        summaries = []
        
        # Numeric summaries
        if data_types['numeric']:
            numeric_summary = df[data_types['numeric']].describe()
            summaries.append(html.Div([
                html.P("Numeric Columns Summary:"),
                dbc.Table.from_dataframe(
                    numeric_summary.round(2),
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3"
                )
            ]))
        
        # Categorical summaries
        for col in data_types['categorical']:
            value_counts = df[col].value_counts().head(5)
            summaries.append(html.Div([
                html.P(f"Top values for {col}:"),
                dbc.Table.from_dataframe(
                    pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values/len(df)*100).round(2)
                    }),
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3"
                )
            ]))
        
        return html.Div(summaries)

    def create_visualization(self, df, viz_type, x_col, y_col, color_col=None):
        """Create visualization based on selected parameters with added pie chart support"""
        try:
            if viz_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                            title=f"Scatter Plot: {y_col} vs {x_col}")
            elif viz_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, color=color_col,
                            title=f"Line Plot: {y_col} vs {x_col}")
            elif viz_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                        title=f"Bar Chart: {y_col} by {x_col}")
            elif viz_type == 'box':
                fig = px.box(df, x=x_col, y=y_col, color=color_col,
                        title=f"Box Plot: {y_col} by {x_col}")
            elif viz_type == 'histogram':
                fig = px.histogram(df, x=x_col, color=color_col,
                                title=f"Histogram of {x_col}")
            elif viz_type == 'heatmap':
                if y_col and color_col:
                    corr = df[[x_col, y_col, color_col]].corr()
                elif y_col:
                    corr = df[[x_col, y_col]].corr()
                else:
                    # For histogram only x is required
                    corr = df[x_col].corr(df[x_col])
                fig = px.imshow(corr, title="Correlation Heatmap")
            elif viz_type == 'pie':
                # For pie charts: x_col is categories, y_col is values
                if y_col:
                    # If we have values column, use it
                    if df[x_col].nunique() > 15:
                        # If too many categories, take top 10
                        grouped = df.groupby(x_col)[y_col].sum().nlargest(10).reset_index()
                        fig = px.pie(grouped, values=y_col, names=x_col, 
                                    title=f"Distribution of {y_col} by {x_col} (Top 10)")
                    else:
                        # Use all categories if reasonable number
                        grouped = df.groupby(x_col)[y_col].sum().reset_index()
                        fig = px.pie(grouped, values=y_col, names=x_col,
                                    title=f"Distribution of {y_col} by {x_col}")
                else:
                    # If no values column provided, use count
                    if df[x_col].nunique() > 15:
                        # If too many categories, take top 10
                        value_counts = df[x_col].value_counts().nlargest(10)
                        fig = px.pie(values=value_counts.values, names=value_counts.index,
                                    title=f"Distribution of {x_col} (Top 10)")
                    else:
                        # Use all categories if reasonable number
                        value_counts = df[x_col].value_counts()
                        fig = px.pie(values=value_counts.values, names=value_counts.index,
                                    title=f"Distribution of {x_col}")
            
            fig.update_layout(
                template='plotly_white',
                title_x=0.5,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return dcc.Graph(figure=fig)
        except Exception as e:
            return html.Div(f"Error creating visualization: {str(e)}")

    def suggest_visualizations(self, df):
        """Suggests visualizations with high accuracy, tailored for various datasets."""

        data_types = self.detect_data_types(df)
        suggestions = [html.H4("Recommended Visualizations", className="mb-4")]
        recommendations = []

        def is_nominal_or_ordinal(col):
            """Enhanced check for nominal/ordinal categorical data."""
            return (
                pd.api.types.is_categorical_dtype(df[col])
                or df[col].dtype == "object"
                and 2 <= df[col].nunique() <= 30  # More restrictive
            )

        def is_quantitative(col):
            """Robust check for quantitative data."""
            return pd.api.types.is_numeric_dtype(df[col])

        def is_temporal(col):
            """Reliable check for temporal data."""
            try:
                pd.to_datetime(df[col], errors="raise")
                return True
            except ValueError:
                return False

        def is_year_like(col):
            """Check if a numeric column might represent years."""
            if is_quantitative(col):
                min_val = df[col].min()
                max_val = df[col].max()
                if (
                    min_val >= 1900
                    and max_val <= 2050
                    and df[col].nunique() <= 50
                ):  # Reasonable year range and limited unique values
                    return True
            return False

        def calculate_entropy(series):
            """Calculate entropy to measure data diversity."""
            probabilities = series.value_counts(normalize=True)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))

        def calculate_correlation(df, col1, col2):
            """Calculate correlation between two quantitative columns."""
            try:
                return abs(df[col1].corr(df[col2]))
            except ValueError:  # Handle cases where correlation can't be computed
                return 0

        # Color palettes
        categorical_colors = px.colors.qualitative.Set2
        sequential_colors = px.colors.sequential.Blues
        diverging_colors = px.colors.diverging.RdYlGn

        # 0. Pie Charts: Precise Part-to-Whole (Color remains the same)
        if any(is_nominal_or_ordinal(col) for col in df.columns) and any(
            is_quantitative(col) for col in df.columns
        ):
            try:
                best_category_col = None
                best_value_col = None
                max_entropy = -1

                for cat_col in df.columns:
                    if (
                        is_nominal_or_ordinal(cat_col)
                        and 2 <= df[cat_col].nunique() <= 10  # Very strict
                    ):
                        entropy = calculate_entropy(df[cat_col])
                        if entropy > max_entropy:
                            max_entropy = entropy
                            best_category_col = cat_col

                if best_category_col:
                    for val_col in df.columns:
                        if is_quantitative(val_col):
                            if (
                                df[val_col].nunique() > 10
                            ):  # Quantitative should have enough variance
                                best_value_col = val_col
                                break

                if best_category_col and best_value_col:
                    fig = px.pie(
                        df,
                        names=best_category_col,
                        values=best_value_col,
                        title=f"Distribution of {best_value_col} by {best_category_col}",
                        color_discrete_sequence=categorical_colors,  # Color
                    )
                    fig.update_layout(template="plotly_white")
                    recommendations.append(
                        {
                            "score": 0.97,  # Highest score
                            "component": dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            f"Recommended: {best_value_col} by {best_category_col}"
                                        ),
                                        html.P(
                                            f"Part-to-whole relationship of {best_value_col} across {best_category_col}."
                                        ),
                                        dcc.Graph(figure=fig),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        }
                    )
            except Exception as e:
                print(f"Error creating pie chart: {str(e)}")

        # 1. Time Series: Accurate Trend Analysis (Color remains the same)
        if sum(is_temporal(col) for col in df.columns) == 1 and sum(
            is_quantitative(col) for col in df.columns
        ) == 1:  # Exactly one of each
            try:
                time_col = next(col for col in df.columns if is_temporal(col))
                value_col = next(col for col in df.columns if is_quantitative(col))

                # Further checks for time series suitability
                if df[time_col].nunique() > 10:  # Enough time points
                    fig = px.line(
                        df, x=time_col, y=value_col, title=f"Time Series of {value_col}",
                        color_discrete_sequence=[sequential_colors[5]],  # Color
                    )
                    fig.update_layout(template="plotly_white")
                    recommendations.append(
                        {
                            "score": 0.96,  # High score
                            "component": dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Recommended: Time Series"),
                                        html.P(f"Trends of {value_col} over time."),
                                        dcc.Graph(figure=fig),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        }
                    )
            except Exception as e:
                print(f"Error creating time series: {str(e)}")

        # 2. Bar Charts: Precise Category Comparison (Coloring with Graph Objects)
        if any(is_nominal_or_ordinal(col) for col in df.columns) and any(
            is_quantitative(col) for col in df.columns
        ):
            try:
                best_category_col = None
                best_value_col = None
                max_categories = 15  # Reasonable limit

                for cat_col in df.columns:
                    if is_nominal_or_ordinal(cat_col) or is_year_like(
                        cat_col
                    ):  # Include year-like
                        if (
                            2 <= df[cat_col].nunique() <= max_categories
                        ):  # Within limits
                            best_category_col = cat_col
                            break  # Take first suitable

                if best_category_col:
                    for val_col in df.columns:
                        if is_quantitative(val_col):
                            best_value_col = val_col
                            break  # Take first suitable

                if best_category_col and best_value_col:
                    num_categories = df[best_category_col].nunique()

                    # Smart Color Selection
                    if num_categories <= 10:
                        color_sequence = categorical_colors  # Use categorical palette
                    elif num_categories <= 20:
                        color_sequence = px.colors.qualitative.Dark24  # More colors
                    else:
                        color_sequence = px.colors.qualitative.Plotly  # Distinct, but not too harsh

                    # Create Plotly Graph Objects figure
                    import plotly.graph_objects as go
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=df[best_category_col],
                                y=df[best_value_col],
                                marker_color=color_sequence[:num_categories], # Explicit color list
                            )
                        ]
                    )

                    fig.update_layout(
                        title=f"{best_value_col} by {best_category_col}",
                        template="plotly_white",
                        showlegend=False,  # Legend not needed for simple bar
                    )

                    recommendations.append(
                        {
                            "score": 0.95,
                            "component": dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Recommended: Bar Chart"),
                                        html.P(
                                            f"Comparison of {best_value_col} across {best_category_col}."
                                        ),
                                        dcc.Graph(figure=fig),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        }
                    )
            except Exception as e:
                print(f"Error creating bar chart: {str(e)}")

        # 3. Scatter Plots: Accurate Correlation (Retail Focused) (Color remains the same)
        if sum(is_quantitative(col) for col in df.columns) >= 2:
            try:
                quantitative_cols = [
                    col for col in df.columns if is_quantitative(col)
                ]
                highest_corr = 0
                best_col1 = None
                best_col2 = None

                # Retail-specific column names (adjust as needed)
                retail_keywords = [
                    "price",
                    "cost",
                    "revenue",
                    "sales",
                    "quantity",
                    "discount",
                    "profit",
                ]

                # Prioritize retail-related columns
                retail_cols = [
                    col
                    for col in quantitative_cols
                    if any(keyword in col.lower() for keyword in retail_keywords)
                ]
                non_retail_cols = [
                    col for col in quantitative_cols if col not in retail_cols
                ]

                # If we have retail cols, prioritize those
                cols_to_use = retail_cols if retail_cols else quantitative_cols

                for i in range(len(cols_to_use)):
                    for j in range(i + 1, len(cols_to_use)):
                        corr = calculate_correlation(
                            df, cols_to_use[i], cols_to_use[j]
                        )
                        if corr > highest_corr and corr > 0.5:  # Threshold
                            highest_corr = corr
                            best_col1 = cols_to_use[i]
                            best_col2 = cols_to_use[j]

                if best_col1 and best_col2:
                    fig = px.scatter(
                        df,
                        x=best_col1,
                        y=best_col2,
                        title=f"Relationship: {best_col1} vs {best_col2}",
                        color_discrete_sequence=[sequential_colors[7]],  # Color
                    )
                    fig.update_layout(template="plotly_white")
                    recommendations.append(
                        {
                            "score": 0.94,
                            "component": dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Recommended: Scatter Plot"),
                                        html.P(
                                            f"Correlation between {best_col1} and {best_col2}."
                                        ),
                                        dcc.Graph(figure=fig),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        }
                    )
            except Exception as e:
                print(f"Error creating scatter: {str(e)}")

        # 4. Histograms: Distribution Analysis (Color remains the same)
        if sum(is_quantitative(col) for col in df.columns) == 1:
            try:
                quantitative_col = next(
                    col for col in df.columns if is_quantitative(col)
                )
                fig = px.histogram(
                    df, x=quantitative_col, title=f"Distribution of {quantitative_col}",
                    color_discrete_sequence=[sequential_colors[3]],  # Color
                )
                fig.update_layout(template="plotly_white")
                recommendations.append(
                    {
                        "score": 0.85,
                        "component": dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Recommended: Histogram"),
                                    html.P(f"Distribution of values in {quantitative_col}."),
                                    dcc.Graph(figure=fig),
                                ]
                            ),
                            className="mb-4",
                        )
                    }
                )
            except Exception as e:
                print(f"Error creating histogram: {str(e)}")

        # 5. Heatmap: Correlation Matrix (Selective) (Color remains the same)
        if sum(is_quantitative(col) for col in df.columns) > 2:
            try:
                quantitative_cols = [
                    col for col in df.columns if is_quantitative(col)
                ]
                # Limit for heatmap
                num_cols_for_heatmap = quantitative_cols[: min(10, len(quantitative_cols))]
                corr = df[num_cols_for_heatmap].corr()
                if (
                    abs(corr.values).max() > 0.6
                ):  # Strong correlation somewhere - important
                    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap",
                                   color_continuous_scale=diverging_colors)  # Color
                    fig.update_layout(template="plotly_white")
                    recommendations.append(
                        {
                            "score": 0.82,
                            "component": dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Recommended: Correlation Heatmap"),
                                        html.P(
                                            "Heatmap of correlations between quantitative variables."
                                        ),
                                        dcc.Graph(figure=fig),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        }
                    )
            except Exception as e:
                print(f"Error creating heatmap: {str(e)}")

        # Finalize and Filter
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        final_recommendations = [
            rec["component"] for rec in recommendations if rec["score"] > 0.90
        ]  # Higher threshold
        final_recommendations = final_recommendations[:3]  # Top 3

        if final_recommendations:
            suggestions.extend(final_recommendations)
        else:
            suggestions.append(
                dbc.Alert(
                    "No significant patterns found. Explore data manually.",
                    color="info",
                )
            )

        return html.Div(suggestions)

    def setup_callbacks(self):
        @self.app.callback(
            [Output('x-axis-dropdown', 'options'),
             Output('y-axis-dropdown', 'options'),
             Output('color-dropdown', 'options'),
             Output('x-axis-dropdown', 'value'),
             Output('y-axis-dropdown', 'value')],
            Input('upload-data', 'contents'),
            State('upload-data', 'filename')
        )
        def update_dropdowns(contents, filename):
            if contents is None:
                return [], [], [], None, None
                
            try:
                df = self.parse_upload(contents, filename)
                if df is None:
                    raise ValueError("Failed to parse file")
                
                # Store the dataframe and filename in the class instance
                self.df = df
                self.filename = filename  # Store the original filename
                print(f"File uploaded: {filename}")
                    
                # Create dropdown options from column names
                options = [{'label': col, 'value': col} for col in df.columns]
                
                # Set default values for axes
                default_x = df.columns[0] if len(df.columns) > 0 else None
                default_y = df.columns[1] if len(df.columns) > 1 else default_x
                
                print(f"Dropdown options populated with {len(options)} columns")
                print(f"Default X: {default_x}, Default Y: {default_y}")
                
                return options, options, options, default_x, default_y
            except Exception as e:
                print(f"Error updating dropdowns: {str(e)}")
                return [], [], [], None, None

        @self.app.callback(
            [Output('output-data-summary', 'children'),
            Output('output-suggested-viz', 'children')],
            Input('upload-data', 'contents'),
            State('upload-data', 'filename')
        )
        def update_output(contents, filename):
            if contents is None:
                return None, None

            try:
                df = self.parse_upload(contents, filename)
                if df is None:
                    raise ValueError("Failed to parse file")

                # Store the dataframe and filename
                self.df = df
                self.filename = filename
                
                # Generate outputs
                summary = self.generate_dataset_summary(df)
                suggestions = self.suggest_visualizations(df)

                # Save to Firestore - store initial file upload
                try:
                    if "user" in session:
                        user_email = session["user"]
                        
                        # Create a simple summary of the upload
                        #summary_text = f"Dataset upload: {df.shape[0]} rows, {df.shape[1]} columns."
                        
                        # Prepare data for Firestore
                        history_data = {
                            "user_email": user_email,
                            "filename": filename,  # Use the actual filename
                            "summary": summary,
                            "timestamp": firestore.SERVER_TIMESTAMP,
                            "columns": list(df.columns)[:20],
                            "action": "upload"
                        }
                        
                        # Add the document to Firestore
                        db.collection("history").add(history_data)
                        print(f"Successfully saved upload to history for user {user_email}")
                except Exception as firestore_error:
                    # Don't break visualization if history saving fails
                    print(f"Error saving upload to Firestore: {str(firestore_error)}")

                return summary, suggestions

            except Exception as e:
                error_msg = dbc.Alert(
                    f"Error processing file: {str(e)}",
                    color="danger",
                    dismissable=True
                )
                return error_msg, error_msg

        @self.app.callback(
            Output('history-content', 'children'),
            Input('upload-data', 'contents')
        )
        def update_history_tab(contents):
            if "user" not in session:
                return html.Div("Please log in to view your history.", className="text-danger")

            try:
                user_email = session["user"]
                print(f"Fetching history for user: {user_email}")
                
                # Query Firestore
                try:
                    history_ref = db.collection("history").where("user_email", "==", user_email)
                    history_ref = history_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)
                    history_docs = history_ref.stream()

                    # Build history content
                    history_items = []
                    for doc in history_docs:
                        history = doc.to_dict()
                        # Handle timestamp - convert to string if it's a Firestore timestamp
                        timestamp = history.get('timestamp')
                        if hasattr(timestamp, 'strftime'):  # Check if it's a datetime-like object
                            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            timestamp_str = str(timestamp)
                            
                        history_items.append(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5(f"File: {history.get('filename', 'N/A')}", className="card-title"),
                                    html.P(f"Summary: {history.get('summary', 'N/A')}", className="card-text"),
                                    html.P(f"Timestamp: {timestamp_str}", className="card-text")
                                ])
                            ], className="mb-3")
                        )

                    if not history_items:
                        return html.Div("No history found. Upload some data to get started!", className="text-muted p-3")

                    return html.Div(history_items)
                    
                except Exception as firestore_error:
                    print(f"Firestore error: {str(firestore_error)}")
                    return html.Div(f"Error retrieving history: {str(firestore_error)}", className="text-danger")

            except Exception as e:
                print(f"Error in update_history_tab: {str(e)}")
                return html.Div(f"Error: {str(e)}", className="text-danger")

        @self.app.callback(
            Output('visualization-output', 'children'),
            [Input('viz-type-dropdown', 'value'),
            Input('x-axis-dropdown', 'value'),
            Input('y-axis-dropdown', 'value'),
            Input('color-dropdown', 'value')],
        )
        def update_visualization(viz_type, x_col, y_col, color_col):
            # Check if we have the required data
            if self.df is None:
                return html.Div("Please upload data first", className="text-warning p-3, text-center")
                
            if not viz_type or not x_col:
                return html.Div("Please select visualization parameters", className="text-warning p-3, text-center")

            try:
                print(f"Creating visualization: {viz_type} with X={x_col}, Y={y_col}, Color={color_col}")
                # Generate the visualization using the stored dataframe
                visualization_component = self.create_visualization(self.df, viz_type, x_col, y_col, color_col)

                # Save to Firestore
                try:
                    if "user" in session:
                        user_email = session["user"]
                        
                        # Create a simple summary
                        summary = f"Dataset with {self.df.shape[0]} rows, {self.df.shape[1]} columns. "
                        if viz_type == 'pie':
                            if y_col:
                                summary += f"Visualization: {viz_type} chart of {y_col} by {x_col}"
                            else:
                                summary += f"Visualization: {viz_type} chart of {x_col} distribution"
                        elif y_col:
                            summary += f"Visualization: {viz_type} of {y_col} vs {x_col}" 
                        else:
                            summary += f"Visualization: {viz_type} of {x_col}"

                        # Prepare data for Firestore
                        history_data = {
                            "user_email": user_email,
                            "filename": self.filename if self.filename else "unnamed_data.csv",  # Use stored filename
                            "summary": summary,
                            "timestamp": firestore.SERVER_TIMESTAMP,
                            "columns": list(self.df.columns)[:20],
                            "viz_type": viz_type,
                            "action": "visualization"
                        }
                        
                        # Add the document to Firestore
                        db.collection("history").add(history_data)
                        print(f"Successfully saved visualization to history for user {user_email}")
                except Exception as firestore_error:
                    # Don't break visualization if history saving fails
                    print(f"Error saving to Firestore: {str(firestore_error)}")

                return visualization_component

            except Exception as e:
                print(f"Error in update_visualization: {str(e)}")
                return html.Div(f"Error creating visualization: {str(e)}", className="text-danger")

# Flask Routes for Login/Signup
@flask_app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@flask_app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        print(f"Signup attempt: {email}")  # Debugging log
        try:
            # Create user in Firebase
            user = auth_client.create_user_with_email_and_password(email, password)
            print("Signup successful!")
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("home"))
        except Exception as e:
            error_message = str(e)
            print(f"Signup error: {error_message}")

            # Check if the error is due to an existing user
            if "EMAIL_EXISTS" in error_message:
                error_message = "This email is already in use. Please log in or use a different email."
            else:
                error_message = "An error occurred during signup. Please try again."

            # Render the signup page with the error message
            return render_template("signup.html", error=error_message)
    return render_template("signup.html", error=None)

@flask_app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        print(f"Login attempt: {email}")
        try:
            # Authenticate user
            user = auth_client.sign_in_with_email_and_password(email, password)
            session["user"] = email  # Store user email in session
            print("Login successful, redirecting to dashboard...")
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        except Exception as e:
            print(f"Login error: {str(e)}")
            flash(f"Login failed: Invalid email or password", "danger")
    return render_template("login.html")

@flask_app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@flask_app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for("home"))
    return render_template("dashboard.html")

@flask_app.route("/history")
def history():
    if "user" not in session:
        flash("Please log in to view your history.", "warning")
        return redirect(url_for("home"))

    try:
        user_email = session["user"]
        
        # Query Firestore
        try:
            history_ref = db.collection("history").where("user_email", "==", user_email)
            history_ref = history_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)
            history_docs = history_ref.stream()

            history_data = []
            for doc in history_docs:
                history_dict = doc.to_dict()
                
                # Handle Firestore timestamps properly
                timestamp = history_dict.get('timestamp')
                if hasattr(timestamp, 'strftime'):  # Check if it's a datetime-like object
                    history_dict['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                history_data.append(history_dict)

            if not history_data:
                message = "You have no history yet. Please try uploading some data!"
            else:
                message = f"Found {len(history_data)} historical analyses for your account."

            return render_template("history.html", history=history_data, message=message)
            
        except Exception as firestore_error:
            print(f"Firestore error in history route: {str(firestore_error)}")
            flash(f"Error retrieving history: {str(firestore_error)}", "danger")
            return render_template("history.html", history=[], message="Error retrieving history.")
    
    except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return render_template("history.html", history=[], message="Error retrieving history.")

# Run the app
if __name__ == "__main__":
    dashboard = Dashboard()
    flask_app.run(debug=True, port=5001)