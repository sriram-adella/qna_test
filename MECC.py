# fetching data:
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from tqdm import tqdm
import plotly.graph_objs as go
import json
import re
from bs4 import BeautifulSoup
import traceback
import numpy as np

pd.options.mode.chained_assignment = None  # Suppresses SettingWithCopyWarning

def adjust_date_to_noon(row):
    report_time = row['Report Date'].time()
    if report_time > pd.to_datetime("12:00:00").time():
        return row['Report Date'].date() + pd.Timedelta(days=1)
    else:
        return row['Report Date'].date()

def last_non_nan(series):
        return series.dropna().iloc[-1] if not series.dropna().empty else None


def find_columns_with_consecutive_zeros(df, threshold=3):
    columns_with_consecutive_zeros = []
    
    for column in df.columns:
        # Create a boolean series where True indicates a 0 value
        zero_series = df[column] == 0
        # Use rolling window to find consecutive zeros
        consecutive_zeros = zero_series.rolling(window=threshold).sum()
        
        # Check if any rolling window has the threshold number of zeros
        if (consecutive_zeros >= threshold).any():
            columns_with_consecutive_zeros.append(column)
    
    return columns_with_consecutive_zeros

def convert_numbers_to_float(dictionary):
    for key, val in dictionary.items():
        # If the value is a dictionary, recurse into it
        if isinstance(val, dict):
            convert_numbers_to_float(val)
        # Check if the value is a numeric type (e.g., np.int64 or np.float64)
        elif isinstance(val, np.int64):
            dictionary[key] = int(val)
        elif isinstance(val, np.float64):
            dictionary[key] = float(val)
    return dictionary
# excecution:
def mecc_output(imo, df ):
    if imo != "":
        #imo wise dataframes:
        df = df[df["IMO No"] == imo] 

    df.drop_duplicates(inplace = True)
    df.sort_values("Report Date", inplace = True, ascending = True)
    df.reset_index(inplace=True,drop = True)

    remarks = ""
    #extracting MECC and AECC data:
    df_mecc = df[['Report Date',"Vessel Name","IMO No","AECC consumption (LTRS)", "MECC consumption (LTRS)","ROB AECC","ROB MECC"]]
    if df_mecc.empty or len(df_mecc) < 2:
        out_dict = {
            "Date": "Data Not Available",
            "current_mecc": "Data Not Available",
            "current_aecc": "Data Not Available",
            "rob_mecc": "Data Not Available",
            "rob_aecc": "Data Not Available",
            "avg_mecc": "Data Not Available",
            "avg_aecc": "Data Not Available",
            "Remarks": "Data Not Available"
        }
        last_6_months = None
        return out_dict, last_6_months
        
    df_mecc["Date"] = pd.to_datetime(df["Report Date"]).dt.strftime('%Y-%m-%d')
    df_mecc.sort_values("Date", inplace = True, ascending = True)

    #replacing nan values by 0 in consumption columns:
    df_mecc[["AECC consumption (LTRS)", "MECC consumption (LTRS)" ]] = df_mecc[["AECC consumption (LTRS)", "MECC consumption (LTRS)"]].fillna(0)
    
    agg_functions = {
        "AECC consumption (LTRS)": 'sum',
        'MECC consumption (LTRS)': 'sum', 
    }

    df_mecc['Date'] = df_mecc.apply(adjust_date_to_noon, axis=1) 

    noon_df = df_mecc[df_mecc['Report Date'].dt.time == pd.to_datetime('12:00:00').time()].sort_values(by = 'Report Date')
    if noon_df.empty:
        latest_date =df_mecc["Report Date"].dt.date.max()
    else:
        latest_date = noon_df["Report Date"].dt.date.max()
    formatted_latest_date = latest_date.strftime('%d %b %Y')

    # Current Consumption:
    current_consumption_mecc  = df_mecc[df_mecc["Date"] == latest_date]["MECC consumption (LTRS)"].sum()
    current_consumption_aecc  = df_mecc[df_mecc["Date"] == latest_date]["AECC consumption (LTRS)"].sum()

    new_df = df_mecc[df_mecc["Date"] == latest_date]

    new_df = df_mecc[df_mecc["Date"] == latest_date]

    current_rob_mecc = new_df["ROB MECC"].replace(0, pd.NA).dropna().iloc[-1] if not new_df["ROB MECC"].replace(0, pd.NA).dropna().empty else "Data Not Available"
    current_rob_aecc = new_df["ROB AECC"].replace(0, pd.NA).dropna().iloc[-1] if not new_df["ROB AECC"].replace(0, pd.NA).dropna().empty else "Data Not Available"


    #past 6 months consumption:
    df_mecc["Months"] = df_mecc["Report Date"].dt.to_period('M').dt.to_timestamp()
    #total mecc sonsumption monthly:
    df_monthly = df_mecc.groupby("Months").agg(agg_functions).reset_index()
    df_monthly.sort_values("Months", inplace = True)
    last_6_months = df_monthly.tail(7)
    current_month_year = pd.Timestamp(latest_date.strftime('%Y-%m-01'))
    # Filter rows where Month_year is not equal to the current month and year
    df_filtered = last_6_months[last_6_months["Months"] != current_month_year]
    if df_filtered.empty:
        last_6_months['Months'] = last_6_months["Months"].dt.strftime('%b-%Y')

        out_dict = {
            "Date": formatted_latest_date,
            "current_mecc": current_consumption_mecc,
            "current_aecc": current_consumption_aecc,
            "rob_mecc": current_rob_mecc,
            "rob_aecc": current_rob_aecc,
            "avg_mecc": "Cannot be calculated",
            "avg_aecc": "Cannot be calculated",
            "Remarks": "Incomplete data"
        }
        return out_dict, last_6_months


    columns = find_columns_with_consecutive_zeros(df_filtered[["MECC consumption (LTRS)", "AECC consumption (LTRS)"]], threshold=3)
    if "AECC consumption (LTRS)" in columns:
        avg_aecc = "Incorrect data"
    else:
        avg_aecc = df_filtered["AECC consumption (LTRS)"].mean().round(2)
    if "MECC consumption (LTRS)" in columns:
        avg_mecc = "Incorrect data"
    else:
        avg_mecc = df_filtered["MECC consumption (LTRS)"].mean().round(2)


    first_month = df_monthly['Months'].min()
    last_month = df_monthly['Months'].max()
    # Create a complete date range with monthly frequency
    complete_range = pd.date_range(first_month, last_month, freq='MS')  # MS for month start

    # Check if there are any missing months
    if len(complete_range) != len(df_monthly):
        remarks = "Incomplete data"
 

    last_6_months['Months'] = last_6_months["Months"].dt.strftime('%b-%Y')
    date_range = f"01-{latest_date.strftime('%b')} to {latest_date.strftime('%d-%b %Y')}"
    last_6_months.at[last_6_months.index[-1], 'Months'] = date_range
    out_dict = {
            "Date": formatted_latest_date,
            "current_mecc": current_consumption_mecc,
            "current_aecc": current_consumption_aecc,
            "rob_mecc": current_rob_mecc,
            "rob_aecc": current_rob_aecc,
            "avg_mecc": avg_mecc,
            "avg_aecc": avg_aecc,
            "Remarks": remarks
        }
    
    return out_dict, last_6_months


def mecc_summary_main(json_input):
    #json to dataframe:
    df = pd.DataFrame(json_input)
    df.drop_duplicates(inplace = True)
    df.dropna(subset = "Report Date", inplace = True)
    df.sort_values("Report Date", inplace = True, ascending = True)
    df.reset_index(inplace=True,drop = True)
    imo = ""
    output, _ = mecc_output(imo , df)
    if output["rob_mecc"] == "Data Not Available":
        output["rob_mecc"] = "Incorrect data"
    if output["rob_aecc"] == "Data Not Available":
        output["rob_aecc"] = "Incorrect data"
    out_dict = convert_numbers_to_float(output)
    return out_dict


def markdown(out_,df,imo):
    output = out_.copy()
    if output["Remarks"] == "Data Not Available": 
        return "Data Not Available"  
    
    output["current_mecc"] = f'''{output["current_mecc"]} Ltrs''' if output["current_mecc"] != 0 else "No Consumption"
    output["current_aecc"] = f'''{output["current_aecc"]} Ltrs''' if output["current_aecc"] != 0 else "No Consumption"
    output["rob_mecc"] = f'''{output["rob_mecc"]} MT''' if output["rob_mecc"] != "Data Not Available" else "Data Not Available"
    output["rob_aecc"] = f'''{output["rob_aecc"]} MT''' if output["rob_aecc"] != "Data Not Available" else "Data Not Available"

    if output["avg_mecc"] in ["Incorrect data", "Cannot be calculated"]:
        output["avg_mecc"] = "Cannot be calculated based on the data available"
    else:
        output["avg_mecc"] = f'''{output["avg_mecc"]} Ltrs/Month'''
    if output["avg_aecc"] in ["Incorrect data", "Cannot be calculated"]:
        output["avg_aecc"] = "Cannot be calculated based on the data available"
    else:
        output["avg_aecc"] = f'''{output["avg_aecc"]} Ltrs/Month'''

    
###########    
    monthly_table = '''
| Months | MECC | AECC |
| --- | --- | --- |
'''
    # Loop through the first 6 rows of the DataFrame and add them to the HTML table
    for _, row in df.iterrows():
        monthly_table += f'''| {row["Months"]} | {row["MECC consumption (LTRS)"]} | {row["AECC consumption (LTRS)"]} |\n'''
###########    

    
    


    if output["avg_mecc"] == "Cannot be calculated based on the data available" :
        cal_mecc = output["avg_mecc"]
    else:
        cal_mecc = f'''$$\\text{{Average Monthly Consumption}} =\\frac{{ {df["MECC consumption (LTRS)"][:-1].sum()}\ Ltrs}}{{{len(df)-1}\ Months}} = {output["avg_mecc"]}$$'''

    if output["avg_aecc"] == "Cannot be calculated based on the data available" :
        cal_aecc = output["avg_aecc"]
    else:
        cal_aecc = f'''$$\\text{{Average Monthly Consumption}} =\\frac{{ {df["AECC consumption (LTRS)"][:-1].sum()}\ Ltrs}}{{{len(df)-1} \ Months}} = {output["avg_aecc"]}$$'''

    calculation = f'''$$\\text{{Average Monthly Consumption}} =\\frac{{\\text{{Total Consumption Over Months}}}}{{\\text{{Number of Months}}}} $$
- For MECC:  
{cal_mecc}
- For AECC:  
{cal_aecc}'''

#########
    out_str = f'''## MECC and AECC Consumption and ROB from Ship Palm consumption log <br><br> 
**Date**: {output["Date"]} (Latest available Data)\n

### Quick Summary:  
- **Current Consumption MECC** : {output["current_mecc"]}
- **Current Consumption AECC** : {output["current_aecc"]}
- **Current ROB MECC** : {output["rob_mecc"]}
- **Current ROB AECC** : {output["rob_aecc"]}
> - **Average Monthly Consumption MECC** : {output["avg_mecc"]} 
> - **Average Monthly Consumption AECC** : {output["avg_aecc"]}

### Data Source:   
- Ship Palm {"V3" if v3_status.find_one({"imo" : int(imo)}) else "V2"}
### Data Fields:
- MECC (Main Engine System Oil) Consumption
- AECC (Auxiliary Engine System Oil)onsumption
- ROB (Remaining On Board)

### Step 1: Data Summary
The current consumption and Remaining On Board (ROB) data are summarized as follows:
| Parameter | MECC | AECC |
| --- | --- | --- |
| Current Consumption | {output["current_mecc"]} | {output["current_aecc"]} |
| Current ROB | {output["rob_mecc"]} | {output["rob_aecc"]} |

### Step 2: Monthly Consumption for Past 6 Months:
We calculate the total monthly consumption for  the past 6 months by adding the daily consumptions.
{monthly_table}

### Step 3: Average Monthly Consumption:    
We calculate the average monthly consumption based on the formula:
{calculation}  
Assumptions:
- We consider the past 6 months (if available) excluding the current month

### Step 4: Logical Assumptions & Handling Special Cases:
- If Consumption data is not available, we consider it as 0.
- If Consumption data is 0 for 3 consecutive months, we consider it as incorrect data.
'''


    return out_str


time8 = datetime.now()
# function to plot the data:
def plots_data(df_mecc):    
    #replacing nan values by 0 in consumption columns:
    df_mecc[["Steaming time (HRS)","AECC consumption (LTRS)", "MECC consumption (LTRS)" ]] = df_mecc[["Steaming time (HRS)","AECC consumption (LTRS)", "MECC consumption (LTRS)"]].fillna(0)
    
    df_mecc['Date'] = df_mecc.apply(adjust_date_to_noon, axis=1)

    # getting values for unique days:
    def last_non_nan(series):
        return series.dropna().iloc[-1] if not series.dropna().empty else None
    
 
    agg_functions = {
        "AECC consumption (LTRS)": 'sum',
        'MECC consumption (LTRS)': 'sum',
        'ROB AECC': last_non_nan,  # Custom function for minimum non-NaN value
        'ROB MECC': last_non_nan,  # Custom function for minimum non-NaN value
        "Steaming time (HRS)": 'sum'
    }
 
    # Group by 'Date' and apply the aggregation functions
    df_mecc.sort_values("Report Date", inplace = True, ascending = True)
    df_mecc_daily = df_mecc.groupby('Date').agg(agg_functions).reset_index()
    #monthly Consumption and Running Hours:
    df_mecc_daily["Month"] = pd.to_datetime(df_mecc_daily["Date"]).dt.strftime('%Y-%m')
    df_mecc_monthly = df_mecc_daily.groupby("Month").agg({"MECC consumption (LTRS)":'sum',"AECC consumption (LTRS)":'sum',"Steaming time (HRS)":"sum"}).reset_index()
    df_mecc_monthly = df_mecc_monthly.sort_values("Month")
 
    # Change the format of months to 'MMM-YYYY'
    df_mecc_monthly["Month"] = pd.to_datetime(df_mecc_monthly["Month"]).dt.strftime('%b-%Y')
    # Get the latest 6 months' data
    latest_6_months = df_mecc_monthly.tail(6)
        
    #Rob Trend:
    rob_trend_df = df_mecc_daily[["Date","ROB MECC","ROB AECC"]].copy()
    rob_trend_df.dropna(subset = ["ROB MECC","ROB AECC"], inplace = True)
    
    return latest_6_months, rob_trend_df



# Function to plot MECC Consumption and Steaming Time
def plot_mecc_consumption_grouped(plot_1_df):
    fig = go.Figure()

    # Add MECC Consumption Bar
    fig.add_trace(go.Bar(
        x=plot_1_df['Month'], 
        y=plot_1_df['MECC consumption (LTRS)'], 
        name='MECC Consumption', 
        marker_color='blue'
    ))

    # Add Steaming Time as another Bar (instead of a line)
    fig.add_trace(go.Bar(
        x=plot_1_df['Month'], 
        y=plot_1_df['Steaming time (HRS)'], 
        name='Steaming Time', 
        marker_color='green'
    ))

    # Set Layout for Grouped Bars
    fig.update_layout(
        title="ME System Oil Consumption and Steaming Time",
        xaxis_title="Month",
        yaxis_title="Consumption/Time",
        barmode='group',
        height = 400

        
    )
    
    fig_json = fig.to_json()
    fig_dict = json.loads(fig_json)
    return fig_dict

# Function to plot AECC Consumption only
def plot_aecc_consumption_grouped(plot_2_df):
    fig = go.Figure()

    # Add AECC Consumption Bar
    fig.add_trace(go.Bar(
        x=plot_2_df['Month'], 
        y=plot_2_df['AECC consumption (LTRS)'], 
        name='AECC Consumption', 
        marker_color='blue'
    ))

    # Set Layout for the Bar
    fig.update_layout(
        title="AE System Oil Consumption",
        xaxis_title="Month",
        yaxis_title="AECC Consumption (LTRS)",
        barmode='group',
        height = 400

        
    )
    
    fig_json = fig.to_json()
    fig_dict = json.loads(fig_json)
    return fig_dict

 

def plot_rob(p2):    
# Create a figure
    fig = go.Figure()

    # Add traces for each fuel column with markers
    fig.add_trace(go.Scatter(
        x=p2['Date'], 
        y=p2['ROB MECC'], 
        mode='lines', 
        name='ROB MECC',
        hovertemplate='%{y:.2f} MT'
    ))

    fig.add_trace(go.Scatter(
        x=p2['Date'], 
        y=p2['ROB AECC'], 
        mode='lines', 
        name='ROB AECC',
        hovertemplate='%{y:.2f} MT'
    ))


    # Get the max date in the data (the most recent date)
    max_date = p2['Date'].max()

    # Customize layout to include a range slider, range selector buttons, and adjust the plot size
    fig.update_layout(
        title=dict(
        text='System ROBs MECC/AECC'),
        annotations=[
        dict(
            text='',  # Subtitle text goes here
            xref='paper', yref='paper',
            x=0.5, y=1.05,  # Positioning the subtitle just below the title
            showarrow=False,
            font=dict(size=12, color="grey"),  # Styling the subtitle
            xanchor='center'
        )],
       
        xaxis_title='Date',
        yaxis_title='Remaining On Board (MT)',
        legend_title='',
        hovermode='x unified',  # Ensures all values for a single date appear together on hover
        xaxis=dict(
            rangeslider=dict(
                visible=True  # Show the range slider at the bottom
            ),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            ),
            type='date',  # Ensure that the x-axis is treated as dates
            range=[max_date - pd.DateOffset(months=1), max_date]  # Set default view to the last 1 month
        ),
        plot_bgcolor='white',  # Set background color for the plot area
        paper_bgcolor='white',  # Set background color for the paper (outside the plot)
        height = 400
        
    )

    fig_json = fig.to_json()
    # Convert the JSON string to a dictionary for MongoDB insertion
    fig_dict = json.loads(fig_json)
    return fig_dict



### END OF FUNCTION DEFINITIONS ###

#fetching data:
time1 = datetime.now()
#connecting to mongodb and getting data for all the vessels:
mongo_uri = "mongodb://syia-etl-dev-readonly:S42tH5iVm3H8@db.syia.ai/?authMechanism=DEFAULT&authSource=syia-etl-dev"
client = MongoClient(mongo_uri)
db = client.get_database("syia-etl-dev")


#getting the list of active IMOs:
time5 = datetime.now()
collection_active = db.get_collection('common_vessel_details')
df_active = pd.DataFrame(collection_active.find({ "status":'ACTIVE'}, {
    "_id": 1,
    "imo": 1
}))
active_imos = df_active['imo'].tolist()
time6 = datetime.now()
print("Fetched active IMOs")
print(f"Time taken to fetch active IMOs is {time6-time5}")

#fetching v3 or v2 status:
v3_status = db.get_collection("common_v3_vessel_status")


#data from common consumption log api:
time1 = datetime.now()
collection = db.get_collection("common_consumption_log_api")
last_year_end = datetime(datetime.now().year -1, 12, 31) 
twelve_months_ago = pd.to_datetime('today') - pd.DateOffset(months=12)
query = {
            "IMO No": {"$in": active_imos},
            "Report Date": {"$gte": twelve_months_ago}
        }
projection = {
    '_id': 0,
    'Vessel Name': 1,
    'IMO No': 1,
    "Report Date" : 1,
    'data':{
        "Steaming time (HRS)":1,
        "AECC consumption (LTRS)":1,
        "MECC consumption (LTRS)":1,
        "ROB AECC":1,
        "ROB MECC":1
    }  
}
df_data = pd.DataFrame(collection.find(query, projection))
df_data = pd.concat([df_data,df_data['data'].apply(pd.Series)], axis=1)
df_data.drop("data",axis = 1, inplace = True)
df_data.dropna(subset = "Report Date", inplace = True)

time2 = datetime.now()
print("Fetched data from MongoDB")
print(f"Time taken to fetch data from MongoDB is {time2-time1}")

vessel_ids = db.get_collection("common_vessels").find({}, {"imo": 1, "_id": 1, "name":1})
vessel_id_df = pd.DataFrame(vessel_ids)
vessel_id_df['_id'] = vessel_id_df['_id'].astype(str)


# pushing to mogodb:
mongo_uri_app_app = "mongodb://answer_update:L4pzye64RF3u@db-etl.prod.syia.ai:27017/?authSource=syia-etl-prod"
client_app = MongoClient(mongo_uri_app_app)
db_app = client_app.get_database("syia-etl-prod")
vesselinfo = db_app.get_collection("vesselinfos")
vesselinfocomponents = db_app.get_collection("vesselinfocomponents")


synrgy_etl=MongoClient("mongodb://synergy-core-etl:rsjPVIO4383h@db-etl.prod.syia.ai:27017/?authSource=synergy-core-etl")['synergy-core-etl']
vesselinfo_synergy=synrgy_etl['vesselinfos']
vesselinfocomponents_synergy=synrgy_etl['vesselinfocomponents']


#derived parameters:
mongo_client_etl_dev = MongoClient(r'mongodb://rohan.sharma:jL6g0bd90PjL@db.syia.ai/?authMechanism=DEFAULT&authSource=syia-etl-dev')
etl_dev = mongo_client_etl_dev['syia-etl-dev']
derived_params = etl_dev['derived_parameters']


# creating output table:
output_table_1 = pd.DataFrame(columns=["imo", "markdown_answer","data"])
valid_imos = list(output_table_1["imo"].tolist())
failed = {}

for imo in tqdm(active_imos):
    try:
        output_dict, df = mecc_output(imo, df_data)
        markdown_answer = markdown(output_dict, df, imo)

        question_number = '36'
        query_app_answers = {'imo': imo}
        update_app_answers = {
            '$set': {
                f'data.{question_number}.value': output_dict,
                f'data.{question_number}.questionName': 'MECC and AECC Consumption and ROB from Ship Palm consumption log',
                f'data.{question_number}.updatedAt':datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
            }
        }
        derived_params.update_one(query_app_answers, update_app_answers, upsert=True)

        new_row = pd.DataFrame({
            "imo": [imo],
            "markdown_answer": [markdown_answer]
        })
        output_table_1 = pd.concat([output_table_1, new_row], ignore_index=True)

    except:
        error_message = traceback.format_exc()
        failed[str(imo)] = error_message
    
no_data_df = output_table_1[output_table_1['markdown_answer'] == "Data Not Available"]

for imo in list(no_data_df['imo'].unique()):
    failed[str(imo)] = "Data not available"


output_table_1.reset_index(drop=True, inplace=True)
output_table_1 = pd.merge(output_table_1, vessel_id_df, left_on="imo", right_on="imo", how="left")


# fetching active vessels:
input_df = output_table_1[output_table_1["markdown_answer"] != "Data Not Available"]

valid_imos = list(input_df["imo"].unique())
 
for imo in tqdm(valid_imos):
    df_mecc = df_data[df_data["IMO No"] == imo]
    df_mecc.drop_duplicates(inplace = True)
    df_mecc.dropna(subset = "Report Date", inplace = True)
    df_mecc.sort_values("Report Date", inplace = True, ascending = True)
    latest_6_months, rob_trend_df = plots_data(df_mecc)

    plot_1_json = plot_mecc_consumption_grouped(latest_6_months[["Month","MECC consumption (LTRS)","Steaming time (HRS)"]])
    plot_2_json = plot_aecc_consumption_grouped(latest_6_months[["Month","AECC consumption (LTRS)","Steaming time (HRS)"]])
    plot_3_json = plot_rob(rob_trend_df)

                    
    #push to mongodb:
    vesselName = input_df[input_df["imo"] == imo]["name"].values[0]
    
    # vesselinfocomponents:
    vesselinfocomponents.update_one({ "questionNo": 36, "componentNo": f"1_36_{imo}"}, {'$set': {"data": plot_1_json, "imo": int(imo),"componentName":"Plotly",'refreshDate' : datetime.now(timezone.utc)}}, upsert = True)
    vesselinfocomponents.update_one({ "questionNo": 36, "componentNo": f"2_36_{imo}"}, {'$set': {"data": plot_2_json, "imo": int(imo),"componentName":"Plotly",'refreshDate' : datetime.now(timezone.utc)}}, upsert = True)
    vesselinfocomponents.update_one({ "questionNo": 36, "componentNo": f"3_36_{imo}"}, {'$set': {"data": plot_3_json, "imo": int(imo),"componentName":"Plotly",'refreshDate' : datetime.now(timezone.utc)}}, upsert = True)

    vesselinfocomponents_synergy.update_one({"componentNo": f"1_36_{imo}", "questionNo": 36},
                                             {"$set": {"data": plot_1_json,
                                              "componentName": "Plotly",
                                              "imo": int(imo),
                                              "refreshDate": datetime.now(timezone.utc)}}, upsert = True)
    vesselinfocomponents_synergy.update_one({"componentNo": f"2_36_{imo}", "questionNo": 36},
                                             {"$set": {"data": plot_2_json,
                                              "componentName": "Plotly",
                                              "imo": int(imo),
                                              "refreshDate": datetime.now(timezone.utc)}}, upsert = True)
    vesselinfocomponents_synergy.update_one({"componentNo": f"3_36_{imo}", "questionNo": 36},
                                             {"$set": {"data": plot_3_json,
                                              "componentName": "Plotly",
                                              "imo": int(imo),
                                              "refreshDate": datetime.now(timezone.utc)}}, upsert = True)
                                              
                                              
    # vesselinfos:
    links = f'''\n\nhttpsdev.syia.ai/chat/Plotlycharts?component=1_36\n\nhttpsdev.syia.ai/chat/Plotlycharts?component=2_36\n\nhttpsdev.syia.ai/chat/Plotlycharts?component=3_36\n\n''' 
    new_answer = output_table_1[output_table_1["imo"] == imo]["markdown_answer"].values[0]
    dev_answer = new_answer + links
    response = vesselinfo.update_one({ "questionNo": 36,"imo": int(imo)}, {'$set': {"answer": dev_answer, 
                                                                        "question" : "MECC and AECC Consumption and ROB from Ship Palm consumption log",
                                                                        "vesselName": vesselName,
                                                                        "detailedAnswer": "",
                                                                        'refreshDate' : datetime.now(timezone.utc)}}, upsert = True)
    
    response_synergy = vesselinfo_synergy.update_one({"questionNo" : 36, "imo" : imo},{"$set" : {"answer": dev_answer, 
                                                                        "question" : "MECC and AECC Consumption and ROB from Ship Palm consumption log",
                                                                        "vesselName": vesselName,
                                                                        "detailedAnswer": "",
                                                                        'refreshDate' : datetime.now(timezone.utc)}}, upsert = True)
    if not (response.upserted_id or response.raw_result['updatedExisting']):
        failed[str(imo)] = "Unable to insert in mongodb"


# Updating failed vessels to mongodb:
mongo_client = MongoClient("mongodb://etl:rSp6X49ScvkDpHE@db.syia.ai:27017/?authMechanism=DEFAULT&authSource=syia-etl&directConnection=true")
etl = mongo_client['syia-etl']
app_answers_failed = etl['app_answers_failed']

app_answers_failed.insert_one({"questionId" : ObjectId("64336dbf3ad8204397b1f95e"),
"questionNo" : 36,
"countActiveVessel": len(active_imos),
"successfulCount": len(active_imos) - len(failed),
"failedCount" : len(failed),
"failed" : failed,   #use imo as key in failed
"platform" : "beta",
"updatedAt" : datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)})


#collection_app_answers from siya-etl-dev:
# empty_update = [el["imo"] for el in failed if el["error"] == "Data not available"]
empty_update = [el for el in failed if failed[el] == "Data not available"]
for imo in tqdm(empty_update):
    imo = int(imo)
    vesselName = vessel_id_df[vessel_id_df["imo"] == imo]["name"].values[0]
    #dev:
    vesselinfo.update_one({ "questionNo": 36,"imo": int(imo)}, {'$set': {"answer":"## MECC and AECC Consumption and ROB from Ship Palm consumption log \n\n> Data Not Available",
                                                                        "question" : "MECC and AECC Consumption and ROB from Ship Palm consumption log",
                                                                        "vesselName": vesselName,
                                                                        "detailedAnswer": "",
                                                                        'refreshDate' : datetime.now()}}, upsert = True)
    vesselinfo_synergy.update_one({"questionNo" : 36, "imo" : imo},{"$set" : {"answer":"## MECC and AECC Consumption and ROB from Ship Palm consumption log \n\n> Data Not Available",
                                                                        "question" : "MECC and AECC Consumption and ROB from Ship Palm consumption log",
                                                                        "vesselName": vesselName,
                                                                        "detailedAnswer": "",
                                                                        'refreshDate' : datetime.now()}}, upsert = True)