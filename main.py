import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pydantic import BaseModel
from typing import List
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from datetime import datetime
import math
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

app = FastAPI()

origins = [
    "http://127.0.0.1:3000",  # Allow your front-end URL
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Sheets Authentication
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet
SHEET_ID = "1g7uVhRn1czV_kSbaXmjwr-88sxjahTDigl9htQ0xoBw"
DATA_CLEANED = "1eohhRmXeB-e7xpoacjfbRIA2Igz8GP7A1Sbe3ekqAdQ"
TEMP_URL = "1vnxBib84T24Hq0BfzWJQobDFaNnmb2A6s36TbLEqrFY"
HOLIDAY_URL = "1EZvUyp8vKcgSDke7iclm9b54Sw2JxCG6VzLvGq93XpU"
PREDICT_ID = '13_f_4pEUgCifgGYRzME0eFNRVrt_vGMKEcygE-MrFJ8'
STOCK_ID = '1-B04126To0UBwG7d-tEmIL2P97DrtnsfvgsMtElFcKk'
BOX_ID = '1cKJet97BXNoePxXzvO3cfny_1TwewqCLvkmijiMdnH0'
 
sheet = client.open_by_key(SHEET_ID).sheet1 

# Define the data model for SalesData
class SalesData(BaseModel):
    invoice_no: str
    sale_date: str
    sale_time: str
    customer_id: str
    drug_code: str
    product_name: str
    batch_no: str
    storage_location: str
    quantity: int
    unit: str
    unit_price: float
    total_amount: float
    gross_profit: float

# For adding new Transaction to All_data Google Sheets.
# https://docs.google.com/spreadsheets/d/1g7uVhRn1czV_kSbaXmjwr-88sxjahTDigl9htQ0xoBw/
@app.post("/update-sales/")
async def update_sales(data: List[SalesData]):
    new_rows = [[
        entry.invoice_no, entry.sale_date, entry.sale_time, entry.customer_id,
        entry.drug_code, entry.product_name, entry.batch_no, entry.storage_location,
        entry.quantity, entry.unit, entry.unit_price, entry.total_amount, entry.gross_profit
    ] for entry in data]

    sheet.append_rows(new_rows, value_input_option="RAW")
    return {"message": "Sales data updated in Google Sheets!", "rows_added": len(new_rows)}

#  API For get all transaction that we have in All_data Google sheets.
# https://docs.google.com/spreadsheets/d/1g7uVhRn1czV_kSbaXmjwr-88sxjahTDigl9htQ0xoBw/
@app.get("/get-all-sales/")
async def get_all_sales():
    data = sheet.get_all_values()
    sales_data = []
    headers = data[0] 
    for row in data[1:]:
        sales_data.append({
            headers[i]: row[i] for i in range(len(headers))
        })
    return {"data": sales_data}

#  API For get all stock status in PriceAndStock Google sheets.
# https://docs.google.com/spreadsheets/d/1-B04126To0UBwG7d-tEmIL2P97DrtnsfvgsMtElFcKk/
@app.get("/get-all-stocks/")
async def get_all_stocks():
    stock = client.open_by_key(STOCK_ID).sheet1 
    data = stock.get_all_values()

    stocks_data = []
    headers = data[0] 
    for row in data[1:]:
        stocks_data.append({
            headers[i]: row[i] for i in range(len(headers))
        })
    return {"data": stocks_data}

#  API For get prediction by filter with drug_code for predict page in Predict Google sheets.
# https://docs.google.com/spreadsheets/d/13_f_4pEUgCifgGYRzME0eFNRVrt_vGMKEcygE-MrFJ8/
@app.get("/get-all-prediction/{drug_code}")
async def get_all_prediction(drug_code: str):
    predict_sheet = client.open_by_key(PREDICT_ID)
    product_sheet = predict_sheet.worksheet(drug_code)

    data = product_sheet.get_all_values()
    predict_data = []
    headers = data[0] 
    for row in data[1:]:
        predict_data.append({
            headers[i]: row[i] for i in range(len(headers))
        })
    return {"data": predict_data}

#  API For get all stock for stock page in PriceAndStock Google sheets.
# https://docs.google.com/spreadsheets/d/1-B04126To0UBwG7d-tEmIL2P97DrtnsfvgsMtElFcKk/
@app.get("/get-stock/{drug_code}/")
async def get_stock(drug_code: str):
    stock = client.open_by_key(STOCK_ID).sheet1 
    data = stock.get_all_values()

    headers = data[0]
    drug_code_index = headers.index('drug_code')
    amount_index = headers.index('amount')
    price_index = headers.index('price')

    for row in data[1:]:
        if row[drug_code_index] == drug_code:
            return {
                'drug_code': row[drug_code_index],
                'amount': row[amount_index],
                'price': row[price_index]
            }

    return {"error": "Drug code not found"}

#  API For get all per-box details in Product Box Google Sheets.
# https://docs.google.com/spreadsheets/d/1cKJet97BXNoePxXzvO3cfny_1TwewqCLvkmijiMdnH0/
@app.get("/get-box/{drug_code}/{real_value}/{predict_value}")
async def get_box(drug_code: str,real_value: str,predict_value: str):
    stock = client.open_by_key(BOX_ID).sheet1 
    data = stock.get_all_values()

    headers = data[0]
    drug_code_index = headers.index('drug_code')
    per_box_index = headers.index('per_box')
    drug_name_index = headers.index('drug_name')
    unit_index = headers.index('product_unit')

    caltotal = int(real_value) - int(predict_value)
    needed_boxes = 0

    for row in data[1:]:
        if row[drug_code_index] == drug_code:
            if (caltotal <= 0):
                needed_boxes = math.ceil(abs(caltotal) / int(row[per_box_index]))
            return {
                'drug_code': row[drug_code_index],
                'name': row[drug_name_index],
                'per_box': row[per_box_index],
                'need': needed_boxes,
                'unit': row[unit_index]
            }

    return {"error": "Drug code not found"}

#  API For get all medicine for first page in All_data Google sheets.
# https://docs.google.com/spreadsheets/d/1g7uVhRn1czV_kSbaXmjwr-88sxjahTDigl9htQ0xoBw/
@app.get("/get-all-medicines/")
async def get_all_medicines():
    all_rows = sheet.get_all_records()
    count = 0
    unique_medicines = {}
    for row in all_rows:
        drug_code = row["drug_code"]
        product_name = row["product_name"]

        if drug_code not in unique_medicines:
            unique_medicines[drug_code] = product_name
            count = count + 1  

    medicines_list = sorted(
        [{"drug_code": code, "product_name": name} for code, name in unique_medicines.items()],
        key=lambda x: x["drug_code"]
    )
    return {"medicines": medicines_list,"count": count}

#  API For get medicine details by filter using drug_code for first page in All_data Google sheets.
# https://docs.google.com/spreadsheets/d/1g7uVhRn1czV_kSbaXmjwr-88sxjahTDigl9htQ0xoBw/
@app.get("/get-medicine-by-id/{drug_code}")
async def get_medicine_by_id(drug_code: str):
    all_rows = sheet.get_all_records()
 
    for row in all_rows:
        if row['drug_code'] == drug_code:
            return {"product_name": row['product_name'], "drug_code": row['drug_code']}
    
    raise HTTPException(status_code=404, detail="Medicine not found")

# using in stock page.
# 
@app.get("/get-predictions-stock")
async def get_predictions():
    predict_sheet = client.open_by_key(PREDICT_ID)
    product_ids = [
        "A0158", "A0160", "A0167", "A0175","A0178" ,"B0071", "B0086", "C0142", "C0173", "C0185", "C0196",
        "D0104", "D0160", "D0167", "D0174", "D0179", "D0189", "E0055", "F0083", "F0096", "G0066", "G0074", 
        "G0090", "G0091", "G0095", "H0038", "I0108", "O0045", "S0125"
    ]
    
    one_month_predictions = {}
    manual_predictions = {}

    sheets = {ws.title: ws for ws in predict_sheet.worksheets()}
    
    for product_id in product_ids:
        try:
            product_sheet = sheets.get(product_id)
            data = product_sheet.get_all_values()
            df = pd.DataFrame(data[13:], columns=data[0])  
            df['predicted_quantity'] = pd.to_numeric(df['predicted_quantity'], errors='coerce')

            one_month_predictions[product_id] = int(df.iloc[:1]['predicted_quantity'])
            manual_predictions[product_id] = df.iloc[:12]['predicted_quantity'].tolist() if df.iloc[:12]['predicted_quantity'].any() else [0] * days
        except gspread.exceptions.WorksheetNotFound:
            one_month_predictions[product_id] = 0
            manual_predictions[product_id] = [0] * 7
    
    return {"one_month_predictions": one_month_predictions, "predictions": manual_predictions}

@app.get("/get-monthly-predictions/{year_month}/{product_ID}")
async def get_monthly_predictions(year_month: str, product_ID: str):
    
    predict_sheet = client.open_by_key(PREDICT_ID)
    
    try:
        product_sheet = predict_sheet.worksheet(product_ID) 
    except gspread.exceptions.WorksheetNotFound:
        return {"error": f"Sheet for product ID '{product_ID}' not found"}
    
    
    data = product_sheet.get_all_values()
    
    df = pd.DataFrame(data[1:], columns=data[0])  
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['predicted_quantity'] = pd.to_numeric(df['predicted_quantity'], errors='coerce')

    start_date = datetime.strptime(year_month + '-01', '%Y-%m-%d')
    end_date = (start_date.replace(day=28) + pd.DateOffset(days=4)).replace(day=1) - pd.DateOffset(days=1)

    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    predictions = filtered_df[['date', 'predicted_quantity']].to_dict(orient='records')

    return {"predictions": predictions}

@app.get("/get-yearly-quantity/{year}/{product_ID}")
async def get_yearly_quantity(year: int, product_ID: str):
    predict_sheet = client.open_by_key(PREDICT_ID)
    try:
        product_sheet = predict_sheet.worksheet(product_ID) 
    except gspread.exceptions.WorksheetNotFound:
        return {"error": f"Sheet for product ID '{product_ID}' not found"}

    data = product_sheet.get_all_values()

    df = pd.DataFrame(data, columns=["date", "predicted_quantity"])

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['predicted_quantity'] = pd.to_numeric(df['predicted_quantity'], errors='coerce')

    df = df[df['date'].dt.year == year]

    df['month'] = df['date'].dt.month

    monthly_summary = df.groupby('month')['predicted_quantity'].sum().reset_index()

    predictions = [
        {"month": row['month'], "total_quantity": row['predicted_quantity']}
        for _, row in monthly_summary.iterrows()
    ]

    return {"product_ID": product_ID, "year": year, "monthly_quantities": predictions}


# 
# 
# 
# 
# ------------------------------------------------ #
# LSTM FUNCTION

# Save data to DATACLEANED google sheets
def updateToGSheetFromCSV(csv_file_path):
    dataclean_sheet = client.open_by_key(DATA_CLEANED)
    ws = dataclean_sheet.get_worksheet(0)
    
    data_from_csv = pd.read_csv(csv_file_path)
    data_from_csv = data_from_csv.fillna('')

    ws.clear()
    ws.update('A1', [data_from_csv.columns.tolist()] + data_from_csv.values.tolist())
    print("Data successfully uploaded to Google Sheets.")

# Get rainfall from TMD api for training
def get_rainfall_api(rangestart,rangeend,province):
    rainfall_data = {}
    for year in range(rangestart, rangeend):
        url = f"https://data.tmd.go.th/api/ThailandMonthlyRainfall/v1/index.php?uid=api&ukey=api12345&format=json&year={year}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            yearly_rainfall = {}
            for station in data.get('StationMonthlyRainfall', []):
                province = station['StationNameThai']
                rainfall = station['MonthlyRainfall']
                yearly_rainfall[province] = {str(i+1).zfill(2): float(rainfall[f'Rainfall{month}']) for i, month in enumerate(
                    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
                )}
            rainfall_data[year] = yearly_rainfall
        else:
            print(f"Failed to fetch rainfall data for {year}")

    return rainfall_data

# LSTM Sequence config.
def create_sequence(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][['quantity', 'อุณหภูมิ', 'ปริมาณน้ำฝน']].values)
        y.append(data.iloc[i+seq_length]['quantity'])
    return np.array(X), np.array(y)

# LSTM Hidden Layer Structure.
def lstm_layer(epoch,batch_size):
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(units=100, return_sequences=True),
        LSTM(units=50, return_sequences=False),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epoch, batch_size=batch_size, verbose=1)
    return model

# Save data to Predict Google sheets
def updatePredictToGSheet(predictions_df):
    predict_sheet = client.open_by_key(PREDICT_ID)

    try:
        ws = predict_sheet.worksheet(name)
        predict_sheet.del_worksheet(sheet)  
    except gspread.WorksheetNotFound:
        pass  

    ps = predict_sheet.add_worksheet(title=name, rows=str(len(predictions_df) + 1), cols="2")
    ps.update('A1', [['date', 'predicted_quantity']])
    ps.update('A2', predictions_df.values.tolist())

@app.get("/predict/{name}")
async def predict(name: str):
    # Open All google sheet to get data.
    data_cleaned = client.open_by_key(SHEET_ID)
    temp = client.open_by_key(TEMP_URL)
    worksheet = data_cleaned.get_worksheet(0)
    tempsheet = temp.get_worksheet(0)
    data = worksheet.get_all_records()
    tdata = tempsheet.get_all_records()
    train_data = pd.DataFrame(data)
    temp_df = pd.DataFrame(tdata)

    # Prepare Date Format for matching with outsource data. Easier for config 
    train_data['sale_date'] = pd.to_datetime(train_data['sale_date'], errors='coerce')
    train_data['ปี'] = train_data['sale_date'].dt.strftime('%Y')
    train_data['เดือน'] = train_data['sale_date'].dt.strftime('%m')
    train_data['เดือน-ปี'] = train_data['sale_date'].dt.strftime('%m-%Y')
    train_data['เดือน-วัน'] = train_data['sale_date'].dt.strftime('%m-%d')

    # Merge Temp to main table by using date format mm-yyyy
    train_data = pd.merge(train_data, temp_df, how='left', left_on='เดือน-ปี', right_on='เดือนอุณหภูมิ')

    # Get Rainfall Data
    province = "นครราชสีมา"
    get_rainfall_api(2019,2025,province)
    train_data['ปริมาณน้ำฝน'] = train_data.apply(lambda row: rainfall_data.get(int(row['ปี']), {}).get(province, {}).get(row['เดือน'], 0.0), axis=1)

    # Drop All column not in use like Date format.
    train_data.drop(columns=['เดือน', 'ปี'], inplace=True)
    train_data = train_data.drop(columns=['เดือนอุณหภูมิ'])
    train_data = train_data.drop(columns=['เดือน-ปี'])
    train_data = train_data.drop(columns=['เดือน-วัน'])
    train_data = train_data.drop(columns=['วันที่'])

    # Save some file for future check logs.
    train_data.to_csv("cleaned_train_data.csv", index=False)

    # Function to upload trained data to Google Sheet.
    updateToGSheetFromCSV("cleaned_train_data.csv")

    # ----------------------------------------------------------------- #
    # grouped data steps. 

    # group all duplicate drugs that have saled in each day to only one transaction per day.
    df = train_data
    df['sale_date'] = pd.to_datetime(df['sale_date'], format='%Y-%m-%d')
    df_grouped = df.groupby(['sale_date', 'drug_code'], as_index=False)['quantity'].sum()
    df_grouped = df_grouped.merge(df[['sale_date','อุณหภูมิ', 'ปริมาณน้ำฝน']].drop_duplicates(), on='sale_date', how='left')
    df_grouped = df_grouped.sort_values(by='sale_date')
    df = df_grouped

    # ----------------------------------------------------------------- #
    # Start of train steps.

    # Sort data by date before train.
    df['วันที่ขาย'] = pd.to_datetime(df['sale_date'])
    df = df.sort_values(by='sale_date')

    # Filter only specific drug_code.
    df_drug = df[df['drug_code'] == name].copy()
    df_drug['year_month'] = df_drug['sale_date'].dt.to_period('M')

    # Reset and Regroup all data again
    df_monthly = df_drug.groupby('year_month').agg({
        'quantity': 'sum',
        'อุณหภูมิ': 'mean',
        'ปริมาณน้ำฝน': 'mean'
    }).reset_index()

    # group data to month for predict in month from 2020 - ....
    df_monthly['year_month'] = df_monthly['year_month'].astype(str)

    # Scale Down data to format between 0 - 1. Lstm need this format for analyze
    scaler_quantity = MinMaxScaler()
    df_monthly['quantity'] = scaler_quantity.fit_transform(df_monthly[['quantity']])
    df_monthly[['อุณหภูมิ', 'ปริมาณน้ำฝน']] = scaler_features.fit_transform(df_monthly[['อุณหภูมิ', 'ปริมาณน้ำฝน']])

    # Set Analyze range to 2 years. Let model focus on 2 year.
    seq_length = 24

    # Call function create sequence.
    X, y = create_sequence(df_monthly, seq_length)

    # Call function LSTM Layer Structure. to start train data.
    model = lstm_layer(300,16)

    # ----------------------------------------------------------------- #
    # Start of Predicting Month steps.

    # get last month from grouped data.
    last_months = df_monthly.tail(seq_length)[['quantity', 'อุณหภูมิ', 'ปริมาณน้ำฝน']].values

    # how many month u want to predict
    num_months = 24  

    # Loop to predict from 1 to num_month. let say it 24 month or 2 years.
    predictions = []
    for i in range(num_months):
        last_months_reshaped = np.reshape(last_months, (1, last_months.shape[0], last_months.shape[1]))
        predicted_quantity = abs(model.predict(last_months_reshaped)[0][0])
        
        # mix with new data to let model can catch the point of change.
        if i < 12:
            temp_rain = df_monthly.iloc[-12 + i][['อุณหภูมิ', 'ปริมาณน้ำฝน']].values
        else:
            temp_rain = df_monthly[['อุณหภูมิ', 'ปริมาณน้ำฝน']].mean().values  

        # add predicted result to array before sent to google sheet to store data.
        predictions.append(predicted_quantity)
        last_months = np.vstack([last_months[1:], [predicted_quantity, temp_rain[0], temp_rain[1]]])

    predictions = scaler_quantity.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Format Date before upload to Google sheets.
    dates = pd.date_range(start='2020-01', periods=num_months, freq='M')
    predictions_df = pd.DataFrame({'date': dates, 'predicted_quantity': predictions})
    predictions_df['date'] = predictions_df['date'].dt.strftime('%Y-%m')

    # Save some file to Local Server for future log.
    predictions_df.to_csv(f'predictions_{name}_2020_2021_monthly.csv', index=False)

    # Save to Predict Google sheet
    updatePredictToGSheet(predictions_df)
    print(f"Predictions for {name} saved to Google Sheets successfully!")
    # END Train Process