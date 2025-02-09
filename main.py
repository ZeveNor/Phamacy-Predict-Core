import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pydantic import BaseModel
from typing import List
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from datetime import datetime
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing purposes (change to specific origins for security)
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
sheet = client.open_by_key(SHEET_ID).sheet1 
PREDICT_ID = '13_f_4pEUgCifgGYRzME0eFNRVrt_vGMKEcygE-MrFJ8'
 

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

@app.post("/update-sales/")
async def update_sales(data: List[SalesData]):
    new_rows = [[
        entry.invoice_no, entry.sale_date, entry.sale_time, entry.customer_id,
        entry.drug_code, entry.product_name, entry.batch_no, entry.storage_location,
        entry.quantity, entry.unit, entry.unit_price, entry.total_amount, entry.gross_profit
    ] for entry in data]

    sheet.append_rows(new_rows, value_input_option="RAW")
    
    return {"message": "Sales data updated in Google Sheets!", "rows_added": len(new_rows)}

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

@app.get("/get-all-medicines/")
async def get_all_medicines():
    all_rows = sheet.get_all_records()

    unique_medicines = { (row['product_name'], row['drug_code']) for row in all_rows }

    medicines_list = [{"product_name": name, "drug_code": code} for name, code in unique_medicines]

    return {"medicines": medicines_list}

@app.get("/get-medicine-by-id/{drug_code}")
async def get_medicine_by_id(drug_code: str):
    all_rows = sheet.get_all_records()
 
    for row in all_rows:
        if row['drug_code'] == drug_code:
            return {"product_name": row['product_name'], "drug_code": row['drug_code']}
    
    raise HTTPException(status_code=404, detail="Medicine not found")


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

def updateToGSheetFromCSV(csv_file_path):
    datacleaned_url = "https://docs.google.com/spreadsheets/d/1eohhRmXeB-e7xpoacjfbRIA2Igz8GP7A1Sbe3ekqAdQ/edit?gid=0#gid=0"
    datacleaned_spreadsheet = client.open_by_url(datacleaned_url)
    datacleaned_worksheet = datacleaned_spreadsheet.get_worksheet(0)
    
    data_from_csv = pd.read_csv(csv_file_path)
    data_from_csv = data_from_csv.fillna('')
    datacleaned_worksheet.clear()
    datacleaned_worksheet.update('A1', [data_from_csv.columns.tolist()] + data_from_csv.values.tolist())
    
    print("Data successfully uploaded to Google Sheets.")

def create_sequence(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][['quantity', 'อุณหภูมิ', 'ปริมาณน้ำฝน']].values)
        y.append(data.iloc[i+seq_length]['quantity'])
    return np.array(X), np.array(y)


@app.get("/predict/{name}")
async def predict(name: str):    
    spreadsheet_url = SHEET_ID
    temp_url = "https://docs.google.com/spreadsheets/d/1vnxBib84T24Hq0BfzWJQobDFaNnmb2A6s36TbLEqrFY/edit?gid=0#gid=0"
    holiday_url = "https://docs.google.com/spreadsheets/d/1EZvUyp8vKcgSDke7iclm9b54Sw2JxCG6VzLvGq93XpU/edit?gid=1209652816#gid=1209652816"

    data_cleaned = client.open_by_key(spreadsheet_url)
    temp = client.open_by_url(temp_url)
    holiday = client.open_by_url(holiday_url)

    worksheet = data_cleaned.get_worksheet(0)
    tempsheet = temp.get_worksheet(0)
    holiday = holiday.get_worksheet(0)

    data = worksheet.get_all_records()
    tdata = tempsheet.get_all_records()
    hdata = holiday.get_all_records()

    train_data = pd.DataFrame(data)
    temp_df = pd.DataFrame(tdata)
    holiday_df = pd.DataFrame(hdata)

    train_data['sale_date'] = pd.to_datetime(train_data['sale_date'], errors='coerce')

    if train_data['sale_date'].isnull().sum() > 0:
        print("Warning: Some dates could not be converted. Check for incorrect formats.")

    train_data['ปี'] = train_data['sale_date'].dt.strftime('%Y')
    train_data['เดือน'] = train_data['sale_date'].dt.strftime('%m')
    train_data['เดือน-ปี'] = train_data['sale_date'].dt.strftime('%m-%Y')
    train_data['เดือน-วัน'] = train_data['sale_date'].dt.strftime('%m-%d')


    train_data = pd.merge(train_data, holiday_df, how='left', left_on='เดือน-วัน', right_on='วันที่')
    train_data['holiday'] = train_data['วันสำคัญ'].notnull()
    train_data = pd.merge(train_data, temp_df, how='left', left_on='เดือน-ปี', right_on='เดือนอุณหภูมิ')

    rainfall_data = {}
    for year in range(2019, 2025):
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

    province = "นครราชสีมา"

    train_data['ปริมาณน้ำฝน'] = train_data.apply(lambda row: rainfall_data.get(int(row['ปี']), {}).get(province, {}).get(row['เดือน'], 0.0), axis=1)
    train_data.drop(columns=['เดือน', 'ปี'], inplace=True)
    train_data = train_data.drop(columns=['เดือนอุณหภูมิ'])
    train_data = train_data.drop(columns=['เดือน-ปี'])
    train_data = train_data.drop(columns=['เดือน-วัน'])
    train_data = train_data.drop(columns=['วันที่'])
    train_data.to_csv("cleaned_train_data.csv", index=False)

    # train_data.to_csv("cleaned_train_data.csv", index=False)

    updateToGSheetFromCSV("cleaned_train_data.csv")

    df = train_data
    df['sale_date'] = pd.to_datetime(df['sale_date'], format='%Y-%m-%d')

    df_grouped = df.groupby(['sale_date', 'drug_code'], as_index=False)['quantity'].sum()
    df_grouped = df_grouped.merge(df[['sale_date', 'holiday', 'อุณหภูมิ', 'ปริมาณน้ำฝน']].drop_duplicates(), on='sale_date', how='left')
    df_grouped = df_grouped.sort_values(by='sale_date')

    # df_grouped.to_csv('processed_sales_data.csv', index=False)

    print("Completed datacleaned files")

    df = df_grouped
    df['วันที่ขาย'] = pd.to_datetime(df['sale_date'])
    df = df.sort_values(by='sale_date')

    df_drug = df[df['drug_code'] == name]

    df_drug = df_drug.copy() 

    scaler = MinMaxScaler()
    df_drug.loc[:, ['quantity', 'อุณหภูมิ', 'ปริมาณน้ำฝน']] = scaler.fit_transform(df_drug[['quantity', 'อุณหภูมิ', 'ปริมาณน้ำฝน']])

    X, y = create_sequence(df_drug, seq_length=30)
    X_train, y_train = X, y

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=50, return_sequences=False),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=8)

    last_30_days = df_drug.tail(30)[['quantity', 'อุณหภูมิ', 'ปริมาณน้ำฝน']].values

    predictions_2021 = []
    for i in range(365):
        last_30_days_reshaped = np.reshape(last_30_days, (1, last_30_days.shape[0], last_30_days.shape[1]))
        predicted_quantity = model.predict(last_30_days_reshaped)
        predictions_2021.append(predicted_quantity[0][0])
        last_30_days = np.vstack([last_30_days[1:], np.array([predicted_quantity[0][0], last_30_days[0, 1], last_30_days[0, 2]])])

    predictions_2021 = scaler.inverse_transform(np.concatenate((np.array(predictions_2021).reshape(-1, 1), np.zeros((len(predictions_2021), 2))), axis=1))[:, 0]
    dates_2021 = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')
    predictions_df = pd.DataFrame({'date': dates_2021, 'predicted_quantity': predictions_2021})

    predictions_df.to_csv('predictions_2021_A0158.csv', index=False)
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.strftime('%Y-%m-%d')

    predict_id = "13_f_4pEUgCifgGYRzME0eFNRVrt_vGMKEcygE-MrFJ8"
    predictsheet = client.open_by_key(predict_id)

    try:
        sheet = predictsheet.worksheet(name)
        predictsheet.del_worksheet(sheet) 
    except gspread.WorksheetNotFound:
        pass  

    sheet = predictsheet.add_worksheet(title=name, rows=str(len(predictions_df)+1), cols="2")
    sheet.update('A1', [['date', 'predicted_quantity']])
    sheet.update('A2', predictions_df.values.tolist())

    print(f"Predictions for {name} saved to Google Sheets successfully!")