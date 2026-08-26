import os
import pandas as pd

def get_metadata(image_path):

    filename = os.path.basename(image_path)
    image_no = int(os.path.splitext(filename)[0])

    df = pd.read_excel(r"Dataset/image_keys.xlsx", header=1)

    result = df[df["Image No"] == image_no][["Case ID", "Region"]]

    if not result.empty:
        case_id = result.iloc[0]["Case ID"]
        region = result.iloc[0]["Region"]
        return str(image_no), case_id, region
    else:
        return str(image_no), "UNKNOWN", "UNKNOWN"



def get_image_metadata_str(image_path) -> str:

    filename = os.path.basename(image_path)
    image_number = int(os.path.splitext(filename)[0])

    df = pd.read_csv(r"Dataset/RatedPathology(Sheet1).csv")

    row = df[df["Image ID"] == image_number]
    if row.empty:
        return f"<b>No metadata found for image {image_number}.</b>"

    row_data = (row.iloc[0]
    .replace("", "Not Available")
    .fillna("Not Available")
    .to_dict()
    )   

    # Build properly styled HTML table
    html = """
    <html>
    <head>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 12px;
        }
        th, td {
            border: 1px solid #888;
            padding: 4px 6px;
            text-align: left;
        }
        th {
            background-color: #2c3e50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    </head>
    <body>
    <table>
    <tr><th>Field</th><th>Value</th></tr>
    """

    for key, value in row_data.items():
        html += f"<tr><td>{key}</td><td>{value}</td></tr>"

    html += "</table></body></html>"
    return html


